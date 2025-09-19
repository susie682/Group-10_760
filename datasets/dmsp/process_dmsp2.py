# -*- coding: utf-8 -*-
"""
DMSP SSUSI EDR-Aurora nearest-point summarizer @ Poker Flat
- Read each NetCDF file
- Convert geographic coordinates (PF_LAT, PF_LON) to AACGM (magnetic coordinates) (recommended altitude: 110 km);
- On north/south hemisphere grids, search for the nearest N points (prefer within threshold, otherwise globally nearest)
- Compute mean/median/max, and aggregate into 3-hour Kp slots
- After processing each file, persist results.json (safe for restart from checkpoint)

Main fixes (### FIX):
1) Southern hemisphere query now uses the same magnetic latitude (mag_lat) as northern hemisphere;
2) Statistics only use “valid values” (>0) to avoid being polluted by fill values/negatives; if insufficient valid points within threshold, automatically fall back to “global nearest”;
3) Configurable minimum valid points threshold MIN_VALID_POINTS, to prevent median=0 caused by only 1–2 valid points;
4) Added option CLAMP_NEGATIVE_TO_ZERO (default ON) to clip negatives to 0;
5) Keep original checkpointing JSON; final CSV only writes aggregated results with sorted keys.
"""

import os
import re
import json
import xarray as xr
import numpy as np
import aacgmv2
from datetime import datetime, timedelta
import pandas as pd
from tqdm import tqdm

# ------------------ Configurations ------------------
PF_LAT = 65.1          # Poker Flat latitude
PF_LON = -147.5        # Poker Flat longitude

### CHANGE 1: Use altitude closer to auroral emission layer, recommended 110 km (original was 0)
ALT_KM = 110           # Altitude for AACGM conversion (km); 0 also works but 110 km is more physical

N_POINTS = 10          # Number of nearest points
### CHANGE 2: Distance threshold can be loosened (150–250 km) to avoid all-zero cases in sparse data
DIST_THRESHOLD_KM = 150.0  # Distance threshold in km

# ### FIX: Require at least this many “valid points” (>0) to calculate statistics directly;
# otherwise fall back to global nearest to fill.
MIN_VALID_POINTS = 1  # Example counts: 2020: 1–136 nonzero, 2019: 1–306 nonzero.

# ### FIX: Whether to clamp negative values to 0 (to avoid noise/background corrections affecting stats)
CLAMP_NEGATIVE_TO_ZERO = True

DEBUG = False  # Toggle detailed print output

# ------------------ Utilities ------------------
def latgrid_for_hemi(lat_grid: np.ndarray, hemi_flag: int) -> np.ndarray:
    """
    Adjust latitude grid according to hemisphere:
    - If lat_grid is absolute values (0..90), mirror to negative for southern hemisphere;
    - If lat_grid already has sign (±), return as is.
    hemi_flag: 0 = North, 1 = South
    """
    lat_min = np.nanmin(lat_grid)
    lat_max = np.nanmax(lat_grid)
    is_absolute = (lat_min >= 0)
    if hemi_flag == 1 and is_absolute:
        return -lat_grid
    return lat_grid

def parse_filename_datetime(filename):
    """Parse datetime from filename pattern YYYYJJJThhmmss."""
    m = re.search(r"(\d{4})(\d{3})T(\d{2})(\d{2})(\d{2})", filename)
    if not m:
        return None
    year = int(m.group(1))
    day_of_year = int(m.group(2))
    hour = int(m.group(3))
    minute = int(m.group(4))
    second = int(m.group(5))
    return datetime(year, 1, 1) + timedelta(days=day_of_year - 1,
                                             hours=hour, minutes=minute, seconds=second)

def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance (km). Supports numpy arrays."""
    R = 6371.0
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad)*np.cos(lat2_rad)*np.sin(dlon/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

def pick(ds, *cands):
    """Pick the first available variable among candidate names."""
    for n in cands:
        if n in ds.variables:
            return ds[n].values
    raise KeyError(f"None of {cands} in file")

def select_hemisphere_data(ds, hemisphere="north"):
    """Return aurora data array, lat, lon, and UT grid by hemisphere."""
    if hemisphere.lower() == "north":
        aur_data = pick(ds, "AUR_ARC_RADIANCE_NORTH", "AUR_ARC_MEDIAN_RAD_NORTH")
        lat_grid = ds["LATITUDE_GEOMAGNETIC_GRID_MAP"].values
        lon_grid = ds["LONGITUDE_GEOMAGNETIC_NORTH_GRID_MAP"].values
        ut_grid = ds.get("UT_N", None)
    else:
        aur_data = pick(ds, "AUR_ARC_RADIANCE_SOUTH", "AUR_ARC_MEDIAN_RAD_SOUTH")
        lat_grid = ds["LATITUDE_GEOMAGNETIC_GRID_MAP"].values
        lon_grid = ds["LONGITUDE_GEOMAGNETIC_SOUTH_GRID_MAP"].values
        ut_grid = ds.get("UT_S", None)
    return aur_data, lat_grid, lon_grid, ut_grid

def query_nearest_points(lat, lon, aur_data, lat_grid, lon_grid, ut_grid=None,
                         n_points=10, max_distance_km=100.0, origin_flag=0):
    """
    Select up to n_points nearest non-NaN points within distance threshold.
    - Keep placeholder entries if insufficient points (for later fallback decision).
    - Return list of points with value, coordinates, UT, distance, and origin flag.
    """
    lat_flat = lat_grid.flatten()
    lon_flat = lon_grid.flatten()
    aur_flat = aur_data.flatten()

    valid_idx = np.where(~np.isnan(aur_flat) & (aur_flat != 0))[0]

    if ut_grid is not None:
        ut_flat = ut_grid.values.flatten() if hasattr(ut_grid, "values") else ut_grid.flatten()
        valid_idx = valid_idx[ut_flat[valid_idx] != 0]

    if len(valid_idx) == 0:
        return [{"value": 0, "lat": None, "lon": None, "ut": None,
                 "distance_km": None, "origin": 0} for _ in range(n_points)], None

    distances = haversine(lat, lon, lat_flat[valid_idx], lon_flat[valid_idx])
    nearest_idx_sorted = np.argsort(distances)
    results = []

    for i in range(n_points):
        if i < len(nearest_idx_sorted):
            idx = valid_idx[nearest_idx_sorted[i]]
            row, col = np.unravel_index(idx, lat_grid.shape)
            dist_km = haversine(lat, lon, lat_grid[row, col], lon_grid[row, col])
            if dist_km <= max_distance_km:
                val = aur_data[row, col]
                ut_val = ut_grid[row, col].item() if ut_grid is not None else None
            else:
                val = 0
                ut_val = None
        else:
            # Fill with zeros if fewer than n_points available
            val, ut_val, dist_km, row, col = 0, None, None, 0, 0

        results.append({
            "value": val,
            "lat": lat_grid[row, col] if val != 0 else None,
            "lon": lon_grid[row, col] if val != 0 else None,
            "ut": ut_val,
            "distance_km": dist_km if val != 0 else None,
            "origin": origin_flag
        })

    # Pick the first UT from nonzero values as representative
    nearest_ut = None
    for r in results:
        if r["value"] != 0:
            nearest_ut = r["ut"]
            break
    return results, nearest_ut

def query_nearest_points_all(lat, lon, aur_data, lat_grid, lon_grid, ut_grid=None,
                             n_points=10, origin_flag=0):
    """Global nearest n_points (no distance threshold)."""
    lat_flat = lat_grid.flatten()
    lon_flat = lon_grid.flatten()
    distances = haversine(lat, lon, lat_flat, lon_flat)
    nearest_idx_sorted = np.argsort(distances)

    results = []
    for i in range(min(n_points, nearest_idx_sorted.size)):
        idx = nearest_idx_sorted[i]
        row, col = np.unravel_index(idx, lat_grid.shape)
        val = aur_data[row, col]
        ut_val = ut_grid[row, col].item() if ut_grid is not None else None
        dist_km = distances[idx]
        results.append({
            "value": float(val),
            "lat": float(lat_grid[row, col]),
            "lon": float(lon_grid[row, col]),
            "ut": ut_val,
            "distance_km": float(dist_km),
            "origin": origin_flag
        })

    nearest_ut = results[0]["ut"] if results else None
    return results, nearest_ut

def map_to_kp_slot(ut_hours):
    """Map UT hours [0,24) to nearest 3-hour slot: 0,3,...,21."""
    slots = np.arange(0, 24, 1)
    diffs = np.abs(slots - ut_hours)
    idx = np.argmin(diffs)
    return int(slots[idx])

### CHANGE 3: New checkpointing: persist results instead of processed_slots
def load_results(json_path):
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            return json.load(f)
    return {}

def save_results(results, json_path):
    # Atomic replacement to avoid corruption if interrupted
    tmp = json_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f)
    os.replace(tmp, json_path)

def hours_to_utc(hours):
    """Convert fractional hours to hh:mm:ss UTC string."""
    if hours is None:
        return "N/A"
    h = int(hours)
    m = int((hours - h) * 60)
    s = int(((hours - h) * 60 - m) * 60)
    return f"{h:02d}:{m:02d}:{s:02d} UTC"

# ------------------ Core per-file processing ------------------

def process_file(filepath, results, results_json, dist_km=DIST_THRESHOLD_KM):
    """
    Process one NetCDF file:
    - Convert PF lat/lon to magnetic coords
    - Find nearest points (north/south hemisphere)
    - Compute mean/median/max stats
    - Update results dict
    - Persist checkpoint JSON after each file
    """
    filename = os.path.basename(filepath)
    ref_dt = parse_filename_datetime(filename)
    if not ref_dt:
        if DEBUG: print(f"Skip (cannot parse datetime): {filename}")
        return

    # Convert geographic to magnetic coordinates (at 110 km altitude)
    mag_lat, mag_lon, _ = aacgmv2.convert_latlon(PF_LAT, PF_LON, ALT_KM, dtime=ref_dt)
    mag_lon_360 = mag_lon % 360

    # Helper function to compute hemisphere stats
    def summarize(aur, latg, long, utg, hemi_flag):
        q_lat = mag_lat  # Observation point itself is not mirrored
        pts, ut = query_nearest_points(q_lat, mag_lon_360, aur, latg, long, utg,
                                       N_POINTS, dist_km, origin_flag=hemi_flag)
        
        # Fallback to global nearest if all values are zero
        if not any(p["value"] != 0 for p in pts):
            pts, ut = query_nearest_points_all(q_lat, mag_lon_360, aur, latg, long, utg,
                                               N_POINTS, origin_flag=hemi_flag)

        # Extract valid values
        vals = np.array([p["value"] for p in pts], dtype=float)
        if CLAMP_NEGATIVE_TO_ZERO:
            vals = np.clip(vals, 0, None)  # clamp negatives to 0
            
        nz = vals[vals > 0]
        # ### FIX: If insufficient valid values, fallback to global nearest
        if nz.size < MIN_VALID_POINTS:
            pts_all, ut_all = query_nearest_points_all(q_lat, mag_lon_360, aur, latg, long, utg,
                                                       N_POINTS, origin_flag=hemi_flag)
            if pts_all:
                vals = np.array([p["value"] for p in pts_all], dtype=float)
                if CLAMP_NEGATIVE_TO_ZERO:
                    vals = np.clip(vals, 0.0, None)
                nz = vals[vals > 0]
                ut = ut_all if ut_all is not None else ut

        # If still no valid values, use zero stats
        target = nz if nz.size > 0 else vals

        mean_val = float(np.mean(target))
        median_val = float(np.median(target)) if target.size else 0.0
        max_val = float(np.max(target)) if target.size else 0.0

        if DEBUG:
            pos, neg, zer = int((vals>0).sum()), int((vals<0).sum()), int((vals==0).sum())
            print(f"[{filename} hemi={hemi_flag}] pos/neg/zero={pos}/{neg}/{zer} "
                  f"mean={mean_val:.2f} median={median_val:.2f} max={max_val:.2f} "
                  f"UT={hours_to_utc(ut)}")

        return {"mean": mean_val, 
                "median": median_val, 
                "max": max_val, 
                "origin": hemi_flag, 
                "ut": ut}

    # Open dataset and process both hemispheres
    try:
        with xr.open_dataset(filepath, engine="netcdf4") as ds:
            aur_n, lat_n, lon_n, ut_n = select_hemisphere_data(ds, "north")
            north = summarize(aur_n, lat_n, lon_n, ut_n, hemi_flag=0)
            aur_s, lat_s, lon_s, ut_s = select_hemisphere_data(ds, "south")
            south = summarize(aur_s, lat_s, lon_s, ut_s, hemi_flag=1)
    except Exception as e:
        print(f"Failed to open/process {filename}: {e}")
        return

    # Choose hemisphere with larger max as representative
    best = north if north["max"] >= south["max"] else south

    # Fallback UT: if grid has no UT, use filename datetime
    ut_hours = (best["ut"] if best["ut"] is not None
                else ref_dt.hour + ref_dt.minute/60 + ref_dt.second/3600)
    slot = map_to_kp_slot(ut_hours)
    key = f"{ref_dt.strftime('%Y-%m-%d')}-{slot:02d}"

    # Slot-level aggregation: replace existing entry if candidate has larger max
    cur = results.get(key)
    cand = {"mean": round(best["mean"], 2),
            "median": round(best["median"], 2),
            "max": round(best["max"], 2),
            "origin": best["origin"]}

    if (cur is None) or (cand["max"] > cur["max"]):
        results[key] = cand

    # Save checkpoint after processing each file
    save_results(results, results_json)
        
    if DEBUG:
        print(f"Processed {filename} → {key} (max={results[key]['max']}, origin={results[key]['origin']})")

# ------------------ Main ------------------

def main():
    base_dir = "/Users/susie/Documents/UOA/Semester Two/COMPSCI 760/Group Project/code/Group-10_760/datasets/dmsp/data"
    year = 2020
    satellite = 17
    data_dir = os.path.join(base_dir, str(year), str(satellite))

    # Output directory
    curr_dir = os.getcwd()
    
    # Write to csv subdirectory
    csv_dir = os.path.join(curr_dir, "csv")
    os.makedirs(csv_dir, exist_ok=True)

    OUT_CSV = os.path.join(csv_dir, f"{year}-{satellite}-24.csv")
    RESULTS_JSON = os.path.join(csv_dir, f"{year}-{satellite}-results-24.json")

    files = sorted([f for f in os.listdir(data_dir) if f.endswith(".nc")])
    print(f"[INFO] scanning: {data_dir}")
    print(f"[INFO] found {len(files)} .nc files")
    if not files:
        print("[WARN] No .nc files found in directory; CSV will only have header. Please check download path/year/satellite.")

    # Restore previous results from checkpoint
    results = load_results(RESULTS_JSON)
    
    # Single tqdm loop (avoid double processing)
    for f in tqdm(files, desc="Processing files", unit="file"):
        filepath = os.path.join(data_dir, f)
        process_file(filepath, results, RESULTS_JSON)
        
    # Save results at the end (even if empty, to produce files for inspection)
    save_results(results, RESULTS_JSON)

    # Write CSV output (with header even if empty)
    if results:
        df = pd.DataFrame.from_dict(results, orient="index")
        df.index.name = "date_slot"
        df = df.reset_index()[["date_slot","mean","median","max","origin"]]
    else:
        df = pd.DataFrame(columns=["date_slot","mean","median","max","origin"])
    df.to_csv(OUT_CSV, index=False)
    print(f"[INFO] CSV written: {OUT_CSV} (rows={len(df)})")
    print(f"[INFO] JSON written: {RESULTS_JSON} (keys={len(results)})")

if __name__ == "__main__":
    main()
