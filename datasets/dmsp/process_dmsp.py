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
PF_LAT = 65.1  # Poker Flat latitude
PF_LON = -147.5  # Poker Flat longitude
ALT_KM = 0  # Ground level altitude

N_POINTS = 10  # Number of nearest points
DIST_THRESHOLD_KM = 100.0  # Distance threshold in km



DEBUG = False  # Toggle detailed print output

# ------------------ Functions ------------------

def parse_filename_datetime(filename):
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
    R = 6371.0
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad)*np.cos(lat2_rad)*np.sin(dlon/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

def select_hemisphere_data(ds, hemisphere="north"):
    if hemisphere.lower() == "north":
        aur_data = ds["AUR_ARC_RADIANCE_NORTH"].values
        lat_grid = ds["LATITUDE_GEOMAGNETIC_GRID_MAP"].values
        lon_grid = ds["LONGITUDE_GEOMAGNETIC_NORTH_GRID_MAP"].values
        ut_grid = ds.get("UT_N", None)
    else:
        aur_data = ds["AUR_ARC_RADIANCE_SOUTH"].values
        lat_grid = ds["LATITUDE_GEOMAGNETIC_GRID_MAP"].values
        lon_grid = ds["LONGITUDE_GEOMAGNETIC_SOUTH_GRID_MAP"].values
        ut_grid = ds.get("UT_S", None)
    return aur_data, lat_grid, lon_grid, ut_grid

def query_nearest_points(lat, lon, aur_data, lat_grid, lon_grid, ut_grid=None,
                         n_points=10, max_distance_km=100.0, origin_flag=0):
    lat_flat = lat_grid.flatten()
    lon_flat = lon_grid.flatten()
    aur_flat = aur_data.flatten()
    valid_idx = np.where(~np.isnan(aur_flat) & (aur_flat != 0))[0]

    if ut_grid is not None:
        ut_flat = ut_grid.values.flatten() if hasattr(ut_grid, "values") else ut_grid.flatten()
        valid_idx = valid_idx[ut_flat[valid_idx] != 0]

    if len(valid_idx) == 0:
        return [{"value":0, "lat":None, "lon":None, "ut":None,
                 "distance_km":None, "origin":0} for _ in range(n_points)], None

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
            val = 0
            ut_val = None

        results.append({
            "value": val,
            "lat": lat_grid[row, col] if val!=0 else None,
            "lon": lon_grid[row, col] if val!=0 else None,
            "ut": ut_val,
            "distance_km": dist_km if val!=0 else None,
            "origin": origin_flag
        })

    nearest_ut = None
    for r in results:
        if r["value"] != 0:
            nearest_ut = r["ut"]
            break
    return results, nearest_ut

def query_nearest_points_all(lat, lon, aur_data, lat_grid, lon_grid, ut_grid=None,
                             n_points=10, origin_flag=0):
    lat_flat = lat_grid.flatten()
    lon_flat = lon_grid.flatten()
    distances = haversine(lat, lon, lat_flat, lon_flat)
    nearest_idx_sorted = np.argsort(distances)

    results = []
    for i in range(n_points):
        idx = nearest_idx_sorted[i]
        row, col = np.unravel_index(idx, lat_grid.shape)
        val = aur_data[row, col]
        ut_val = ut_grid[row, col].item() if ut_grid is not None else None
        dist_km = distances[idx]
        results.append({
            "value": val,
            "lat": lat_grid[row, col],
            "lon": lon_grid[row, col],
            "ut": ut_val,
            "distance_km": dist_km,
            "origin": origin_flag
        })

    nearest_ut = results[0]["ut"] if results else None
    return results, nearest_ut

def map_to_kp_slot(ut_hours):
    slots = np.arange(0, 24, 3)
    diffs = np.abs(slots - ut_hours)
    idx = np.argmin(diffs)
    return int(slots[idx])

def load_checkpoint(checkpoint_file):
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, "r") as f:
            return json.load(f)
    return {}

def save_checkpoint(data, checkpoint_file):
    with open(checkpoint_file, "w") as f:
        json.dump(data, f)


def hours_to_utc(hours):
    if hours is None:
        return "N/A"
    h = int(hours)
    m = int((hours - h) * 60)
    s = int(((hours - h) * 60 - m) * 60)
    return f"{h:02d}:{m:02d}:{s:02d} UTC"

def process_file(filepath, results, processed_slots, checkpoint_file):
    filename = os.path.basename(filepath)
    ref_dt = parse_filename_datetime(filename)
    if not ref_dt:
        if DEBUG: print(f"Skipping {filename}, cannot parse datetime")
        return

    mag_lat, mag_lon, _ = aacgmv2.convert_latlon(PF_LAT, PF_LON, ALT_KM, dtime=ref_dt)
    mag_lon_360 = mag_lon % 360

    try:
        ds = xr.open_dataset(filepath, engine="netcdf4")
    except Exception as e:
        print(f"Failed to open file {filename}: {e}")
        return

    printed = False

    # ------------------ North Hemisphere ------------------
    try:
        aur_data_n, lat_grid_n, lon_grid_n, ut_grid_n = select_hemisphere_data(ds, "north")
        nearest_n, ut_n = query_nearest_points(mag_lat, mag_lon_360,
                                               aur_data_n, lat_grid_n, lon_grid_n, ut_grid_n,
                                               N_POINTS, DIST_THRESHOLD_KM, origin_flag=0)
        values = [p["value"] for p in nearest_n]
        origins = [p["origin"] for p in nearest_n]
        ut_val = ut_n

        if any(v != 0 for v in values):
            if DEBUG:
                print(f"[North] {filename} nearest points:")
                for i, p in enumerate(nearest_n, 1):
                    print(f" {i}: value={p['value']}, lat={p['lat']}, lon={p['lon']}, "
                          f"UT={hours_to_utc(p['ut'])}, distance={p['distance_km']}, origin={p['origin']}")
            printed = True
    except KeyError:
        if DEBUG:
            print(f"North hemisphere variables missing, skipping north for {filename}")

    # ------------------ South Hemisphere ------------------
    try:
        aur_data_s, lat_grid_s, lon_grid_s, ut_grid_s = select_hemisphere_data(ds, "south")
        nearest_s, ut_s = query_nearest_points(-mag_lat, mag_lon_360,
                                               aur_data_s, lat_grid_s, lon_grid_s, ut_grid_s,
                                               N_POINTS, DIST_THRESHOLD_KM, origin_flag=1)
        values = [p["value"] for p in nearest_s]
        origins = [p["origin"] for p in nearest_s]
        ut_val = ut_s
        if any(v != 0 for v in values):
            if DEBUG:
                print(f"[South] {filename} nearest points:")
                for i, p in enumerate(nearest_s, 1):
                    print(f" {i}: value={p['value']}, lat={p['lat']}, lon={p['lon']}, "
                          f"UT={hours_to_utc(p['ut'])}, distance={p['distance_km']}, origin={p['origin']}")
            printed = True
    except KeyError:
        if DEBUG:
            print(f"South hemisphere variables missing, skipping south for {filename}")

    # ------------------ Fallback ------------------
    if not printed:
        # Determine which hemisphere data to use for fallback
        try:
            aur_data_n, lat_grid_n, lon_grid_n, ut_grid_n = select_hemisphere_data(ds, "north")
            aur_data_use = aur_data_n
            lat_grid_use = lat_grid_n
            lon_grid_use = lon_grid_n
            ut_grid_use = ut_grid_n
            origin_flag = 0
            mag_lat_use = mag_lat
        except KeyError:
            try:
                aur_data_s, lat_grid_s, lon_grid_s, ut_grid_s = select_hemisphere_data(ds, "south")
                aur_data_use = aur_data_s
                lat_grid_use = lat_grid_s
                lon_grid_use = lon_grid_s
                ut_grid_use = ut_grid_s
                origin_flag = 1
                mag_lat_use = -mag_lat  # use mirrored latitude for Poker Flat
            except KeyError:
                print(f"No aurora data available in file {filename}, skipping file.")
                return

        nearest_all, ut_all = query_nearest_points_all(
            mag_lat_use, mag_lon_360, aur_data_use, lat_grid_use, lon_grid_use,
            ut_grid_use, N_POINTS, origin_flag=origin_flag
        )
        values = [p["value"] for p in nearest_all]
        origins = [p["origin"] for p in nearest_all]
        ut_val = ut_all or 0
        if DEBUG:
            print(f"[Fallback-All] {filename} nearest points (no valid values found):")
            for i, p in enumerate(nearest_all, 1):
                print(f" {i}: value={p['value']}, lat={p['lat']}, lon={p['lon']}, "
                      f"UT={hours_to_utc(p['ut'])}, distance={p['distance_km']}, origin={p['origin']}")

    # ------------------ Statistics ------------------
    mean_val = round(float(np.mean(values)), 2)
    med_val = round(float(np.median(values)), 2)
    max_val = round(float(np.max(values)), 2)
    ut_val = ut_val or 0

    h = int(ut_val)
    m = int((ut_val - h) * 60)
    s = int(((ut_val - h) * 60 - m) * 60)
    ut_time = timedelta(hours=h, minutes=m, seconds=s)
    ut_hours = ut_time.total_seconds() / 3600.0
    slot = map_to_kp_slot(ut_hours)
    key = f"{ref_dt.strftime('%Y-%m-%d')}-{slot:02d}"

    if key in results:
        if max_val > results[key]["max"]:
            results[key] = {"mean": mean_val, "median": med_val,
                            "max": max_val, "origin": origins[0]}
    else:
        results[key] = {"mean": mean_val, "median": med_val,
                        "max": max_val, "origin": origins[0]}

    processed_slots[key] = True
    save_checkpoint(processed_slots, checkpoint_file)

    if DEBUG:
        print(f"Processed {filename} -> {key}\n")



# ------------------ Main ------------------

def main():
    base_dir = r"C:\Users\Jervid\Desktop\2025s2\760\dmsp"
    year = 2012
    satelite = 17
    data_dir = os.path.join(base_dir, str(year), str(satelite))

    # Create csv folder if it does not exist
    csv_dir = os.path.join(base_dir, "csv")
    os.makedirs(csv_dir, exist_ok=True)

    # Set output file name as YYYY-17.csv
    OUTPUT_FILE = os.path.join(csv_dir, f"{year}-{satelite}.csv")
    CHECKPOINT_FILE = os.path.join(csv_dir, f"{year}-{satelite}-checkpoint.json")


    processed_slots = load_checkpoint(CHECKPOINT_FILE)
    results = {}

    files = sorted([f for f in os.listdir(data_dir) if f.endswith(".nc")])
    for f in files:
        filepath = os.path.join(data_dir, f)
        process_file(filepath, results, processed_slots, CHECKPOINT_FILE)

    files = sorted([f for f in os.listdir(data_dir) if f.endswith(".nc")])
    for f in tqdm(files, desc="Processing files", unit="file"):
        filepath = os.path.join(data_dir, f)
        process_file(filepath, results, processed_slots, CHECKPOINT_FILE)

    # Write final CSV
    df = pd.DataFrame([
        {"date_slot": k,
         "mean": v["mean"],
         "median": v["median"],
         "max": v["max"],
         "origin": v["origin"]}
        for k, v in sorted(results.items())
    ])
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Final results written to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
