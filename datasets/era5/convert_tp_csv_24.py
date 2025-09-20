# convert_era5_tp_to_1hr_csv.py
import xarray as xr
import numpy as np
import pandas as pd
import os

# ---------------------------
# User parameters
# ---------------------------
years = range(2012, 2021)  # 2012–2020
csv_root_dir = 'split-csv-data'  # Output root directory
os.makedirs(csv_root_dir, exist_ok=True)

center_lat = 65.13
center_lon = -147.47
radii_km = [150]       # Can extend to multiple radii
tp_thr_mm = 0.1

# ---------------------------
# Utility functions
# ---------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance (km) using Haversine formula"""
    R = 6371.0
    lat1r = np.deg2rad(lat1)
    lon1r = np.deg2rad(lon1)
    lat2r = np.deg2rad(lat2)
    lon2r = np.deg2rad(lon2)
    dlat = lat2r - lat1r
    dlon = lon2r - lon1r
    a = np.sin(dlat/2.0)**2 + np.cos(lat1r)*np.cos(lat2r)*np.sin(dlon/2.0)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def area_weights(latitudes, longitudes):
    """Latitude-based area weights"""
    wlat = np.cos(np.deg2rad(latitudes))
    W = np.outer(wlat, np.ones_like(longitudes))
    return W

def compute_mask(ds, center_lat, center_lon, radius_km):
    """Return a boolean mask of grid cells within radius_km"""
    lats = ds['latitude'].values
    lons = ds['longitude'].values
    LAT, LON = np.meshgrid(lats, lons, indexing='ij')
    dist = haversine_km(center_lat, center_lon, LAT, LON)
    mask = dist <= radius_km
    return mask

# ---------------------------
# Loop over years
# ---------------------------
for year in years:
    tp_06_file = f'merged-nc-data/ERA5_{year}_06UTC_merged-24.nc'
    tp_18_file = f'merged-nc-data/ERA5_{year}_18UTC_merged-24.nc'

    if not (os.path.exists(tp_06_file) and os.path.exists(tp_18_file)):
        print(f"Missing file(s) for year {year}, skip.")
        continue

    print(f"\n=== Processing year {year} ===")
    ds_06 = xr.open_dataset(tp_06_file).rename({'valid_time':'time'})
    ds_18 = xr.open_dataset(tp_18_file).rename({'valid_time':'time'})

    # Find TP variable
    tp_var = None
    for ds_tmp in [ds_06, ds_18]:
        for cand in ['tp','tp_mm','total_precipitation']:
            if cand in ds_tmp.data_vars:
                tp_var = cand
                break
    if tp_var is None:
        raise RuntimeError(f"No TP variable found for year {year}")

    # Convert unit m -> mm
    tp_06 = ds_06[tp_var] * 1000.0
    tp_18 = ds_18[tp_var] * 1000.0

    # ---------------------------
    # Compute statistics for each 1-hour step
    # ---------------------------
    rows = []
    mask = compute_mask(ds_06, center_lat, center_lon, radii_km[0])
    W = area_weights(ds_06['latitude'].values, ds_06['longitude'].values)
    W_masked = W * mask
    W_sum = W_masked.sum()

    # process 06 UTC forecasts
    for i, t in enumerate(ds_06['time'].values):
        time_str = pd.Timestamp(t).strftime('%Y-%m-%d-%H')
        arr = tp_06.isel(time=i).values
        arr_masked = np.where(mask, arr, np.nan)
        flat = arr_masked.flatten()
        flat = flat[~np.isnan(flat)]
        row = {
            'time': time_str,
            'tp_mm_mean_aw': float(np.nansum(arr_masked * W_masked)/W_sum) if flat.size>0 else np.nan,
            'tp_mm_median': float(np.nanmedian(flat)) if flat.size>0 else np.nan,
            'tp_mm_max': float(np.nanmax(flat)) if flat.size>0 else np.nan,
            'tp_mm_p75': float(np.nanpercentile(flat,75)) if flat.size>0 else np.nan,
            f'tp_frac_gt_{tp_thr_mm}': float((flat>tp_thr_mm).sum()/flat.size) if flat.size>0 else np.nan
        }
        rows.append(row)

    # process 18 UTC forecasts
    for i, t in enumerate(ds_18['time'].values):
        time_str = pd.Timestamp(t).strftime('%Y-%m-%d-%H')
        arr = tp_18.isel(time=i).values
        arr_masked = np.where(mask, arr, np.nan)
        flat = arr_masked.flatten()
        flat = flat[~np.isnan(flat)]
        row = {
            'time': time_str,
            'tp_mm_mean_aw': float(np.nansum(arr_masked * W_masked)/W_sum) if flat.size>0 else np.nan,
            'tp_mm_median': float(np.nanmedian(flat)) if flat.size>0 else np.nan,
            'tp_mm_max': float(np.nanmax(flat)) if flat.size>0 else np.nan,
            'tp_mm_p75': float(np.nanpercentile(flat,75)) if flat.size>0 else np.nan,
            f'tp_frac_gt_{tp_thr_mm}': float((flat>tp_thr_mm).sum()/flat.size) if flat.size>0 else np.nan
        }
        rows.append(row)

    df = pd.DataFrame(rows).set_index('time')

    # Sort by time
    df['time_dt'] = pd.to_datetime(df.index, format='%Y-%m-%d-%H')
    df = df.sort_values('time_dt').drop(columns='time_dt')
    df.index.name = 'time'

    # Output CSV to split-csv-data-1hr/year/
    year_dir = os.path.join(csv_root_dir, str(year))
    os.makedirs(year_dir, exist_ok=True)
    out_file = os.path.join(year_dir, f"totalPrecipitation_{year}_24.csv")
    df.to_csv(out_file)
    print(f"Saved CSV: {out_file} (rows={len(df)})")
