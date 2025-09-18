# convert_era5_humidity_to_csv.py
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

# ---------------------------
# Utility functions
# ---------------------------
def compute_stats(arr):
    flat = arr.flatten()
    flat = flat[~np.isnan(flat)]
    if flat.size == 0:
        return np.nan, np.nan, np.nan, np.nan
    return (
        float(np.nanmean(flat)),          # Mean
        float(np.nanmedian(flat)),        # Median
        float(np.nanmax(flat)),           # Maximum
        float(np.nanpercentile(flat, 75)) # 75th percentile
    )

# ---------------------------
# Loop over years
# ---------------------------
for year in years:
    in_file = f"{year}-raw-nc-data/ERA5-{year}-humidity.nc"
    if not os.path.exists(in_file):
        print(f"Missing file, skip: {in_file}")
        continue

    print(f"\n=== Processing {in_file} ===")
    ds = xr.open_dataset(in_file)

    # Find relative humidity variable
    rh_var = None
    for cand in ['r', 'relative_humidity']:
        if cand in ds.data_vars:
            rh_var = cand
            break
    if rh_var is None:
        raise RuntimeError(f"No humidity variable found in {in_file}")

    # Extract time dimension (ERA5-complete uses valid_time)
    time_coord = ds['valid_time'].values

    # Compute statistics for each time point
    rows = []
    for i, t in enumerate(time_coord):
        arr = ds[rh_var].isel(valid_time=i).values  # Note: using valid_time here
        mean, median, maxv, p75 = compute_stats(arr)
        row = {
            'time': pd.Timestamp(t).strftime('%Y-%m-%d-%H'),
            'rh_mean': mean,
            'rh_median': median,
            'rh_max': maxv,
            'rh_p75': p75,
        }
        rows.append(row)

    df = pd.DataFrame(rows).set_index('time')

    # Create folder for this year
    year_dir = os.path.join(csv_root_dir, str(year))
    os.makedirs(year_dir, exist_ok=True)

    # Output CSV
    out_file = os.path.join(year_dir, f"humidity_{year}.csv")
    df.to_csv(out_file)
    print(f"Saved CSV: {out_file} (rows={len(df)})")

