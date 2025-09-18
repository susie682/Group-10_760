# convert_era5_variable_to_csv.py
import xarray as xr
import numpy as np
import pandas as pd
import os

# ---------------------------
# User parameters
# ---------------------------
years = range(2012, 2021)
csv_root_dir = 'split-csv-data'
os.makedirs(csv_root_dir, exist_ok=True)

# Variable to process (modify each time)
variable_name = 'humidity'  
in_template = f"{{year}}-raw-nc-data/ERA5-{{year}}-{variable_name}.nc"
out_prefix = variable_name

# ---------------------------
# Utility functions
# ---------------------------
def compute_stats(arr):
    flat = arr.flatten()
    flat = flat[~np.isnan(flat)]
    if flat.size == 0:
        return np.nan, np.nan, np.nan, np.nan
    return (
        float(np.nanmean(flat)),
        float(np.nanmedian(flat)),
        float(np.nanmax(flat)),
        float(np.nanpercentile(flat, 75))
    )

# ---------------------------
# Loop over years
# ---------------------------
for year in years:
    in_file = in_template.format(year=year)
    if not os.path.exists(in_file):
        print(f"Missing file, skip: {in_file}")
        continue

    print(f"\n=== Processing {in_file} ===")
    ds = xr.open_dataset(in_file)

    # Find variable
    var_candidates = ['r','relative_humidity'] if variable_name=='humidity' else \
                     ['tcc','total_cloud_cover'] if variable_name=='tcc' else \
                     ['t2m','2m_temperature','temperature']
    var_name = None
    for cand in var_candidates:
        if cand in ds.data_vars:
            var_name = cand
            break
    if var_name is None:
        raise RuntimeError(f"No variable found in {in_file}, candidates={var_candidates}")

    # Time dimension (ERA5-complete uses valid_time)
    time_coord = ds['valid_time'].values

    rows = []
    for i, t in enumerate(time_coord):
        arr = ds[var_name].isel(valid_time=i).values
        if variable_name == 'temp':  # Convert temperature to Celsius
            arr = arr - 273.15
        mean, median, maxv, p75 = compute_stats(arr)
        row = {
            'time': pd.Timestamp(t).strftime('%Y-%m-%d-%H'),
            f'{out_prefix}_mean': mean,
            f'{out_prefix}_median': median,
            f'{out_prefix}_max': maxv,
            f'{out_prefix}_p75': p75,
        }
        rows.append(row)

    df = pd.DataFrame(rows).set_index('time')

    # Create folder for the year
    year_dir = os.path.join(csv_root_dir, str(year))
    os.makedirs(year_dir, exist_ok=True)

    # Output CSV
    out_file = os.path.join(year_dir, f"{out_prefix}_{year}.csv")
    df.to_csv(out_file)
    print(f"Saved CSV: {out_file} (rows={len(df)})")

