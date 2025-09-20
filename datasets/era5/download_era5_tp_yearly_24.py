# download_and_merge_era5_tp_yearly.py
from cdsapi import Client
import pandas as pd
import xarray as xr
import glob
import os

# ---------------------------
# User parameters
# ---------------------------
years = range(2012, 2021)  # 2013-2020
area = '66.13/-148.47/64.13/-146.47'  # lat_max/lon_min/lat_min/lon_max
grid = '1.0/1.0'
requests = [
    {'time': '06:00', 'steps': [str(i) for i in range(1,13)]},
    {'time': '18:00', 'steps': [str(i) for i in range(1,13)]}
]

c = Client()

# ---------------------------
# Loop over years: download and merge
# ---------------------------
for year in years:
    raw_dir = f'{year}-raw-nc-data'
    merged_dir = 'merged-nc-data'
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(merged_dir, exist_ok=True)

    print(f"\n=== Processing year {year} ===")

    # ---------------------------
    # Download monthly from Jan 1 to Nov 30
    # ---------------------------
    for month in range(1, 12):  # Jan~Nov
        start_date = pd.Timestamp(f'{year}-{month:02d}-01')
        end_date = start_date + pd.offsets.MonthEnd(0)
        month_str = f'{start_date.strftime("%Y%m")}'
        for req in requests:
            fname = f'{raw_dir}/ERA5-{month_str}-tp_{req["time"].replace(":","")}-24.nc'
            if os.path.exists(fname):
                print(f"File exists, skip: {fname}")
                continue
            date_range = f"{start_date.strftime('%Y-%m-%d')}/to/{end_date.strftime('%Y-%m-%d')}"
            c.retrieve(
                'reanalysis-era5-complete',
                {
                    'date': date_range,
                    'levtype': 'sfc',
                    'param': '228',
                    'stream': 'oper',
                    'time': req['time'],
                    'step': req['steps'],
                    'type': 'fc',
                    'area': area,
                    'grid': grid,
                    'format': 'netcdf',
                },
                fname
            )
            print(f"Submitted request for {date_range} {req['time']} UTC")

    # ---------------------------
    # Download Dec 1 to Dec 30
    # ---------------------------
    start_dec = pd.Timestamp(f'{year}-12-01')
    end_dec = pd.Timestamp(f'{year}-12-30')
    for req in requests:
        fname = f'{raw_dir}/ERA5-{year}1201to1230-tp_{req["time"].replace(":","")}-24.nc'
        if os.path.exists(fname):
            print(f"File exists, skip: {fname}")
            continue
        date_range = f"{start_dec.strftime('%Y-%m-%d')}/to/{end_dec.strftime('%Y-%m-%d')}"
        c.retrieve(
            'reanalysis-era5-complete',
            {
                'date': date_range,
                'levtype': 'sfc',
                'param': '228',
                'stream': 'oper',
                'time': req['time'],
                'step': req['steps'],
                'type': 'fc',
                'area': area,
                'grid': grid,
                'format': 'netcdf',
            },
            fname
        )
        print(f"Submitted request for {date_range} {req['time']} UTC")

    # ---------------------------
    # Download Dec 31 separately
    # ---------------------------
    dec31 = pd.Timestamp(f'{year}-12-31')
    for req in requests:
        steps = req['steps']
        # For 18:00 UTC, exclude steps 9 and 12
        if req['time'] == '18:00':
            steps = [s for s in steps if s not in ['9','10','11','12']]
        fname = f'{raw_dir}/ERA5-{year}1231-tp_{req["time"].replace(":","")}-24.nc'
        if os.path.exists(fname):
            print(f"File exists, skip: {fname}")
            continue
        c.retrieve(
            'reanalysis-era5-complete',
            {
                'date': dec31.strftime('%Y-%m-%d'),
                'levtype': 'sfc',
                'param': '228',
                'stream': 'oper',
                'time': req['time'],
                'step': steps,
                'type': 'fc',
                'area': area,
                'grid': grid,
                'format': 'netcdf',
            },
            fname
        )
        print(f"Submitted request for {dec31.strftime('%Y-%m-%d')} {req['time']} UTC (steps={steps})")

    # ---------------------------
    # Merge NC files
    # ---------------------------
    files_06 = sorted(glob.glob(f'{raw_dir}/ERA5-*-tp_0600-24.nc'))
    files_18 = sorted(glob.glob(f'{raw_dir}/ERA5-*-tp_1800-24.nc'))

    print(f"Merging {len(files_06)} files for 06UTC and {len(files_18)} files for 18UTC...")

    ds_06 = xr.open_mfdataset(files_06, combine='by_coords',
                              preprocess=lambda ds: ds.sortby('valid_time'))
    ds_18 = xr.open_mfdataset(files_18, combine='by_coords',
                              preprocess=lambda ds: ds.sortby('valid_time'))

    # Save merged data
    ds_06.to_netcdf(f'{merged_dir}/ERA5_{year}_06UTC_merged-24.nc')
    ds_18.to_netcdf(f'{merged_dir}/ERA5_{year}_18UTC_merged-24.nc')
    print(f"Saved merged NC files: {merged_dir}/ERA5_{year}_06UTC_merged.nc and {merged_dir}/ERA5_{year}_18UTC_merged.nc")
