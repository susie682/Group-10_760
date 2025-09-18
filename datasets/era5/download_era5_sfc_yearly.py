# download_era5_humidity_yearly.py
from cdsapi import Client
import pandas as pd
import os

# ---------------------------
# User parameters
# ---------------------------
years = range(2012, 2021)  # 2012–2020
area = '66.13/-148.47/64.13/-146.47'  # lat_max/lon_min/lat_min/lon_max
grid = '1.0/1.0'

param = '167'  # relative humidity
paramName = 'temp'
# total cloud cover: tcc, 164
# 2m temp: temp, 167

c = Client()

# ---------------------------
# Loop over years to download
# ---------------------------
for year in years:
    raw_dir = f'{year}-raw-nc-data'
    os.makedirs(raw_dir, exist_ok=True)

    print(f"\n=== Processing year {year} ===")

    # Download the whole year at once
    fname = f'{raw_dir}/ERA5-{year}-{paramName}.nc'
    if os.path.exists(fname):
        print(f"File exists, skip: {fname}")
        continue

    c.retrieve(
        'reanalysis-era5-complete',
        {
            'date': f'{year}-01-01/to/{year}-12-31',
            'levtype': 'sfc',
            'param': param,
            'stream': 'oper',
            'time': '00/to/23/by/3',
            'type': 'an',
            'area': area,
            'grid': grid,
            'format': 'netcdf',
        },
        fname
    )
    print(f"Submitted request for {year}")
