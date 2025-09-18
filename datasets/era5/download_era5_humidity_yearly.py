# download_era5_humidity_yearly.py
from cdsapi import Client
import pandas as pd
import os

# ---------------------------
# 用户参数
# ---------------------------
years = range(2012, 2021)  # 2012–2020
area = '66.13/-148.47/64.13/-146.47'  # lat_max/lon_min/lat_min/lon_max
grid = '1.0/1.0'
levels = '1000'  # 气压层
param = '157'  # relative humidity

c = Client()

# ---------------------------
# 年份循环下载
# ---------------------------
for year in years:
    raw_dir = f'{year}-raw-nc-data'
    os.makedirs(raw_dir, exist_ok=True)

    print(f"\n=== Processing year {year} ===")

    # 整年一次性下载
    fname = f'{raw_dir}/ERA5-{year}-humidity.nc'
    if os.path.exists(fname):
        print(f"File exists, skip: {fname}")
        continue

    c.retrieve(
        'reanalysis-era5-complete',
        {
            'date': f'{year}-01-01/to/{year}-12-31',
            'levtype': 'pl',
            'levelist': levels,
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
