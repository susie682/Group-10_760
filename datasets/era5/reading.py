import xarray as xr

# Open NC files
ds_06 = xr.open_dataset('ERA5_2015_06UTC_merged.nc')
ds_18 = xr.open_dataset('ERA5_2015_18UTC_merged.nc')

# Check time/valid_time dimension
print(ds_06)
print(ds_18)

# Only check the time coordinate array
print("06UTC times:")
print(ds_06['valid_time'].values)  # If your merged file has time dimension named valid_time
print("18UTC times:")
print(ds_18['valid_time'].values)

