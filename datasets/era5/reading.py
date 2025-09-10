import xarray as xr

# Open the downloaded nc file
ds = xr.open_dataset('ERA5-2015.nc')

# Inspect the entire dataset (optional)
# print(ds)

# Calculate the average total cloud cover over latitude and longitude
tcc_avg = ds['tcc'].mean(dim=['latitude', 'longitude'])

# Show result
print(tcc_avg)
