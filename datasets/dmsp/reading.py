import xarray as xr
import numpy as np
import aacgmv2
from datetime import datetime, timedelta


# Poker Flat geographic coordinates
PF_LAT = 65.1
PF_LON = -147.5
ALT_KM = 0  # Ground level

# Convert to geomagnetic coordinates
dt = datetime(2016, 1, 1, 6, 0, 0)
mag_lat, mag_lon, _ = aacgmv2.convert_latlon(PF_LAT, PF_LON, ALT_KM, dtime=dt)
mag_lon_360 = mag_lon % 360
print(f"Poker Flat geomagnetic coordinates: lat={mag_lat}, lon={mag_lon} ({mag_lon_360})")

# Open file
file_path = "2016/dmspf18_ssusi_edr-aurora_2016001T064259-2016001T082450-REV31990_vA8.2.0r000.nc" 

ds = xr.open_dataset(file_path, engine="netcdf4")

def select_hemisphere_data(hemisphere="north"):
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

def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance formula, return distance between two points (km)"""
    R = 6371.0  # Earth radius in km
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    return R * c

def query_nearest_points(lat, lon, aur_data, lat_grid, lon_grid, ut_grid=None, n_points=3):
    """
    Query the nearest n_points grid points with valid data (exclude NaN and 0),
    sorted by great-circle distance
    """
    lat_flat = lat_grid.flatten()
    lon_flat = lon_grid.flatten()
    aur_flat = aur_data.flatten()

    # Select valid indices: not NaN and not 0
    valid_idx = np.where((~np.isnan(aur_flat)) & (aur_flat != 0))[0]

    # If UT grid exists, require UT != 0
    if ut_grid is not None:
        ut_flat = ut_grid.values.flatten() if hasattr(ut_grid, "values") else ut_grid.flatten()
        valid_idx = valid_idx[ut_flat[valid_idx] != 0]

    if len(valid_idx) == 0:
        return []  # No valid data

    # Compute distances for valid points
    distances = haversine(lat, lon, lat_flat[valid_idx], lon_flat[valid_idx])

    # Sort by distance and select top n_points
    nearest_idx = valid_idx[np.argsort(distances)[:n_points]]

    results = []
    for idx in nearest_idx:
        row, col = np.unravel_index(idx, lat_grid.shape)
        covered = True
        ut_value = None
        if ut_grid is not None:
            ut_value = ut_grid[row, col].item()  # convert to Python scalar
            covered = ut_value != 0
        value = aur_data[row, col]
        results.append({
            "value": value,
            "row": row,
            "col": col,
            "lat": lat_grid[row, col],
            "lon": lon_grid[row, col],
            "covered": covered,
            "ut": ut_value,
            "distance_km": haversine(lat, lon, lat_grid[row, col], lon_grid[row, col])
        })
    return results

def hours_to_utc(hours):
    h = int(hours)
    m = int((hours - h) * 60)
    s = int(((hours - h) * 60 - m) * 60)
    return f"{h:02d}:{m:02d}:{s:02d} UTC"

# Query the nearest 3 valid points in the Northern Hemisphere
aur_data_n, lat_grid_n, lon_grid_n, ut_grid_n = select_hemisphere_data("north")
nearest_north = query_nearest_points(mag_lat, mag_lon, aur_data_n, lat_grid_n, lon_grid_n, ut_grid_n, n_points=3)

# Query the nearest 3 valid points in the Southern Hemisphere
aur_data_s, lat_grid_s, lon_grid_s, ut_grid_s = select_hemisphere_data("south")
nearest_south = query_nearest_points(-mag_lat, mag_lon, aur_data_s, lat_grid_s, lon_grid_s, ut_grid_s, n_points=3)

# Output results
print("=== Nearest 3 valid points in Northern Hemisphere ===")
for i, p in enumerate(nearest_north, 1):
    print(f"{i}: value={p['value']}, lat={p['lat']:.2f}, lon={p['lon']:.2f}, "
          f"UT={hours_to_utc(p['ut'])}, distance ≈ {p['distance_km']:.2f} km")

print("=== Nearest 3 valid points in Southern Hemisphere ===")
for i, p in enumerate(nearest_south, 1):
    print(f"{i}: value={p['value']}, lat={p['lat']:.2f}, lon={p['lon']:.2f}, "
          f"UT={hours_to_utc(p['ut'])}, distance ≈ {p['distance_km']:.2f} km")


