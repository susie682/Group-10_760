import pandas as pd

# File paths
satellite_file = "/Users/madhujachenthilraj/project/workspace/CS760_Advanced_ML/satellite_all_years.csv"
climate_file   = "/Users/madhujachenthilraj/project/workspace/CS760_Advanced_ML/climate_all_years.csv"
kp_file        = "/Users/madhujachenthilraj/project/workspace/CS760_Advanced_ML/Kp_all_years.csv"
keogram_file   = "/Users/madhujachenthilraj/Downloads/keogram/keogram_all_years.csv"
output_file    = "/Users/madhujachenthilraj/project/workspace/CS760_Advanced_ML/final.csv"

# --- Load datasets ---
sat = pd.read_csv(satellite_file)
climate = pd.read_csv(climate_file)
kp = pd.read_csv(kp_file)
keogram = pd.read_csv(keogram_file)

# --- Fix time columns ---
# Satellite
sat["time"] = pd.to_datetime(sat["date_slot"], format="%Y-%m-%d-%H")
sat = sat.drop(columns=["date_slot"])

# Climate
climate["time"] = pd.to_datetime(climate["time"])

# Kp
kp["time"] = pd.to_datetime(kp["time"])

# Keogram
keogram["time"] = pd.to_datetime(keogram["time"])

# --- Rename columns to avoid clashes ---
sat = sat.rename(columns={
    "mean": "satellite_mean",
    "median": "satellite_median",
    "max": "satellite_max",
    "origin": "satellite_origin"
})

keogram = keogram.rename(columns={
    "mean": "keogram_mean",
    "median": "keogram_median",
    "max": "keogram_max"
})

# --- Merge all datasets ---
merged = (climate.merge(sat, on="time", how="outer")
                  .merge(kp, on="time", how="outer")
                  .merge(keogram, on="time", how="outer"))

# Sort by time
merged = merged.sort_values("time")

# --- Save final dataset ---
merged.to_csv(output_file, index=False)

print(f"[INFO] Final master dataset saved to {output_file}")
print(f"[INFO] Shape: {merged.shape}")
print(f"[INFO] Time range: {merged['time'].min()} → {merged['time'].max()}")
