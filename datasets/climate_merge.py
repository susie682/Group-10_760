import pandas as pd
import glob
import os

# Base folder containing yearly csvs
base_dir = "/Users/madhujachenthilraj/project/workspace/CS760_Advanced_ML/era5"

# Variables to merge
variables = ["humidity", "tcc", "temp", "totalPrecipitation"]

yearly_dfs = []

for year in range(2012, 2020):  # adjust to your available years
    dfs = []
    for var in variables:
        file_path = os.path.join(base_dir, f"{var}_{year}.csv")
        df = pd.read_csv(file_path)
        
        # Make sure 'time' is datetime
        df["time"] = pd.to_datetime(df["time"])
        
        dfs.append(df)
    
    # Merge all 4 variables for this year on 'time'
    year_df = dfs[0]
    for df in dfs[1:]:
        year_df = year_df.merge(df, on="time", how="outer")
    
    # Sort by time
    year_df = year_df.sort_values("time")
    
    # Save per-year merged dataset
    year_df.to_csv(os.path.join(base_dir, f"climate_{year}.csv"), index=False)
    yearly_dfs.append(year_df)

# Concatenate all years together
climate_all = pd.concat(yearly_dfs, ignore_index=True).sort_values("time")

# Save combined dataset
climate_all.to_csv(os.path.join(base_dir, "climate_all_years.csv"), index=False)

print("[INFO] Final climate dataset shape:", climate_all.shape)
