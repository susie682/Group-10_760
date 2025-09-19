import pandas as pd
import glob
import os

# Path where yearly Kp CSVs are stored
base_dir = "/Users/madhujachenthilraj/project/workspace/CS760_Advanced_ML/kp"

# Find all Kp CSVs
files = glob.glob(os.path.join(base_dir, "Kp_*.csv"))

dfs = []
for f in files:
    df = pd.read_csv(f)
    
    # Ensure time column is parsed
    df["time"] = pd.to_datetime(df["time"], format="%Y-%m-%d-%H")
    
    dfs.append(df)

# Concatenate all years
kp_all = pd.concat(dfs, ignore_index=True)

# Sort by time
kp_all = kp_all.sort_values("time")

# Save merged file
out_file = os.path.join(base_dir, "Kp_all_years.csv")
kp_all.to_csv(out_file, index=False)

print(f"[INFO] Final Kp dataset saved to {out_file} with {kp_all.shape[0]} rows")
