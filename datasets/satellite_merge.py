import pandas as pd
import glob
import os

# Path where all yearly satellite CSVs are stored
base_dir = "/Users/madhujachenthilraj/project/workspace/CS760_Advanced_ML"

# Get all CSV files (assuming filenames like "2015-17.csv", "2016-17.csv", etc.)
files = glob.glob(os.path.join(base_dir, "*.csv"))

dfs = []
for f in files:
    df = pd.read_csv(f)
    # ensure consistent dtypes
    df["date_slot"] = df["date_slot"].astype(str)
    dfs.append(df)

# Concatenate all years together
satellite_all = pd.concat(dfs, ignore_index=True)

# Sort by date_slot (optional but recommended)
satellite_all = satellite_all.sort_values("date_slot")

# Save merged dataset
out_file = os.path.join(base_dir, "satellite_all_years.csv")
satellite_all.to_csv(out_file, index=False)

print(f"[INFO] Combined dataset written to {out_file} with {len(satellite_all)} rows")
