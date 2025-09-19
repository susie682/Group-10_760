import pandas as pd
import glob, os

base_dir = "/Users/madhujachenthilraj/Downloads/keogram"

# Match your actual filenames
files = glob.glob(os.path.join(base_dir, "keogram_segment_stats*_filtered.csv"))

print("[INFO] Found", len(files), "files")

dfs = []
for f in files:
    df = pd.read_csv(f)
    # Convert 'segment' column to datetime
    df["time"] = pd.to_datetime(df["segment"], format="%Y-%m-%d-%H")
    df = df.drop(columns=["segment"])
    dfs.append(df)

if dfs:  # only if something matched
    keogram_all = pd.concat(dfs, ignore_index=True).sort_values("time")
    keogram_all.to_csv(os.path.join(base_dir, "keogram_all_years.csv"), index=False)
    print("[INFO] keogram_all_years.csv built:", keogram_all.shape)
    print("Range:", keogram_all["time"].min(), "→", keogram_all["time"].max())
else:
    print("[ERROR] No keogram files found in:", base_dir)
