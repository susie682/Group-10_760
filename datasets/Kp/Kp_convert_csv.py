import os
import pandas as pd

# Folder paths
txt_folder = "txt"   # Folder containing Kp_txt_YYYY files
csv_folder = "csv"   # Folder to save output CSV files

# Ensure the output folder exists
os.makedirs(csv_folder, exist_ok=True)

# Loop over years 2012-2020
for year in range(2012, 2021):
    txt_file = os.path.join(txt_folder, f"Kp_txt_{year}.txt")
    csv_file = os.path.join(csv_folder, f"Kp_{year}.csv")
    
    # Read the txt file
    with open(txt_file, 'r') as f:
        lines = f.readlines()
    
    records = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 9:
            continue  # Skip empty or malformed lines
        
        YYYY, MM, DD, hh_h = parts[0], parts[1], parts[2], parts[3]
        Kp, ap = parts[7], parts[8]
        
        # Convert to YYYY-MM-DD-UT format
        hour_int = int(float(hh_h))
        time_str = f"{YYYY}-{MM.zfill(2)}-{DD.zfill(2)}-{hour_int:02d}"
        
        # Append record
        records.append([time_str, Kp, ap])
    
    # Convert to DataFrame and add column names
    df = pd.DataFrame(records, columns=["time", "Kp", "ap"])
    
    # Save CSV with header
    df.to_csv(csv_file, index=False)
    print(f"Saved {csv_file}")
