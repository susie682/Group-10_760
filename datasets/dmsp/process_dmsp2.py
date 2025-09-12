# -*- coding: utf-8 -*-
"""
DMSP SSUSI EDR-Aurora nearest-point summarizer @ Poker Flat
- 读取每个 NetCDF 文件
- 将地理坐标 (PF_LAT, PF_LON) 转为 AACGM(磁坐标)
- 在北/南半球网格上查找附近 N 个点（优先阈值内，否则全域最近）
- 计算 mean/median/max，并按 3 小时 Kp slot 聚合
- 每处理一个文件就把 results.json 持久化（断点续跑安全）

!!! 修改点用 `### CHANGE` 标注 !!!
"""

import os
import re
import json
import xarray as xr
import numpy as np
import aacgmv2
from datetime import datetime, timedelta
import pandas as pd
from tqdm import tqdm

# ------------------ Configurations ------------------
PF_LAT = 65.1          # Poker Flat latitude
PF_LON = -147.5        # Poker Flat longitude

### CHANGE 1: 更贴近极光层高度，推荐 110 km（原来是 0）
ALT_KM = 110           # Altitude for AACGM conversion (km), 0 also works but 110km is more physical

N_POINTS = 10          # Number of nearest points
### CHANGE 2: 阈值可适当放宽，避免全 0；如数据稀疏可调 150-250
DIST_THRESHOLD_KM = 150.0  # Distance threshold in km

DEBUG = False  # Toggle detailed print output

# ------------------ Utilities ------------------

def parse_filename_datetime(filename):
    """Parse YYYYJJJThhmmss from filename."""
    m = re.search(r"(\d{4})(\d{3})T(\d{2})(\d{2})(\d{2})", filename)
    if not m:
        return None
    year = int(m.group(1))
    day_of_year = int(m.group(2))
    hour = int(m.group(3))
    minute = int(m.group(4))
    second = int(m.group(5))
    return datetime(year, 1, 1) + timedelta(days=day_of_year - 1,
                                             hours=hour, minutes=minute, seconds=second)

def haversine(lat1, lon1, lat2, lon2):
    """Great-circle distance (km). Supports numpy arrays."""
    R = 6371.0
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad)*np.cos(lat2_rad)*np.sin(dlon/2)**2
    return 2*R*np.arcsin(np.sqrt(a))

def select_hemisphere_data(ds, hemisphere="north"):
    """Return aurora array, lat, lon, and UT grid by hemisphere."""
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

def query_nearest_points(lat, lon, aur_data, lat_grid, lon_grid, ut_grid=None,
                         n_points=10, max_distance_km=100.0, origin_flag=0):
    """
    在阈值内（<=max_distance_km）取最近的 n_points 非零点；
    若没有合法点，返回 n_points 个占位项。
    """
    lat_flat = lat_grid.flatten()
    lon_flat = lon_grid.flatten()
    aur_flat = aur_data.flatten()

    valid_idx = np.where(~np.isnan(aur_flat) & (aur_flat != 0))[0]

    if ut_grid is not None:
        ut_flat = ut_grid.values.flatten() if hasattr(ut_grid, "values") else ut_grid.flatten()
        valid_idx = valid_idx[ut_flat[valid_idx] != 0]

    if len(valid_idx) == 0:
        return [{"value": 0, "lat": None, "lon": None, "ut": None,
                 "distance_km": None, "origin": 0} for _ in range(n_points)], None

    distances = haversine(lat, lon, lat_flat[valid_idx], lon_flat[valid_idx])
    nearest_idx_sorted = np.argsort(distances)
    results = []

    for i in range(n_points):
        if i < len(nearest_idx_sorted):
            idx = valid_idx[nearest_idx_sorted[i]]
            row, col = np.unravel_index(idx, lat_grid.shape)
            dist_km = haversine(lat, lon, lat_grid[row, col], lon_grid[row, col])
            if dist_km <= max_distance_km:
                val = aur_data[row, col]
                ut_val = ut_grid[row, col].item() if ut_grid is not None else None
            else:
                val = 0
                ut_val = None
        else:
            # 不足 n_points 的填充
            val, ut_val, dist_km, row, col = 0, None, None, 0, 0

        results.append({
            "value": val,
            "lat": lat_grid[row, col] if val != 0 else None,
            "lon": lon_grid[row, col] if val != 0 else None,
            "ut": ut_val,
            "distance_km": dist_km if val != 0 else None,
            "origin": origin_flag
        })

    # 选一个最近的 UT 作为代表
    nearest_ut = None
    for r in results:
        if r["value"] != 0:
            nearest_ut = r["ut"]
            break
    return results, nearest_ut

def query_nearest_points_all(lat, lon, aur_data, lat_grid, lon_grid, ut_grid=None,
                             n_points=10, origin_flag=0):
    """全域最近 n_points（不限制距离）"""
    lat_flat = lat_grid.flatten()
    lon_flat = lon_grid.flatten()
    distances = haversine(lat, lon, lat_flat, lon_flat)
    nearest_idx_sorted = np.argsort(distances)

    results = []
    for i in range(min(n_points, nearest_idx_sorted.size)):
        idx = nearest_idx_sorted[i]
        row, col = np.unravel_index(idx, lat_grid.shape)
        val = aur_data[row, col]
        ut_val = ut_grid[row, col].item() if ut_grid is not None else None
        dist_km = distances[idx]
        results.append({
            "value": float(val),
            "lat": float(lat_grid[row, col]),
            "lon": float(lon_grid[row, col]),
            "ut": ut_val,
            "distance_km": float(dist_km),
            "origin": origin_flag
        })

    nearest_ut = results[0]["ut"] if results else None
    return results, nearest_ut

def map_to_kp_slot(ut_hours):
    """Map hours [0,24) to nearest 3-hour slot: 0,3,...,21"""
    slots = np.arange(0, 24, 3)
    diffs = np.abs(slots - ut_hours)
    idx = np.argmin(diffs)
    return int(slots[idx])

### CHANGE 3: 新的结果断点续跑（持久化 results 而不是 processed_slots）
def load_results(json_path):
    if os.path.exists(json_path):
        with open(json_path, "r") as f:
            return json.load(f)
    return {}

def save_results(results, json_path):
    # 原子替换，避免中途写坏
    tmp = json_path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(results, f)
    os.replace(tmp, json_path)

def hours_to_utc(hours):
    if hours is None:
        return "N/A"
    h = int(hours)
    m = int((hours - h) * 60)
    s = int(((hours - h) * 60 - m) * 60)
    return f"{h:02d}:{m:02d}:{s:02d} UTC"

# ------------------ Core per-file processing ------------------

def process_file(filepath, results, results_json, dist_km=DIST_THRESHOLD_KM):
    """
    处理单个文件，更新 results（slot -> 统计量），并立即保存到 results_json
    """
    filename = os.path.basename(filepath)
    ref_dt = parse_filename_datetime(filename)
    if not ref_dt:
        if DEBUG: print(f"Skip (cannot parse datetime): {filename}")
        return

    # 磁坐标（推荐 110km）
    mag_lat, mag_lon, _ = aacgmv2.convert_latlon(PF_LAT, PF_LON, ALT_KM, dtime=ref_dt)
    mag_lon_360 = mag_lon % 360

    # 小帮手：在一个半球上做“阈值内最近N个/无则全域最近N个”，并计算统计量
    def summarize(aur, latg, long, utg, hemi_flag):
        # 北半球用 mag_lat，南半球镜像到 -mag_lat（以便在南网格上找最近）
        q_lat = mag_lat if hemi_flag == 0 else -mag_lat

        pts, ut = query_nearest_points(q_lat, mag_lon_360, aur, latg, long, utg,
                                       N_POINTS, dist_km, origin_flag=hemi_flag)
        if not any(p["value"] != 0 for p in pts):
            pts, ut = query_nearest_points_all(q_lat, mag_lon_360, aur, latg, long, utg,
                                               N_POINTS, origin_flag=hemi_flag)

        vals = np.array([p["value"] for p in pts], dtype=float)
        # 如果全是 0，也会给出 0 的统计量，不会是 NaN
        return {
            "mean": float(np.mean(vals)),
            "median": float(np.median(vals)),
            "max": float(np.max(vals)),
            "origin": hemi_flag,
            "ut": ut
        }

    # 用 with 自动关闭文件，避免句柄/内存泄漏 —— ### CHANGE 4
    try:
        with xr.open_dataset(filepath, engine="netcdf4") as ds:
            # 北半球
            aur_n, lat_n, lon_n, ut_n = select_hemisphere_data(ds, "north")
            north = summarize(aur_n, lat_n, lon_n, ut_n, hemi_flag=0)
            # 南半球
            aur_s, lat_s, lon_s, ut_s = select_hemisphere_data(ds, "south")
            south = summarize(aur_s, lat_s, lon_s, ut_s, hemi_flag=1)
    except Exception as e:
        print(f"Failed to open/process {filename}: {e}")
        return

    # 选“max 更大”的半球 —— ### CHANGE 5（更明确的半球选择规则）
    best = north if north["max"] >= south["max"] else south

    # UT 兜底：如果网格没有 UT，就用文件名解析的时间（最接近轨道实际时间）—— ### CHANGE 6
    ut_hours = (best["ut"] if best["ut"] is not None
                else ref_dt.hour + ref_dt.minute/60 + ref_dt.second/3600)
    slot = map_to_kp_slot(ut_hours)

    key = f"{ref_dt.strftime('%Y-%m-%d')}-{slot:02d}"

    # 聚合策略：同一 slot 取 “max 更大” 的样本，同时更新 mean/median —— 保持“峰值优先”
    if key in results:
        if best["max"] > results[key]["max"]:
            results[key] = {"mean": round(best["mean"], 2),
                            "median": round(best["median"], 2),
                            "max": round(best["max"], 2),
                            "origin": best["origin"]}
    else:
        results[key] = {"mean": round(best["mean"], 2),
                        "median": round(best["median"], 2),
                        "max": round(best["max"], 2),
                        "origin": best["origin"]}

    # 每个文件完成就保存一次 —— ### CHANGE 7: 断点续跑更稳
    save_results(results, results_json)

    if DEBUG:
        print(f"Processed {filename} → {key} (max={results[key]['max']}, origin={results[key]['origin']})")

# ------------------ Main ------------------

def main():
    base_dir = "/Users/susie/Documents/UOA/Semester Two/COMPSCI 760/Group Project/code/Group-10_760/datasets/dmsp/data"
    year = 2020
    satellite = 17
    data_dir = os.path.join(base_dir, str(year), str(satellite))

    # 输出目录
    csv_dir = os.path.join(base_dir, "csv")
    os.makedirs(csv_dir, exist_ok=True)

    OUT_CSV = os.path.join(csv_dir, f"{year}-{satellite}.csv")
    RESULTS_JSON = os.path.join(csv_dir, f"{year}-{satellite}-results.json")

    # 恢复历史聚合 —— ### CHANGE 8
    results = load_results(RESULTS_JSON)

    files = sorted([f for f in os.listdir(data_dir) if f.endswith(".nc")])

    ### CHANGE 9: 只保留一层 tqdm 循环（原代码有两遍循环会重复处理）
    for f in tqdm(files, desc="Processing files", unit="file"):
        filepath = os.path.join(data_dir, f)
        process_file(filepath, results, RESULTS_JSON)

    # 写最终 CSV
    df = pd.DataFrame([
        {"date_slot": k,
         "mean": v["mean"],
         "median": v["median"],
         "max": v["max"],
         "origin": v["origin"]}
        for k, v in sorted(results.items())
    ])
    df.to_csv(OUT_CSV, index=False)
    print(f"Final results written to {OUT_CSV}")

if __name__ == "__main__":
    main()
