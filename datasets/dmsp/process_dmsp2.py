# -*- coding: utf-8 -*-
"""
DMSP SSUSI EDR-Aurora nearest-point summarizer @ Poker Flat
- 读取每个 NetCDF 文件
- 将地理坐标 (PF_LAT, PF_LON) 转为 AACGM(磁坐标)（推荐 110km 高度）；
- 在北/南半球网格上查找附近 N 个点（优先阈值内，否则全域最近）
- 计算 mean/median/max，并按 3 小时 Kp slot 聚合
- 每处理一个文件就把 results.json 持久化（断点续跑安全）

主要修复（### FIX）：
1) 南半球查询改为使用与北半球同样的磁纬 mag_lat；
2) 统计时只用“有效值”（>0）计算，避免被 0 填充/负值污染；若阈值内有效点不足，则自动回退到“全域最近”以凑足；
3) 可配置最少有效点阈值 MIN_VALID_POINTS，防止因仅 1~2 个点导致中位数为 0；
4) 新增对负值夹紧为 0 的选项 CLAMP_NEGATIVE_TO_ZERO（默认开启）；
5) 保留原来的断点续跑 JSON；最终 CSV 只写 key 排序后的聚合结果。
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

# ### FIX: 至少需要多少个“有效点”（>0）才直接做统计；不足则回退全域最近
MIN_VALID_POINTS = 1 # 2020年：1-136条非0数据，3-91条非0数据； 2019年：1-306条非0数据。

# ### FIX: 是否把负值夹紧为 0（避免噪声/背景校正导致的负值污染统计）
CLAMP_NEGATIVE_TO_ZERO = True


DEBUG = False  # Toggle detailed print output

# ------------------ Utilities ------------------
def latgrid_for_hemi(lat_grid: np.ndarray, hemi_flag: int) -> np.ndarray:
    """
    根据半球选择用于“距离计算”的纬度网格：
    - 如果 lat_grid 全为非负（常见：0..90 的绝对值），南半球需要取负号镜像；
    - 如果 lat_grid 已含负值（带符号），原样返回。
    hemi_flag: 0=北、1=南
    """
    lat_min = np.nanmin(lat_grid)
    lat_max = np.nanmax(lat_grid)
    # 判断是否为“绝对值纬度网格”
    is_absolute = (lat_min >= 0)
    if hemi_flag == 1 and is_absolute:
        return -lat_grid
    return lat_grid

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

def pick(ds, *cands):
    for n in cands:
        if n in ds.variables:
            return ds[n].values
    raise KeyError(f"None of {cands} in file")

def select_hemisphere_data(ds, hemisphere="north"):
    """Return aurora array, lat, lon, and UT grid by hemisphere."""
    if hemisphere.lower() == "north":
        aur_data = pick(ds, "AUR_ARC_RADIANCE_NORTH", "AUR_ARC_MEDIAN_RAD_NORTH")
        lat_grid = ds["LATITUDE_GEOMAGNETIC_GRID_MAP"].values
        lon_grid = ds["LONGITUDE_GEOMAGNETIC_NORTH_GRID_MAP"].values
        ut_grid = ds.get("UT_N", None)
    else:
        aur_data = pick(ds, "AUR_ARC_RADIANCE_SOUTH", "AUR_ARC_MEDIAN_RAD_SOUTH")
        lat_grid = ds["LATITUDE_GEOMAGNETIC_GRID_MAP"].values
        lon_grid = ds["LONGITUDE_GEOMAGNETIC_SOUTH_GRID_MAP"].values
        ut_grid = ds.get("UT_S", None)
    return aur_data, lat_grid, lon_grid, ut_grid

def query_nearest_points(lat, lon, aur_data, lat_grid, lon_grid, ut_grid=None,
                         n_points=10, max_distance_km=100.0, origin_flag=0):
    """
    先在“距离阈值内”挑最近的 n_points 个非 NaN 点（值允许为任意实数，后续再做过滤/夹紧）。
    若阈值内不足 n_points，则使用 0 填充占位（后续 summarize 会判断是否需要回退到全域最近）。
    之所以保留占位，是为了保留“阈值内有效点的数量信息”。
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

    # 地理坐标转换为磁坐标（推荐 110km）
    mag_lat, mag_lon, _ = aacgmv2.convert_latlon(PF_LAT, PF_LON, ALT_KM, dtime=ref_dt)
    mag_lon_360 = mag_lon % 360

    # 小帮手：在一个半球上做“阈值内最近N个/无则全域最近N个”，并计算统计量
    def summarize(aur, latg, long, utg, hemi_flag):
        # 北半球用 mag_lat，南半球镜像到 -mag_lat（以便在南网格上找最近）
        # q_lat = mag_lat if hemi_flag == 0 else -mag_lat
        
        # 观测点的磁纬：北=+mag_lat，南也用 +mag_lat（观测点本身不镜像）
        

        # ### 关键：对“网格纬度”做半球适配（若是绝对值网格，则南半球取负）
        # latg_use = latgrid_for_hemi(latg, hemi_flag)
        
         # 观测点本身不镜像
        q_lat = mag_lat
        
        pts, ut = query_nearest_points(q_lat, mag_lon_360, aur, latg, long, utg,
                                       N_POINTS, dist_km, origin_flag=hemi_flag)
        
        # 如果全 0，再试试全域最近        
        if not any(p["value"] != 0 for p in pts):
            pts, ut = query_nearest_points_all(q_lat, mag_lon_360, aur, latg, long, utg,
                                               N_POINTS, origin_flag=hemi_flag)

        # 取得“有效值”向量：>0
        vals = np.array([p["value"] for p in pts], dtype=float)
        if CLAMP_NEGATIVE_TO_ZERO:
            vals = np.clip(vals, 0, None)  # 负值夹紧为 0
            
        nz = vals[vals > 0]  # 有效值
        # ### FIX: 若阈值内有效值数量 < MIN_VALID_POINTS，回退到“全域最近”凑数
        if nz.size < MIN_VALID_POINTS:
            pts_all, ut_all = query_nearest_points_all(q_lat, mag_lon_360, aur, latg, long, utg,
                                                       N_POINTS, origin_flag=hemi_flag)
            if pts_all:
                vals = np.array([p["value"] for p in pts_all], dtype=float)
                if CLAMP_NEGATIVE_TO_ZERO:
                    vals = np.clip(vals, 0.0, None)
                nz = vals[vals > 0]
                ut = ut_all if ut_all is not None else ut

        # 如果最终仍没有有效值（全是 0），就用全 0 统计（与原有逻辑兼容）
        target = nz if nz.size > 0 else vals

        mean_val = float(np.mean(target))
        median_val = float(np.median(target))
        max_val = float(np.max(target)) if target.size > 0 else 0.0
        median_val = float(np.median(target)) if target.size else 0.0
        max_val    = float(np.max(target))    if target.size else 0.0

        if DEBUG:
            pos, neg, zer = int((vals>0).sum()), int((vals<0).sum()), int((vals==0).sum())
            print(f"[{filename} hemi={hemi_flag}] pos/neg/zero={pos}/{neg}/{zer} "
                  f"mean={mean_val:.2f} median={median_val:.2f} max={max_val:.2f} "
                  f"UT={hours_to_utc(ut)}")


        return {"mean": mean_val, 
                "median": median_val, 
                "max": max_val, 
                "origin": hemi_flag, 
                "ut": ut}

    # 读取数据（with 自动关闭）
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

    # 选择“max 更大”的半球作为该文件代表
    best = north if north["max"] >= south["max"] else south

    # UT 兜底：网格没有 UT 就用文件名时间
    ut_hours = (best["ut"] if best["ut"] is not None
                else ref_dt.hour + ref_dt.minute/60 + ref_dt.second/3600)
    slot = map_to_kp_slot(ut_hours)
    key = f"{ref_dt.strftime('%Y-%m-%d')}-{slot:02d}"

    # 槽级聚合：若已有记录，用“max 更大”者替换；否则直接填入
    cur = results.get(key)
    cand = {"mean": round(best["mean"], 2),
            "median": round(best["median"], 2),
            "max": round(best["max"], 2),
            "origin": best["origin"]}

    if (cur is None) or (cand["max"] > cur["max"]):
        results[key] = cand

    # 断点续跑：每文件保存一次
    save_results(results, results_json)
        

    if DEBUG:
        print(f"Processed {filename} → {key} (max={results[key]['max']}, origin={results[key]['origin']})")

# ------------------ Main ------------------

def main():
    base_dir = "/Users/susie/Documents/UOA/Semester Two/COMPSCI 760/Group Project/code/Group-10_760/datasets/dmsp/data"
    # base_dir
    year = 2018
    satellite = 17
    data_dir = os.path.join(base_dir, str(year), str(satellite))

    # 输出目录
    curr_dir = os.getcwd()
    
    # 拼接到 csv 子目录
    csv_dir = os.path.join(curr_dir, "csv")
    
    os.makedirs(csv_dir, exist_ok=True)

    OUT_CSV = os.path.join(csv_dir, f"{year}-{satellite}.csv")
    RESULTS_JSON = os.path.join(csv_dir, f"{year}-{satellite}-results.json")

    

    files = sorted([f for f in os.listdir(data_dir) if f.endswith(".nc")])
    print(f"[INFO] scanning: {data_dir}")
    print(f"[INFO] found {len(files)} .nc files")
    if not files:
        print("[WARN] 目录下没有 .nc 文件，CSV 将只有表头；请先确认下载路径/年份/卫星号。")

    # 恢复历史聚合 —— ### CHANGE 8
    results = load_results(RESULTS_JSON)
    
    
    ### CHANGE 9: 只保留一层 tqdm 循环（原代码有两遍循环会重复处理）
    for f in tqdm(files, desc="Processing files", unit="file"):
        filepath = os.path.join(data_dir, f)
        process_file(filepath, results, RESULTS_JSON)
        
    # 结束时再保存一次（即使空也会生成 json，便于检查）
    save_results(results, RESULTS_JSON)

    # —— 写 CSV：即使空也写出表头
    if results:
        df = pd.DataFrame.from_dict(results, orient="index")
        df.index.name = "date_slot"
        df = df.reset_index()[["date_slot","mean","median","max","origin"]]
    else:
        df = pd.DataFrame(columns=["date_slot","mean","median","max","origin"])
    df.to_csv(OUT_CSV, index=False)
    print(f"[INFO] CSV written: {OUT_CSV} (rows={len(df)})")
    print(f"[INFO] JSON written: {RESULTS_JSON} (keys={len(results)})")

if __name__ == "__main__":
    main()
