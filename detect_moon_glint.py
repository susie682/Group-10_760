#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keogram moon-glint detector (single bright round spot, default k=1)


Notebook:
rows = run_detect(
    input_path="/path/to/keogram_or_dir",
    output_dir="./out",
    save_overlay=True, save_mask=True, save_inpaint=True,
    csv_path="./out/detections.csv",
    side_exclude_ratio=0.12, y_min_ratio=0.5,
    r_scale=1.3, r_min=6, r_max=84, size_bias_gamma=1.8,
    strict_r_min=12, roundness_min=0.58, delta_v_min=28, delta_v_zmin=1.2
)
"""

import argparse
import csv
from pathlib import Path
from typing import List, Tuple, Sequence, Union, Optional

import cv2
import numpy as np


# ---------- Helpers ----------

def parse_scales(scales_str: str) -> List[float]:
    return [float(s) for s in scales_str.split(",") if s.strip()]


def nms_circles_distance(points: List[Tuple[int, int, int, float]], thr: float = 0.6) -> List[Tuple[int, int, int, float]]:
    """Distance NMS: drop if center distance <= thr * min(r_i, r_j)."""
    if not points:
        return []
    pts = sorted(points, key=lambda p: p[3], reverse=True)
    keep: List[Tuple[int, int, int, float]] = []
    for x, y, r, s in pts:
        ok = True
        for X, Y, R, S in keep:
            d = np.hypot(x - X, y - Y)
            if d <= thr * min(r, R):
                ok = False
                break
        if ok:
            keep.append((x, y, r, s))
    return keep


def rank_key_size_biased(p: Tuple[int, int, int, float], gamma: float) -> float:
    x, y, r, s = p
    return float(s) * (float(r) ** float(gamma))


# ---------- Vertical bar detection & suppression ----------

def make_vertical_bar_mask(
    img_bgr: np.ndarray,
    white_s_max: int = 60, white_v_min: int = 220,
    purple_h_low: int = 125, purple_h_high: int = 170,
    purple_s_min: int = 60, purple_v_min: int = 60,
    bar_frac: float = 0.55,
    smooth_sigma: float = 3.0
) -> np.ndarray:
    """返回 uint8 掩膜(H,W)。近白或紫色在一整列中占比高（贯穿性强）的列为255。"""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    Hc = hsv[..., 0]
    Sc = hsv[..., 1].astype(np.float32)
    Vc = hsv[..., 2].astype(np.float32)

    white = (Sc <= white_s_max) & (Vc >= white_v_min)
    purple = (Hc >= purple_h_low) & (Hc <= purple_h_high) & (Sc >= purple_s_min) & (Vc >= purple_v_min)
    col_mask = (white | purple).astype(np.uint8) * 255  # [H,W]

    col_frac = col_mask.mean(axis=0) / 255.0           # [W]
    col_frac = cv2.GaussianBlur(col_frac.astype(np.float32)[None, :], (0, 0), smooth_sigma).squeeze()

    bar_cols = (col_frac >= bar_frac).astype(np.uint8)
    bar_mask = np.repeat(bar_cols[None, :], col_mask.shape[0], axis=0) * 255
    return bar_mask


# ---------- Content bounds (auto black-border detection) ----------

def estimate_content_bounds(img_bgr: np.ndarray, black_v_thresh: int = 12) -> Tuple[int, int]:
    """
    用 HSV-V 列均值找非黑内容区的左右边界。
    返回 (left_idx, right_idx)，均含端点。
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    V = hsv[..., 2].astype(np.float32)
    col_mean = V.mean(axis=0)
    W = V.shape[1]

    left = 0
    while left < W and col_mean[left] <= black_v_thresh:
        left += 1
    right = W - 1
    while right >= 0 and col_mean[right] <= black_v_thresh:
        right -= 1

    left = max(0, min(left, W - 1))
    right = max(left, min(right, W - 1))
    return left, right


# ---------- Candidate validation (strict) ----------

def robust_sigma_from_mad(x: np.ndarray) -> float:
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return float(1.4826 * mad + 1e-6)


def validate_candidate(
    V_roi: np.ndarray, x_roi: int, y_roi: int, r: int,
    *,
    roundness_min: float,
    delta_v_min: float,
    delta_v_zmin: float,
    v_global_sigma: float,
    annulus_in: float = 1.5,
    annulus_out: float = 2.4,
) -> Tuple[bool, float, float]:
    """
    对候选进行严格验证（形状安全版本）：
    - 计算圆内均值 V_in、环带背景中位 V_bg（r*annulus_in ~ r*annulus_out）
    - ΔV = V_in - V_bg；z = ΔV / v_global_sigma
    - 圆度：基于二阶矩的特征值比（越接近1越圆），阈值 roundness_min
    返回 (是否通过, delta_v, roundness)
    """
    H, W = V_roi.shape
    r_in = int(max(2, r))
    r_a1 = int(min(max(r_in * annulus_in, r_in + 2), max(H, W)))
    r_a2 = int(min(max(r_in * annulus_out, r_a1 + 1), max(H, W)))

    y0 = max(0, y_roi - r_a2); y1 = min(H, y_roi + r_a2 + 1)
    x0 = max(0, x_roi - r_a2); x1 = min(W, x_roi + r_a2 + 1)
    patch = V_roi[y0:y1, x0:x1]
    ph, pw = patch.shape

    # 坐标网格与掩膜（与 patch 同形状，避免 boolean indexing 维度不匹配）
    Yg, Xg = np.meshgrid(np.arange(y0, y1), np.arange(x0, x1), indexing='ij')  # (ph, pw)
    dist2 = (Yg - y_roi) ** 2 + (Xg - x_roi) ** 2

    m_in = dist2 <= (r_in ** 2)
    m_ring = (dist2 >= (r_a1 ** 2)) & (dist2 <= (r_a2 ** 2))

    if m_in.sum() < 20 or m_ring.sum() < 20:
        return False, 0.0, 0.0

    V_in = float(patch[m_in].mean())
    V_bg = float(np.median(patch[m_ring]))
    delta_v = V_in - V_bg
    z = delta_v / v_global_sigma

    # 圆度（强）：用二阶矩，权重=像素强度
    w = patch[m_in].astype(np.float32) + 1e-6
    yc = (Yg[m_in] - y_roi).astype(np.float32)
    xc = (Xg[m_in] - x_roi).astype(np.float32)
    wsum = float(w.sum())
    mx = float((xc * w).sum() / wsum)
    my = float((yc * w).sum() / wsum)
    x0c = xc - mx
    y0c = yc - my
    cov_xx = float((w * x0c * x0c).sum() / wsum)
    cov_yy = float((w * y0c * y0c).sum() / wsum)
    cov_xy = float((w * x0c * y0c).sum() / wsum)
    trace = cov_xx + cov_yy
    det = cov_xx * cov_yy - cov_xy * cov_xy
    disc = max(trace * trace / 4.0 - det, 0.0)
    lam1 = trace / 2.0 + np.sqrt(disc)
    lam2 = trace / 2.0 - np.sqrt(disc)
    roundness = float(min(lam2 / (lam1 + 1e-9), lam1 / (lam2 + 1e-9)))  # 0~1

    ok = (delta_v >= delta_v_min) and (z >= delta_v_zmin) and (roundness >= roundness_min)
    return ok, delta_v, roundness


# ---------- Core detection (central-only & bar-suppressed) ----------

def detect_candidates(
    img_bgr: np.ndarray,
    *,
    hsv_s_max: int = 80,
    hsv_v_min: int = 200,
    roi_margin_x: float = 0.02,
    roi_margin_top: float = 0.02,
    roi_margin_bottom: float = 0.14,
    scales: Optional[Sequence[float]] = None,
    # isotropy_shrink: kept for compatibility but unused in strict checks
    isotropy_shrink: float = 0.55,
    bar_frac: float = 0.55,
    r_scale: float = 1.3,
    r_min: int = 6,
    r_max: int = 84,
    y_min_ratio: float = 0.5,
    side_exclude_ratio: float = 0.12,
    strict_r_min: int = 12,
    roundness_min: float = 0.58,
    delta_v_min: float = 28.0,
    delta_v_zmin: float = 1.2,
    n_cand_limit: int = 3000,
) -> List[Tuple[int, int, int, float]]:
    """
    返回主体区域内、通过严格验证的候选 (x,y,r,score)：
      - 排除内容区两侧 side_exclude_ratio 的彩色区域；
      - 强制屏蔽贯穿竖条；
      - 仅保留下半区；
      - r >= strict_r_min，圆度 >= roundness_min，ΔV>=delta_v_min 且 z>=delta_v_zmin。
    """
    H, W = img_bgr.shape[:2]
    if scales is None:
        scales = [1.8, 2.4, 3.2, 4.2, 5.4, 6.8]  # 更偏大

    # ROI（避免左右白边、底部刻度）
    x0 = int(W * roi_margin_x)
    x1 = int(W * (1 - roi_margin_x))
    y0 = int(H * roi_margin_top)
    y1 = int(H * (1 - roi_margin_bottom))
    x0 = max(0, min(x0, W - 1)); x1 = max(1, min(x1, W))
    y0 = max(0, min(y0, H - 1)); y1 = max(1, min(y1, H))
    roi = img_bgr[y0:y1, x0:x1].copy()
    h, w = roi.shape[:2]

    # 内容区边界 & 主体横向范围
    c_left, c_right = estimate_content_bounds(img_bgr, black_v_thresh=12)
    inner_left = c_left + int((c_right - c_left + 1) * side_exclude_ratio)
    inner_right = c_right - int((c_right - c_left + 1) * side_exclude_ratio)
    inner_left = max(0, min(inner_left, W - 1))
    inner_right = max(inner_left, min(inner_right, W - 1))

    # 贯穿竖条屏蔽（白/紫）
    bar_mask_full = make_vertical_bar_mask(img_bgr, bar_frac=bar_frac)
    bar_mask_roi = (bar_mask_full[y0:y1, x0:x1] > 0)

    # HSV gate（近白且高亮）并剔除贯穿竖条
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    S = hsv[..., 1].astype(np.float32)
    V = hsv[..., 2].astype(np.float32)
    gate = ((S <= hsv_s_max) & (V >= hsv_v_min)) & (~bar_mask_roi)
    gate = gate.astype(np.uint8)

    # 全局亮度稳健尺度（用于 z 分数）
    v_global_sigma = robust_sigma_from_mad(hsv[..., 2].astype(np.float32))

    # 多尺度 LoG 响应
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
    resp_stack = []
    for s in scales:
        g = cv2.GaussianBlur(gray, (0, 0), float(s))
        lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
        nlog = (-lap) * (float(s) ** 2)
        nlog[nlog < 0] = 0
        resp_stack.append(nlog)
    resp_stack = np.stack(resp_stack, axis=-1)

    # 环形对比度增强
    small = cv2.GaussianBlur(gray, (0, 0), 1.2)
    large = cv2.GaussianBlur(gray, (0, 0), 6.0)
    ring = small - large
    ring[ring < 0] = 0
    score_stack = resp_stack * ring[..., None]
    for kch in range(score_stack.shape[-1]):
        score_stack[..., kch] *= gate

    flat = score_stack.reshape(-1)
    N_cand = min(n_cand_limit, flat.size)
    idxs = np.argpartition(flat, -N_cand)[-N_cand:]

    y_floor = int(H * y_min_ratio)
    cand: List[Tuple[int, int, int, float]] = []

    for idx in idxs:
        y, x, s_idx = np.unravel_index(idx, score_stack.shape)
        sigma = float(scales[s_idx])
        r_est = 1.8 * np.sqrt(2) * sigma * float(r_scale)  # 使用 r_scale
        r = int(np.clip(r_est, r_min, r_max))
        score = float(score_stack[y, x, s_idx])

        xf, yf = int(x + x0), int(y + y0)

        # 位置与尺寸先决
        if yf < y_floor or xf < inner_left or xf > inner_right:
            continue
        if r < strict_r_min:
            continue

        # 严格验证（圆度 + 亮度对比）——在 ROI 的 V 上
        ok, delta_v, roundness = validate_candidate(
            hsv[..., 2], x, y, r,
            roundness_min=roundness_min,
            delta_v_min=delta_v_min,
            delta_v_zmin=delta_v_zmin,
            v_global_sigma=v_global_sigma,
            annulus_in=1.5, annulus_out=2.4
        )
        if not ok:
            continue

        cand.append((xf, yf, int(r), score))

    cand = nms_circles_distance(cand, thr=0.6)
    return cand


def detect_glints_bright_round(
    img_bgr: np.ndarray,
    k: int = 1,
    *,
    hsv_s_max: int = 80,
    hsv_v_min: int = 200,
    roi_margin_x: float = 0.02,
    roi_margin_top: float = 0.02,
    roi_margin_bottom: float = 0.14,
    scales: Optional[Sequence[float]] = None,
    isotropy_shrink: float = 0.55,  # kept for compatibility
    bar_frac: float = 0.55,
    r_scale: float = 1.3,
    r_min: int = 6,
    r_max: int = 84,
    y_min_ratio: float = 0.5,
    size_bias_gamma: float = 1.8,
    side_exclude_ratio: float = 0.12,
    strict_r_min: int = 12,
    roundness_min: float = 0.58,
    delta_v_min: float = 28.0,
    delta_v_zmin: float = 1.2,
) -> Tuple[List[Tuple[int, int, int, float]], np.ndarray]:
    """
    主体区域 + 竖条屏蔽；严格验证后，按 size-biased 分数排序，取 top-k。
    """
    merged = detect_candidates(
        img_bgr,
        hsv_s_max=hsv_s_max, hsv_v_min=hsv_v_min,
        roi_margin_x=roi_margin_x, roi_margin_top=roi_margin_top, roi_margin_bottom=roi_margin_bottom,
        scales=scales, isotropy_shrink=isotropy_shrink,
        bar_frac=bar_frac,
        r_scale=r_scale, r_min=r_min, r_max=r_max, y_min_ratio=y_min_ratio,
        side_exclude_ratio=side_exclude_ratio,
        strict_r_min=strict_r_min, roundness_min=roundness_min,
        delta_v_min=delta_v_min, delta_v_zmin=delta_v_zmin,
    )

    merged.sort(key=lambda p: rank_key_size_biased(p, size_bias_gamma), reverse=True)
    dets = merged[:k] if k > 0 else merged

    H, W = img_bgr.shape[:2]
    mask = np.zeros((H, W), np.uint8)
    for (x, y, r, _) in dets:
        cv2.circle(mask, (x, y), r, 255, -1)
    return dets, mask


def inpaint_with_mask(img_bgr: np.ndarray, mask: np.ndarray, radius=3, method="telea") -> np.ndarray:
    method_flag = cv2.INPAINT_TELEA if method.lower() == "telea" else cv2.INPAINT_NS
    return cv2.inpaint(img_bgr, mask, radius, method_flag)


# ---------- Single image processing (skip-if-none logic) ----------

def process_one(
    in_path: Path,
    out_dir: Path,
    k: int,
    hsv_s_max: int,
    hsv_v_min: int,
    margins: Tuple[float, float, float],
    scales: Sequence[float],
    isotropy_shrink: float,
    save_overlay: bool,
    save_mask: bool,
    save_inpaint: bool,
    inpaint_radius: int,
    inpaint_method: str,
    bar_frac: float,
    r_scale: float,
    r_min: int,
    r_max: int,
    y_min_ratio: float,
    size_bias_gamma: float,
    side_exclude_ratio: float,
    strict_r_min: int,
    roundness_min: float,
    delta_v_min: float,
    delta_v_zmin: float,
    # skip policy
    min_accept_radius: int = 10
) -> Tuple[str, List[Tuple[int, int, int, float]]]:
    img = cv2.imread(str(in_path))
    if img is None:
        raise RuntimeError(f"Failed to read {in_path}")

    dets, mask = detect_glints_bright_round(
        img,
        k=k,
        hsv_s_max=hsv_s_max,
        hsv_v_min=hsv_v_min,
        roi_margin_x=margins[0],
        roi_margin_top=margins[1],
        roi_margin_bottom=margins[2],
        scales=scales,
        isotropy_shrink=isotropy_shrink,
        bar_frac=bar_frac,
        r_scale=r_scale,
        r_min=r_min,
        r_max=r_max,
        y_min_ratio=y_min_ratio,
        size_bias_gamma=size_bias_gamma,
        side_exclude_ratio=side_exclude_ratio,
        strict_r_min=strict_r_min,
        roundness_min=roundness_min,
        delta_v_min=delta_v_min,
        delta_v_zmin=delta_v_zmin,
    )

    stem = in_path.stem

    # 无现象：不输出任何文件
    if not dets or (dets and dets[0][2] < min_accept_radius):
        return stem, []

    if save_overlay:
        overlay = img.copy()
        for (x, y, r, _) in dets:
            cv2.circle(overlay, (x, y), r, (0, 255, 0), 2)
            cv2.drawMarker(overlay, (x, y), (0, 255, 0), markerType=cv2.MARKER_CROSS, thickness=2)
        cv2.imwrite(str(out_dir / f"{stem}_overlay.png"), overlay)

    if save_mask:
        cv2.imwrite(str(out_dir / f"{stem}_mask.png"), mask)

    if save_inpaint:
        inpainted = inpaint_with_mask(img, mask, radius=inpaint_radius, method=inpaint_method)
        cv2.imwrite(str(out_dir / f"{stem}_inpaint.png"), inpainted)

    return stem, dets


# ---------- Notebook-first API ----------

def run_detect(
    input_path: Union[str, Path],
    output_dir: Union[str, Path],
    *,
    k: int = 1,
    scales: str = "1.8,2.4,3.2,4.2,5.4,6.8",
    s_max: int = 80,
    v_min: int = 200,
    margins: str = "0.02,0.02,0.14",
    isotropy: float = 0.55,
    save_overlay: bool = True,
    save_mask: bool = False,
    save_inpaint: bool = False,
    inpaint_radius: int = 3,
    inpaint_method: str = "telea",
    csv_path: Optional[Union[str, Path]] = None,
    exts: str = ".png,.jpg,.jpeg,.tif,.tiff,.bmp",
    bar_frac: float = 0.55,
    r_scale: float = 1.3,
    r_min: int = 6,
    r_max: int = 84,
    y_min_ratio: float = 0.5,
    size_bias_gamma: float = 1.8,
    side_exclude_ratio: float = 0.12,
    strict_r_min: int = 12,
    roundness_min: float = 0.58,
    delta_v_min: float = 28.0,
    delta_v_zmin: float = 1.2,
):
    """
    返回 rows（[file,x,y,radius,score]）。若跳过处理则该行为空字段。
    """
    in_path = Path(input_path)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    scales_list = parse_scales(scales)
    m = [float(x) for x in margins.split(",")]
    margins_tuple = (m[0], m[1], m[2])

    rows = []
    paths: List[Path] = []
    if in_path.is_dir():
        extset = {e.lower().strip() for e in exts.split(",")}
        for p in sorted(in_path.iterdir()):
            if p.is_file() and p.suffix.lower() in extset:
                paths.append(p)
    else:
        paths.append(in_path)

    for p in paths:
        stem, dets = process_one(
            p, out_dir,
            k=k,
            hsv_s_max=s_max,
            hsv_v_min=v_min,
            margins=margins_tuple,
            scales=scales_list,
            isotropy_shrink=isotropy,
            save_overlay=save_overlay or (not save_mask and not save_inpaint),
            save_mask=save_mask,
            save_inpaint=save_inpaint,
            inpaint_radius=inpaint_radius,
            inpaint_method=inpaint_method,
            bar_frac=bar_frac,
            r_scale=r_scale,
            r_min=r_min, r_max=r_max,
            y_min_ratio=y_min_ratio,
            size_bias_gamma=size_bias_gamma,
            side_exclude_ratio=side_exclude_ratio,
            strict_r_min=strict_r_min,
            roundness_min=roundness_min,
            delta_v_min=delta_v_min,
            delta_v_zmin=delta_v_zmin,
        )
        if dets:
            for (x, y, r, s) in dets:
                rows.append([str(p), x, y, r, f"{s:.3f}"])
        else:
            rows.append([str(p), "", "", "", ""])

    if csv_path:
        csv_path = Path(csv_path)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        with csv_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["file", "x", "y", "radius", "score"])
            w.writerows(rows)

    print(f"Processed {len(paths)} image(s). Outputs saved to {out_dir}")
    if csv_path:
        print(f"CSV saved to: {csv_path}")
    return rows


# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description="Detect (and optionally remove) bright round moon-glint in keogram images.")
    ap.add_argument("--input", "-i", required=True, help="Path to an image OR a directory of images")
    ap.add_argument("--output", "-o", required=True, help="Output directory")
    ap.add_argument("--k", type=int, default=1, help="Keep top-k spots (default 1)")
    ap.add_argument("--scales", type=str, default="1.8,2.4,3.2,4.2,5.4,6.8", help="Gaussian sigmas, comma separated")
    ap.add_argument("--s-max", type=int, default=80, help="HSV S upper bound (near-white gate)")
    ap.add_argument("--v-min", type=int, default=200, help="HSV V lower bound (brightness gate)")
    ap.add_argument("--margins", type=str, default="0.02,0.02,0.14", help="ROI margins: left-right, top, bottom (fractions)")
    ap.add_argument("--isotropy", type=float, default=0.55, help="(kept for compatibility)")

    ap.add_argument("--save-overlay", action="store_true", help="Save overlay with circles")
    ap.add_argument("--save-mask", action="store_true", help="Save binary mask")
    ap.add_argument("--save-inpaint", action="store_true", help="Save inpainted image")
    ap.add_argument("--inpaint-radius", type=int, default=3, help="Inpaint radius (px)")
    ap.add_argument("--inpaint-method", type=str, choices=["telea", "ns"], default="telea")
    ap.add_argument("--csv", type=str, default="", help="Optional CSV to write detections (file,x,y,r,score)")
    ap.add_argument("--exts", type=str, default=".png,.jpg,.jpeg,.tif,.tiff,.bmp", help="Valid image extensions for folder input")

    # region/size/brightness controls
    ap.add_argument("--bar-frac", type=float, default=0.55, help="Column fraction threshold to consider vertical bar")
    ap.add_argument("--r-scale", type=float, default=1.3, help="Global radius scale factor")
    ap.add_argument("--r-min", type=int, default=6, help="Minimum radius")
    ap.add_argument("--r-max", type=int, default=84, help="Maximum radius")
    ap.add_argument("--y-min-ratio", type=float, default=0.5, help="Keep candidates with y >= H * ratio")
    ap.add_argument("--size-bias-gamma", type=float, default=1.8, help="Sort by score * r**gamma")
    ap.add_argument("--side-exclude-ratio", type=float, default=0.12, help="Inside content bounds, exclude this fraction on both sides")

    # strict filters
    ap.add_argument("--strict-r-min", type=int, default=12, help="Hard minimum radius to accept a blob")
    ap.add_argument("--roundness-min", type=float, default=0.58, help="Minimum roundness (0~1), higher is rounder")
    ap.add_argument("--delta-v-min", type=float, default=28.0, help="Minimum V contrast (circle mean - annulus median)")
    ap.add_argument("--delta-v-zmin", type=float, default=1.2, help="Minimum robust z-score of ΔV (relative to global V sigma)")

    args = ap.parse_args()

    run_detect(
        input_path=args.input,
        output_dir=args.output,
        k=args.k,
        scales=args.scales,
        s_max=args.s_max,
        v_min=args.v_min,
        margins=args.margins,
        isotropy=args.isotropy,
        save_overlay=args.save_overlay,
        save_mask=args.save_mask,
        save_inpaint=args.save_inpaint,
        inpaint_radius=args.inpaint_radius,
        inpaint_method=args.inpaint_method,
        csv_path=args.csv if args.csv else None,
        exts=args.exts,
        bar_frac=args.bar_frac,
        r_scale=args.r_scale,
        r_min=args.r_min, r_max=args.r_max,
        y_min_ratio=args.y_min_ratio,
        size_bias_gamma=args.size_bias_gamma,
        side_exclude_ratio=args.side_exclude_ratio,
        strict_r_min=args.strict_r_min,
        roundness_min=args.roundness_min,
        delta_v_min=args.delta_v_min,
        delta_v_zmin=args.delta_v_zmin,
    )


if __name__ == "__main__":
    main()
    # 单图
    rows = run_detect(
        input_path="20150101__pfrr_asi3_full-keo-rgb.png",
        output_dir="./keogram-out",
        save_overlay=True, save_inpaint=True, save_mask=True,
        csv_path="./keogram-out/detections.csv"
    )

