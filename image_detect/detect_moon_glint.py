#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Keogram moon-glint detector (single bright round spot, default k=1)
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
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    Hc = hsv[..., 0]
    Sc = hsv[..., 1].astype(np.float32)
    Vc = hsv[..., 2].astype(np.float32)

    white = (Sc <= white_s_max) & (Vc >= white_v_min)
    purple = (Hc >= purple_h_low) & (Hc <= purple_h_high) & (Sc >= purple_s_min) & (Vc >= purple_v_min)
    col_mask = (white | purple).astype(np.uint8) * 255

    col_frac = col_mask.mean(axis=0) / 255.0
    col_frac = cv2.GaussianBlur(col_frac.astype(np.float32)[None, :], (0, 0), smooth_sigma).squeeze()

    bar_cols = (col_frac >= bar_frac).astype(np.uint8)
    bar_mask = np.repeat(bar_cols[None, :], col_mask.shape[0], axis=0) * 255
    return bar_mask


# ---------- Content bounds (auto black-border detection) ----------

def estimate_content_bounds(img_bgr: np.ndarray, black_v_thresh: int = 12) -> Tuple[int, int]:
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
    H, W = V_roi.shape
    r_in = int(max(2, r))
    r_a1 = int(min(max(r_in * annulus_in, r_in + 2), max(H, W)))
    r_a2 = int(min(max(r_in * annulus_out, r_a1 + 1), max(H, W)))

    y0 = max(0, y_roi - r_a2); y1 = min(H, y_roi + r_a2 + 1)
    x0 = max(0, x_roi - r_a2); x1 = min(W, x_roi + r_a2 + 1)
    patch = V_roi[y0:y1, x0:x1]

    Yg, Xg = np.meshgrid(np.arange(y0, y1), np.arange(x0, x1), indexing='ij')
    dist2 = (Yg - y_roi) ** 2 + (Xg - x_roi) ** 2

    m_in = dist2 <= (r_in ** 2)
    m_ring = (dist2 >= (r_a1 ** 2)) & (dist2 <= (r_a2 ** 2))

    if m_in.sum() < 20 or m_ring.sum() < 20:
        return False, 0.0, 0.0

    V_in = float(patch[m_in].mean())
    V_bg = float(np.median(patch[m_ring]))
    delta_v = V_in - V_bg
    z = delta_v / v_global_sigma

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
    roundness = float(min(lam2 / (lam1 + 1e-9), lam1 / (lam2 + 1e-9)))

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
    isotropy_shrink: float = 0.55,
    bar_frac: float = 0.55,
    r_scale: float = 1.3,
    r_min: int = 6,
    r_max: int = 84,
    y_min_ratio: float = 0.6,             # ← 下 40% 搜索
    side_exclude_ratio: float = 0.12,
    strict_r_min: int = 12,
    roundness_min: float = 0.58,
    delta_v_min: float = 28.0,
    delta_v_zmin: float = 1.2,
    n_cand_limit: int = 3000,
) -> List[Tuple[int, int, int, float]]:
    H, W = img_bgr.shape[:2]
    if scales is None:
        scales = [1.8, 2.4, 3.2, 3.8, 4.6, 5.4]  # 原始偏好

    # ROI
    x0 = int(W * roi_margin_x); x1 = int(W * (1 - roi_margin_x))
    y0 = int(H * roi_margin_top); y1 = int(H * (1 - roi_margin_bottom))
    x0 = max(0, min(x0, W - 1)); x1 = max(1, min(x1, W))
    y0 = max(0, min(y0, H - 1)); y1 = max(1, min(y1, H))
    roi = img_bgr[y0:y1, x0:x1].copy()

    # 横向主体范围
    c_left, c_right = estimate_content_bounds(img_bgr, black_v_thresh=12)
    inner_left = c_left + int((c_right - c_left + 1) * side_exclude_ratio)
    inner_right = c_right - int((c_right - c_left + 1) * side_exclude_ratio)
    inner_left = max(0, min(inner_left, W - 1))
    inner_right = max(inner_left, min(inner_right, W - 1))

    # 竖条屏蔽
    bar_mask_full = make_vertical_bar_mask(img_bgr, bar_frac=bar_frac)
    bar_mask_roi = (bar_mask_full[y0:y1, x0:x1] > 0)

    # HSV gate
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    S = hsv[..., 1].astype(np.float32)
    V = hsv[..., 2].astype(np.float32)
    gate = ((S <= hsv_s_max) & (V >= hsv_v_min)) & (~bar_mask_roi)
    gate = gate.astype(np.uint8)

    # 亮度稳健尺度
    v_global_sigma = robust_sigma_from_mad(hsv[..., 2].astype(np.float32))

    # LoG 多尺度
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
    resp_stack = []
    for s in scales:
        g = cv2.GaussianBlur(gray, (0, 0), float(s))
        lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
        nlog = (-lap) * (float(s) ** 2)
        nlog[nlog < 0] = 0
        resp_stack.append(nlog)
    resp_stack = np.stack(resp_stack, axis=-1)

    # 环形对比度
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
        r_est = 1.8 * np.sqrt(2) * sigma * float(r_scale)
        r = int(np.clip(r_est, r_min, r_max))
        score = float(score_stack[y, x, s_idx])

        xf, yf = int(x + x0), int(y + y0)
        if yf < y_floor or xf < inner_left or xf > inner_right:
            continue
        if r < strict_r_min:
            continue

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
    isotropy_shrink: float = 0.55,
    bar_frac: float = 0.55,
    r_scale: float = 1.3,
    r_min: int = 6,
    r_max: int = 84,
    y_min_ratio: float = 0.6,      # ← 下 40%
    size_bias_gamma: float = 1.8,
    side_exclude_ratio: float = 0.12,
    strict_r_min: int = 12,
    roundness_min: float = 0.58,
    delta_v_min: float = 28.0,
    delta_v_zmin: float = 1.2,
) -> Tuple[List[Tuple[int, int, int, float]], np.ndarray]:
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


# ---------- 强力去亮点：径向远取样 + Poisson ----------

def _auto_grow_px_from_mask(mask: np.ndarray, ratio: float = 0.40, min_px: int = 4, max_px: int = 32) -> int:
    m = (mask > 0).astype(np.uint8)
    if m.max() == 0:
        return min_px
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 3)
    r_eff = float(dist.max())
    return int(max(min_px, min(max_px, r_eff * ratio)))


def _expand_mask_by_brightness(img_bgr: np.ndarray, mask: np.ndarray, max_iter: int = 16) -> np.ndarray:
    """更激进的亮度驱动扩圈：把高亮晕圈尽量纳入。"""
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    V = hsv[..., 2].astype(np.float32)
    m = (mask > 0).astype(np.uint8)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

    for _ in range(max_iter):
        dil = cv2.dilate(m, k)
        border = (dil == 1) & (m == 0)
        if not border.any():
            break
        # 远环统计
        far = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))) - \
              cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (17, 17)))
        far_idx = far > 0
        V_bg = float(np.median(V[far_idx])) if far_idx.any() else float(np.median(V))
        Vin = float(np.median(V[m > 0])) if (m > 0).any() else V_bg
        # 阈值：更严格（取更远处亮度）
        thr = V_bg + max(12.0, 0.65 * (Vin - V_bg))
        grow_idx = border & (V >= thr)
        if not grow_idx.any():
            break
        m[grow_idx] = 1
    return (m * 255).astype(np.uint8)


def inpaint_radial_poisson(
    img_bgr: np.ndarray,
    mask: np.ndarray,
    *,
    offset_inner: int = 18,   # 与边界保持的最小距离（像素）
    offset_outer: int = 38,   # 取样最远距离（像素）
    feather_sigma: float = 2.2,
    post_sigma: float = 1.6
) -> np.ndarray:
    """
    逻辑：
      1) 掩膜按亮度自适应扩圈（含晕圈）；
      2) 对掩膜内每个像素，沿“远离质心”的径向，从 [offset_inner, offset_outer] 的随机位置采样颜色；
      3) 用该颜色场替换掩膜内像素；
      4) 对替换区域轻度模糊后，使用 Poisson NORMAL_CLONE 无缝融合。
    """
    if mask is None or mask.max() == 0:
        return img_bgr.copy()

    h, w = img_bgr.shape[:2]

    # 1) 扩圈
    mask_exp = _expand_mask_by_brightness(img_bgr, mask, max_iter=16)

    # 2) 径向远取样坐标
    ys, xs = np.where(mask_exp > 0)
    if ys.size == 0:
        return img_bgr.copy()

    M = cv2.moments(mask_exp)
    if M["m00"] > 0:
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
    else:
        cx = float(xs.mean()); cy = float(ys.mean())

    dx = xs.astype(np.float32) - float(cx)
    dy = ys.astype(np.float32) - float(cy)
    norm = np.sqrt(dx * dx + dy * dy) + 1e-6
    vx = dx / norm
    vy = dy / norm

    # 每个点一个随机取样半径（避免纹理同步）
    rng = np.random.default_rng(12345)
    sample_r = rng.uniform(offset_inner, offset_outer, size=xs.shape[0]).astype(np.float32)

    samp_x = xs.astype(np.float32) + vx * sample_r
    samp_y = ys.astype(np.float32) + vy * sample_r

    # 边界裁剪
    samp_x = np.clip(samp_x, 0, w - 1)
    samp_y = np.clip(samp_y, 0, h - 1)

    # 双线性采样
    def bilinear_sample(img, x, y):
        x0 = np.floor(x).astype(np.int32)
        y0 = np.floor(y).astype(np.int32)
        x1 = np.clip(x0 + 1, 0, img.shape[1] - 1)
        y1 = np.clip(y0 + 1, 0, img.shape[0] - 1)
        wa = (x1 - x) * (y1 - y)
        wb = (x - x0) * (y1 - y)
        wc = (x1 - x) * (y - y0)
        wd = (x - x0) * (y - y0)
        res = (img[y0, x0] * wa[:, None] +
               img[y0, x1] * wb[:, None] +
               img[y1, x0] * wc[:, None] +
               img[y1, x1] * wd[:, None])
        return res

    sampled = bilinear_sample(img_bgr.astype(np.float32), samp_x, samp_y)
    prefill = img_bgr.copy().astype(np.float32)
    prefill[ys, xs] = sampled

    # 3) 轻微模糊与羽化混合，使替换区更柔和
    prefill_blur = cv2.GaussianBlur(prefill, (0, 0), post_sigma)
    soft = cv2.GaussianBlur(mask_exp.astype(np.float32), (0, 0), feather_sigma)
    a = (soft / 255.0)[..., None]
    blended = img_bgr.astype(np.float32) * (1.0 - a) + prefill_blur * a
    blended = np.clip(blended, 0, 255).astype(np.uint8)

    # 4) Poisson NORMAL_CLONE（比 MIXED 更“覆盖式”，不把亮梯度带回）
    center = (int(round(cx)), int(round(cy)))
    out = cv2.seamlessClone(blended, img_bgr, mask_exp, center, cv2.NORMAL_CLONE)
    return out


# ---------- Single image processing ----------

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
    inpaint_radius: int,   # 兼容参数（未用）
    inpaint_method: str,   # 兼容参数（未用）
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
    valid = bool(dets) and (dets[0][2] >= min_accept_radius)

    if valid and save_overlay:
        overlay = img.copy()
        for (x, y, r, _) in dets:
            cv2.circle(overlay, (x, y), r, (0, 255, 0), 2)
            cv2.drawMarker(overlay, (x, y), (0, 255, 0), markerType=cv2.MARKER_CROSS, thickness=2)
        cv2.imwrite(str(out_dir / f"{stem}_overlay.png"), overlay)

    if valid and save_mask:
        cv2.imwrite(str(out_dir / f"{stem}_mask.png"), mask)

    if save_inpaint:
        if valid:
            inpainted = inpaint_radial_poisson(
                img, mask,
                offset_inner=18,   # 可以适当再增大到 22~28
                offset_outer=38,   # 或 40~50
                feather_sigma=2.2,
                post_sigma=1.6
            )
        else:
            inpainted = img.copy()
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
    inpaint_radius: int = 3,      # 兼容参数（未用）
    inpaint_method: str = "telea",# 兼容参数（未用）
    csv_path: Optional[Union[str, Path]] = None,
    exts: str = ".png,.jpg,.jpeg,.tif,.tiff,.bmp",
    bar_frac: float = 0.55,
    r_scale: float = 1.3,
    r_min: int = 6,
    r_max: int = 84,
    y_min_ratio: float = 0.6,     # ← 下 40%
    size_bias_gamma: float = 1.8,
    side_exclude_ratio: float = 0.12,
    strict_r_min: int = 12,
    roundness_min: float = 0.58,
    delta_v_min: float = 28.0,
    delta_v_zmin: float = 1.2,
):
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
    ap.add_argument("--inpaint-radius", type=int, default=3, help="Inpaint radius (px) [compat]")
    ap.add_argument("--inpaint-method", type=str, choices=["telea", "ns"], default="telea", help="Inpaint method [compat]")
    ap.add_argument("--csv", type=str, default="", help="Optional CSV to write detections (file,x,y,r,score)")
    ap.add_argument("--exts", type=str, default=".png,.jpg,.jpeg,.tif,.tiff,.bmp", help="Valid image extensions for folder input")

    ap.add_argument("--bar-frac", type=float, default=0.55, help="Column fraction threshold to consider vertical bar")
    ap.add_argument("--r-scale", type=float, default=1.3, help="Global radius scale factor")
    ap.add_argument("--r-min", type=int, default=6, help="Minimum radius")
    ap.add_argument("--r-max", type=int, default=84, help="Maximum radius")
    ap.add_argument("--y-min-ratio", type=float, default=0.6, help="Keep candidates with y >= H * ratio (0.6 = bottom 40%)")
    ap.add_argument("--size-bias-gamma", type=float, default=1.8, help="Sort by score * r**gamma")
    ap.add_argument("--side-exclude-ratio", type=float, default=0.12, help="Inside content bounds, exclude this fraction on both sides")

    ap.add_argument("--strict-r-min", type=int, default=12, help="Hard minimum radius to accept a blob")
    ap.add_argument("--roundness-min", type=float, default=0.58, help="Minimum roundness (0~1), higher is rounder")
    ap.add_argument("--delta-v-min", type=float, default=28.0, help="Minimum V contrast (circle mean - annulus median)")
    ap.add_argument("--delta-v-zmin", type=float, default=1.2, help="Minimum robust z-score of ΔV")

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


#if __name__ == "__main__":
#    main()

rows = run_detect(
    input_path="./2015",
    output_dir="./keogram-out",
    k=1, save_overlay=True, save_mask=True, save_inpaint=True,
    csv_path="./keogram-out/detections.csv"
) 
