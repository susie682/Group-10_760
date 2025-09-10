#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Moon-glint detector for keograms: single bright round spot (default k=1)
- Near-white + bright gate (HSV)
- Multi-scale LoG + ring-contrast score
- Global maximum (or top-k with NMS)
- Optional inpaint to remove the spot(s)
Author: you :)
"""

import argparse
import csv
import os
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np


# ---------- Core detection ----------

def parse_scales(scales_str: str) -> List[float]:
    return [float(s) for s in scales_str.split(",") if s.strip()]


def nms_points(points: List[Tuple[int, int, int, float]], iou_thr=0.3) -> List[Tuple[int, int, int, float]]:
    """
    Non-maximum suppression for circle detections.
    points: list of (x, y, r, score). IOU computed on circles' bounding boxes approximation.
    """
    if not points:
        return []
    # sort by score desc
    pts = sorted(points, key=lambda p: p[3], reverse=True)
    keep = []
    while pts:
        best = pts.pop(0)
        keep.append(best)
        bx, by, br, _ = best
        b_area = (2 * br) * (2 * br)
        new_pts = []
        for (x, y, r, s) in pts:
            # bbox overlap
            w = max(0, min(bx + br, x + r) - max(bx - br, x - r))
            h = max(0, min(by + br, y + r) - max(by - br, y - r))
            inter = w * h
            area = (2 * r) * (2 * r)
            iou = inter / (b_area + area - inter + 1e-9)
            if iou <= iou_thr:
                new_pts.append((x, y, r, s))
        pts = new_pts
    return keep


def detect_glints_bright_round(
    img_bgr: np.ndarray,
    k: int = 1,
    hsv_s_max: int = 80,
    hsv_v_min: int = 200,
    roi_margin_x: float = 0.02,
    roi_margin_top: float = 0.02,
    roi_margin_bottom: float = 0.14,
    scales: List[float] = None,
    isotropy_shrink: float = 0.55,
) -> Tuple[List[Tuple[int, int, int, float]], np.ndarray]:
    """
    Return: (list of detections [(x,y,r,score), ...] in full-image coords, mask[H,W] uint8)
    """
    H, W = img_bgr.shape[:2]
    if scales is None:
        scales = [1.5, 2.2, 3.0, 3.8, 4.6, 5.4]

    # ROI （避免左右白条/下方刻度轴）
    x0 = int(W * roi_margin_x)
    x1 = int(W * (1 - roi_margin_x))
    y0 = int(H * roi_margin_top)
    y1 = int(H * (1 - roi_margin_bottom))
    x0 = max(0, min(x0, W - 1)); x1 = max(1, min(x1, W))
    y0 = max(0, min(y0, H - 1)); y1 = max(1, min(y1, H))
    roi = img_bgr[y0:y1, x0:x1].copy()
    h, w = roi.shape[:2]

    # HSV 近白高亮门控
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    S = hsv[..., 1].astype(np.float32)
    V = hsv[..., 2].astype(np.float32)
    gate = ((S <= hsv_s_max) & (V >= hsv_v_min)).astype(np.uint8)

    # 灰度 + 多尺度 LoG（亮斑取 -Lap）
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY).astype(np.float32)
    resp_stack = []
    for s in scales:
        g = cv2.GaussianBlur(gray, (0, 0), s)
        lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
        nlog = (-lap) * (s ** 2)
        nlog[nlog < 0] = 0
        resp_stack.append(nlog)
    resp_stack = np.stack(resp_stack, axis=-1)

    # 环形对比度（小模糊 - 大模糊）增强
    small = cv2.GaussianBlur(gray, (0, 0), 1.2)
    large = cv2.GaussianBlur(gray, (0, 0), 6.0)
    ring = small - large
    ring[ring < 0] = 0
    score_stack = resp_stack * ring[..., None]
    for kch in range(score_stack.shape[-1]):
        score_stack[..., kch] *= gate

    # 取 top-k：先取若干候选，再用圆形 NMS
    flat = score_stack.reshape(-1)
    # 取前 N 个候选（过大图像时避免全排序）
    N_cand = min(2000, flat.size)
    idxs = np.argpartition(flat, -N_cand)[-N_cand:]
    cand = []
    for idx in idxs:
        y, x, s_idx = np.unravel_index(idx, score_stack.shape)
        sigma = scales[s_idx]
        r = int(np.clip(1.8 * np.sqrt(2) * sigma, 3, 30))
        score = score_stack[y, x, s_idx]

        # 各向同性检查（更圆则分数更可靠）
        g = cv2.GaussianBlur(gray, (0, 0), sigma)
        dxx = cv2.Sobel(g, cv2.CV_32F, 2, 0, ksize=3)
        dyy = cv2.Sobel(g, cv2.CV_32F, 0, 2, ksize=3)
        dxy = cv2.Sobel(g, cv2.CV_32F, 1, 1, ksize=3)
        ys, xs = max(1, y - 2), max(1, x - 2)
        ye, xe = min(h - 2, y + 2), min(w - 2, x + 2)
        Hxx = float(np.mean(dxx[ys:ye + 1, xs:xe + 1]))
        Hyy = float(np.mean(dyy[ys:ye + 1, xs:xe + 1]))
        Hxy = float(np.mean(dxy[ys:ye + 1, xs:xe + 1]))
        trace = Hxx + Hyy
        det = Hxx * Hyy - Hxy * Hxy
        disc = max(trace * trace / 4 - det, 0.0)
        lam1 = trace / 2 + np.sqrt(disc)
        lam2 = trace / 2 - np.sqrt(disc)
        iso = min(abs(lam1 / (lam2 + 1e-9)), abs(lam2 / (lam1 + 1e-9)))
        if iso < isotropy_shrink:
            r = int(max(3, 0.7 * r))

        # 转回全图坐标
        cand.append((int(x + x0), int(y + y0), int(r), float(score)))

    # NMS + 取前 k
    cand = nms_points(cand, iou_thr=0.3)
    cand = sorted(cand, key=lambda p: p[3], reverse=True)
    if k > 0:
        cand = cand[:k]

    # 生成掩膜
    mask = np.zeros((H, W), np.uint8)
    for (x, y, r, _) in cand:
        cv2.circle(mask, (x, y), r, 255, -1)

    return cand, mask


def inpaint_with_mask(img_bgr: np.ndarray, mask: np.ndarray, radius=3, method="telea") -> np.ndarray:
    method_flag = cv2.INPAINT_TELEA if method.lower() == "telea" else cv2.INPAINT_NS
    return cv2.inpaint(img_bgr, mask, radius, method_flag)


# ---------- CLI ----------

def process_one(
    in_path: Path,
    out_dir: Path,
    k: int,
    hsv_s_max: int,
    hsv_v_min: int,
    margins: Tuple[float, float, float],
    scales: List[float],
    isotropy_shrink: float,
    save_overlay: bool,
    save_mask: bool,
    save_inpaint: bool,
    inpaint_radius: int,
    inpaint_method: str,
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
    )

    stem = in_path.stem
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


def main():
    ap = argparse.ArgumentParser(description="Detect (and optionally remove) bright round moon-glint in keogram images.")
    ap.add_argument("--input", "-i", required=True, help="Path to an image OR a directory of images")
    ap.add_argument("--output", "-o", required=True, help="Output directory")
    ap.add_argument("--k", type=int, default=1, help="Keep top-k spots (default 1)")
    ap.add_argument("--scales", type=str, default="1.5,2.2,3.0,3.8,4.6,5.4", help="Gaussian sigmas, comma separated")
    ap.add_argument("--s-max", type=int, default=80, help="HSV S upper bound (near-white gate)")
    ap.add_argument("--v-min", type=int, default=200, help="HSV V lower bound (brightness gate)")
    ap.add_argument("--margins", type=str, default="0.02,0.02,0.14", help="ROI margins: left-right, top, bottom (fractions)")
    ap.add_argument("--isotropy", type=float, default=0.55, help="Isotropy threshold; lower -> more shrink")
    ap.add_argument("--save-overlay", action="store_true", help="Save overlay with circles")
    ap.add_argument("--save-mask", action="store_true", help="Save binary mask")
    ap.add_argument("--save-inpaint", action="store_true", help="Save inpainted image")
    ap.add_argument("--inpaint-radius", type=int, default=3, help="Inpaint radius (px)")
    ap.add_argument("--inpaint-method", type=str, choices=["telea", "ns"], default="telea")
    ap.add_argument("--csv", type=str, default="", help="Optional CSV to write detections (file,x,y,r,score)")
    ap.add_argument("--exts", type=str, default=".png,.jpg,.jpeg,.tif,.tiff,.bmp", help="Valid image extensions for folder input")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    scales = parse_scales(args.scales)
    m = [float(x) for x in args.margins.split(",")]
    margins = (m[0], m[1], m[2])

    rows = []
    paths = []
    if in_path.is_dir():
        exts = {e.lower().strip() for e in args.exts.split(",")}
        for p in sorted(in_path.iterdir()):
            if p.is_file() and p.suffix.lower() in exts:
                paths.append(p)
    else:
        paths.append(in_path)

    for p in paths:
        stem, dets = process_one(
            p, out_dir,
            k=args.k,
            hsv_s_max=args.s_max,
            hsv_v_min=args.v_min,
            margins=margins,
            scales=scales,
            isotropy_shrink=args.isotropy,
            save_overlay=args.save_overlay or (not args.save_mask and not args.save_inpaint),  # 默认至少保存overlay
            save_mask=args.save_mask,
            save_inpaint=args.save_inpaint,
            inpaint_radius=args.inpaint_radius,
            inpaint_method=args.inpaint_method,
        )
        if dets:
            for (x, y, r, s) in dets:
                rows.append([str(p), x, y, r, f"{s:.3f}"])
        else:
            rows.append([str(p), "", "", "", ""])

    if args.csv:
        csv_path = Path(args.csv)
        with csv_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["file", "x", "y", "radius", "score"])
            w.writerows(rows)

    print(f"Processed {len(paths)} image(s). Outputs saved to {out_dir}")


if __name__ == "__main__":
    main()