#!/usr/bin/env python3
# batch_split_keograms.py

import os
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import itertools
import csv
import re
from pathlib import Path


# ---------- reuse your helper functions ----------
def convert_image_to_RGBarray(image_path):
    img = Image.open(image_path).convert("RGB")
    arr = np.array(img)
    H, W = arr.shape[:2]
    return arr, H, W

def convert_RGB_to_grayscale(image_array):
    return (0.299 * image_array[:, :, 0] +
            0.587 * image_array[:, :, 1] +
            0.114 * image_array[:, :, 2])

def hour_to_col(h, width):
    return int(np.floor((h / 24.0) * width))

def split_into_sections(rgb, n_sections=8):
    H, W = rgb.shape[:2]
    edges = [hour_to_col(24 * i / n_sections, W) for i in range(n_sections)]
    edges.append(W)

    slices, col_ranges, hour_ranges = [], [], []
    for i in range(n_sections):
        c0, c1 = edges[i], edges[i + 1]
        section = rgb[:, c0:c1, :]
        slices.append(section)
        col_ranges.append((c0, c1))
        hour_ranges.append((24 * i / n_sections, 24 * (i + 1) / n_sections))
    return slices, col_ranges, hour_ranges

def plot_sections(slices, hour_ranges, nrows=2, ncols=4, figsize=(16, 6)):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.ravel()
    for i, (ax, section) in enumerate(zip(axes, slices)):
        ax.imshow(section)
        ax.axis("off")
        sh, eh = hour_ranges[i]
        ax.set_title(f"{int(sh)}–{int(eh)} h")
    plt.tight_layout()
    plt.show()

def plot_gray_sections(slices, hour_ranges, nrows=2, ncols=4, figsize=(16, 6)):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = axes.ravel()
    for i, (ax, section) in enumerate(zip(axes, slices)):
        gray = convert_RGB_to_grayscale(section)
        ax.imshow(gray, cmap="gray", aspect="auto")
        ax.axis("off")
        sh, eh = hour_ranges[i]
        ax.set_title(f"{int(sh)}–{int(eh)} h (gray)")
    plt.tight_layout()
    plt.show()

def print_intensity_stats_by_block(rgb, n_sections=8):
    gray = convert_RGB_to_grayscale(rgb)
    intensity_per_col = gray.mean(axis=0)
    W = gray.shape[1]
    for i in range(n_sections):
        start_h = 24 * i / n_sections
        end_h   = 24 * (i + 1) / n_sections
        c0 = hour_to_col(start_h, W)
        c1 = hour_to_col(end_h,   W)
        vals = intensity_per_col[c0:c1]
        print(f"{int(start_h):02d}–{int(end_h):02d}h "
              f"(cols {c0}–{c1-1}): mean={vals.mean():.2f}, median={np.median(vals):.2f}")

def save_sections(slices, out_dir, base_name="slice"):
    os.makedirs(out_dir, exist_ok=True)
    out_paths = []
    for i, section in enumerate(slices, 1):
        p = os.path.join(out_dir, f"{base_name}_{i:02d}.png")
        Image.fromarray(section).save(p)
        out_paths.append(p)
    return out_paths

#below functions are for the csv
def _extract_yyyymmdd(stem: str) -> str:
    """Return first 8-digit date (YYYYMMDD) from a filename stem."""
    m = re.search(r"\d{8}", stem)
    if not m:
        raise ValueError(f"Cannot find YYYYMMDD in '{stem}'")
    return m.group(0)

def save_grid(slices, hour_ranges, out_path, nrows=2, ncols=4):
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 6))
    axes = axes.ravel()
    for i, (ax, section) in enumerate(zip(axes, slices)):
        ax.imshow(section)
        ax.axis("off")
        sh, eh = hour_ranges[i]
        ax.set_title(f"{int(sh)}–{int(eh)} h")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

def write_segment_stats_csv(input_dir="keogram-out2",
                            output_csv="keogram_segment_stats.csv",
                            n_sections=8):
    """
    For every *inpaint.png (and *inpant.png) under input_dir:
      - compute mean/median/max for each of 8 segments,
      - append one CSV row per segment: YYYY-MM-DD-h1:h2, mean, median, max
    """
    in_dir = Path(input_dir)
    files = sorted(set(in_dir.glob("**/*inpaint.png")) |
                   set(in_dir.glob("**/*inpant.png")))
    if not files:
        print(f"[warn] No inpainted PNGs found under {in_dir}")
        return

    with open(output_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["segment", "mean", "median", "max"])  # header

        for p in files:
            rgb, H, W = convert_image_to_RGBarray(p)
            gray = convert_RGB_to_grayscale(rgb)
            yyyymmdd = _extract_yyyymmdd(p.stem)
            date_fmt = f"{yyyymmdd[0:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}"

            for i in range(n_sections):
                start_h = int(24 * i / n_sections)
                end_h   = int(24 * (i + 1) / n_sections)
                c0 = hour_to_col(start_h, W)
                c1 = hour_to_col(end_h,   W)

                seg_vals = gray[:, c0:c1].ravel()
                mean_v   = float(seg_vals.mean())
                median_v = float(np.median(seg_vals))
                max_v    = float(seg_vals.max())

                seg_label = f"{date_fmt}-{start_h}:{end_h}"
                w.writerow([seg_label, f"{mean_v:.2f}", f"{median_v:.2f}", f"{max_v:.2f}"])

            print(f"[ok] {p.name} -> 8 rows")

    print(f"[done] Wrote segment stats to {output_csv}")

def process_all_with_csv(input_dir="keogram-out2",
                         output_dir="segments-out",
                         output_csv="keogram_segment_stats.csv",
                         n_sections=8):
    in_dir = Path(input_dir)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    files = sorted(set(in_dir.glob("**/*inpaint.png")) |
                   set(in_dir.glob("**/*inpant.png")))
    if not files:
        print(f"[warn] No inpainted PNGs found under {in_dir}")
        return

    # open CSV once; append rows as we go
    with open(output_csv, "w", newline="") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["segment", "mean", "median", "max"])

        for p in files:
            print(f"[info] {p.name}")
            rgb, H, W = convert_image_to_RGBarray(p)
            slices, col_ranges, hour_ranges = split_into_sections(rgb, n_sections=n_sections)

            # save slices
            stem = p.stem
            per_img_out = out_root / stem
            out_paths = save_sections(slices, per_img_out, base_name=stem)

            # save grid preview (no display)
            save_grid(slices, hour_ranges, per_img_out / f"{stem}_grid.png")

            # per-segment stats -> CSV rows
            gray = convert_RGB_to_grayscale(rgb)
            for i in range(n_sections):
                start_h = int(24 * i / n_sections)
                end_h   = int(24 * (i + 1) / n_sections)
                c0 = hour_to_col(start_h, W)
                c1 = hour_to_col(end_h,   W)
                seg_vals = gray[:, c0:c1].ravel()
                mean_v   = float(seg_vals.mean())
                median_v = float(np.median(seg_vals))
                max_v    = float(seg_vals.max())

                yyyymmdd = _extract_yyyymmdd(stem)
                date_fmt = f"{yyyymmdd[0:4]}-{yyyymmdd[4:6]}-{yyyymmdd[6:8]}"
                seg_label = f"{date_fmt}-{start_h}:{end_h}"
                w.writerow([seg_label, f"{mean_v:.2f}", f"{median_v:.2f}", f"{max_v:.2f}"])

            print(f"[ok] Saved {len(out_paths)} slices + grid, wrote CSV rows for {p.name}")

    print(f"[done] All files processed. CSV -> {output_csv}, slices -> {output_dir}")

# ---------- batch driver ----------
def process_all(input_dir="keogram-out2", output_dir="segments-out", n_sections=8):
    in_dir = Path(input_dir)
    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    candidates = sorted(set(in_dir.glob("**/*inpaint.png")) |
                        set(in_dir.glob("**/*inpant.png")))

    if not candidates:
        print(f"[warn] No matching files in {in_dir}")
        return

    for img_path in candidates:
        print(f"[info] Processing {img_path.name}")
        rgb, H, W = convert_image_to_RGBarray(img_path)
        slices, col_ranges, hour_ranges = split_into_sections(rgb, n_sections=n_sections)

        # per-image folder
        stem = img_path.stem
        per_img_out = out_root / stem
        out_paths = save_sections(slices, per_img_out, base_name=stem)

        # also save a grid preview
        fig, axes = plt.subplots(2, 4, figsize=(16, 6))
        for ax, section, (sh, eh) in zip(axes.ravel(), slices, hour_ranges):
            ax.imshow(section)
            ax.axis("off")
            ax.set_title(f"{int(sh)}–{int(eh)} h")
        plt.tight_layout()
        fig.savefig(per_img_out / f"{stem}_grid.png", dpi=150)
        plt.close(fig)

        print(f"[ok] Saved {len(out_paths)} slices + grid -> {per_img_out}")

if __name__ == "__main__":
    # process_all(input_dir="keogram-out2", output_dir="segments-out", n_sections=8)
    process_all_with_csv(
        input_dir="keogram-out2",
        output_dir="segments-out",
        output_csv="keogram_segment_stats.csv",
        n_sections=8
    )
