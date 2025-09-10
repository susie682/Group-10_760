# keogram_process.py
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# ---------- Step 1: load image as RGB numpy array ----------
def convert_image_to_RGBarray(image_path):
    """
    Returns (arr, H, W) where arr is an RGB numpy array of shape (H, W, 3).
    """
    img = Image.open(image_path).convert("RGB")   # PIL Image
    arr = np.array(img)                           # -> numpy array
    H, W = arr.shape[:2]
    return arr, H, W

# Optional: grayscale + intensity per column (1 value per time column)
def convert_RGB_to_grayscale(image_array):
    return (0.299 * image_array[:, :, 0] +
            0.587 * image_array[:, :, 1] +
            0.114 * image_array[:, :, 2])

def hour_to_col(h, width):
    """Map hour-of-day in [0, 24] to a column index [0, width]."""
    return int(np.floor((h / 24.0) * width))

# ---------- Step 2: split RGB image into N equal vertical sections ----------
def split_into_sections(rgb, n_sections=8):
    """
    Split the image into n_sections vertical slices.
    Returns:
      slices: list of (H, Wi, 3) arrays
      col_ranges: list of (c0, c1) inclusive/exclusive column ranges
      hour_ranges: list of (start_hour, end_hour)
    """
    H, W = rgb.shape[:2]
    # Boundaries via hour mapping (so 3h blocks line up exactly)
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

# ---------- Step 3: plot the 8 sections with matplotlib ----------
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

# ---------- NEW: plot grayscale sections ----------
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

# ---------- Optional: compute & print intensity stats per 3-hour block ----------
def print_intensity_stats_by_block(rgb, n_sections=8):
    gray = convert_RGB_to_grayscale(rgb)
    intensity_per_col = gray.mean(axis=0)  # average over rows -> 1 value per column (time)
    W = gray.shape[1]
    for i in range(n_sections):
        start_h = 24 * i / n_sections
        end_h   = 24 * (i + 1) / n_sections
        c0 = hour_to_col(start_h, W)
        c1 = hour_to_col(end_h,   W)
        vals = intensity_per_col[c0:c1]
        print(f"{int(start_h):02d}–{int(end_h):02d}h (cols {c0}–{c1-1}): "
              f"mean={vals.mean():.2f}, median={np.median(vals):.2f}")

# ---------- Save each section as PNG ----------
def save_sections(slices, out_dir, base_name="slice"):
    os.makedirs(out_dir, exist_ok=True)
    out_paths = []
    for i, section in enumerate(slices, 1):
        p = os.path.join(out_dir, f"{base_name}_{i:02d}.png")
        Image.fromarray(section).save(p)
        out_paths.append(p)
    return out_paths

# ---------- Example usage ----------
if __name__ == "__main__":
    # Set your path (relative to the script). Example: inside a "2015" folder.
    image_path = "2015/20150101__pfrr_asi3_full-keo-rgb.png"

    rgb, H, W = convert_image_to_RGBarray(image_path)
    print(f"Loaded image: {image_path}  -> shape={rgb.shape} (H={H}, W={W})")

    slices, col_ranges, hour_ranges = split_into_sections(rgb, n_sections=8)
    print("Column ranges per section:", col_ranges)

    # Draw the 8 sections
    plot_sections(slices, hour_ranges, nrows=2, ncols=4, figsize=(16, 6))
    plot_gray_sections(slices, hour_ranges, nrows=2, ncols=4, figsize=(16, 6))
    # (Optional) Print intensity statistics per 3-hour block
    print_intensity_stats_by_block(rgb, n_sections=8)

    # (Optional) Save each section as a PNG
    out_paths = save_sections(slices, out_dir="out_slices_20150105",
                              base_name="20150105_slice")
    print("Saved slices:", out_paths)
