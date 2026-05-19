"""
03_enhance.py
-------------
Apply the full enhancement pipeline to an AEMET radar reflectivity field
and save all comparison and output plots to data/plots/.

Pipeline (via enhance_field):
    Raw  ->  Denoise  ->  Upscale 2x (ResNet if trained, else bicubic)  ->  Sharpen

Outputs (all in data/plots/):
    enhance_pipeline_steps.png   -- 4-panel step-by-step view
    enhance_before_after.png     -- side-by-side raw vs final
    enhance_difference_map.png   -- what the algorithm added over naive upscale
    enhanced_final_ppi.png       -- publication-quality circular output image

Usage:
    python 03_enhance.py
    python 03_enhance.py -f aemet_nacional_20260515_1146.gif

Run 04_superresolution.py first to train the ResNet model.
If no model exists, the pipeline uses high-quality bicubic upscaling instead.
"""

from __future__ import annotations

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.radar_io import read_radar, get_field_array
from utils.enhancement import (
    denoise_advanced, upscale, enhance_field,
    compute_snr, generate_synthetic_radar_field,
)

parser = argparse.ArgumentParser(description="Enhance AEMET radar data")
parser.add_argument("-f", type=str, default=None,
    help="GIF filename inside data/  (e.g. aemet_nacional_20260515_1146.gif)")
args = parser.parse_args()

SCALE      = 2
DATA_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
PLOTS_DIR  = os.path.join(DATA_DIR, "plots")
MODELS_DIR = os.path.join(DATA_DIR, "models")
os.makedirs(PLOTS_DIR, exist_ok=True)

VMIN, VMAX = 0.0, 72.0

_REF_COLORS = [
    "#000080", "#0000FC", "#0094FC", "#00FCFC",
    "#00C000", "#00FC00", "#FCFC00", "#FCAC00",
    "#FC7400", "#FC0000", "#BC0000", "#FC00FC",
]
NWS_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "nws_aemet", list(zip(np.linspace(0, 1, len(_REF_COLORS)), _REF_COLORS))
)


def _dark_fig_axes(n, figsize):
    fig = plt.figure(figsize=figsize, facecolor="#0d0d0d")
    axes = [fig.add_subplot(1, n, i + 1) for i in range(n)]
    for ax in axes:
        ax.set_facecolor("#0d0d0d")
    return fig, axes


def _imshow(ax, data, title):
    im = ax.imshow(data, aspect="auto", origin="upper",
                   vmin=VMIN, vmax=VMAX, cmap=NWS_CMAP, interpolation="nearest")
    ax.set_title(title, color="white", fontsize=10, pad=6)
    ax.tick_params(colors="white", labelsize=7)
    ax.set_xlabel("Column", color="white", fontsize=8)
    ax.set_ylabel("Row", color="white", fontsize=8)
    return im


def _colorbar(fig, im, ax):
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.92)
    cbar.set_label("Reflectivity (dBZ)", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=8)


def _save(fig, path):
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved -> {path}  ({os.path.getsize(path)//1024} KB)")


def plot_pipeline_steps(raw, denoised, upscaled, enhanced, label, save_path):
    panels = [
        (raw,      f"(1) Raw\n{raw.shape[0]}x{raw.shape[1]}"),
        (denoised, f"(2) Denoised\n{denoised.shape[0]}x{denoised.shape[1]}"),
        (upscaled, f"(3) Upscaled {SCALE}x\n{upscaled.shape[0]}x{upscaled.shape[1]}"),
        (enhanced, f"(4) Enhanced\n{enhanced.shape[0]}x{enhanced.shape[1]}"),
    ]
    fig, axes = _dark_fig_axes(4, (20, 5))
    im = None
    for ax, (d, t) in zip(axes, panels):
        im = _imshow(ax, d, t)
    if im:
        _colorbar(fig, im, axes[-1])
    fig.suptitle(f"AEMET Enhancement Pipeline -- {label}", color="white", fontsize=12, y=1.01)
    fig.tight_layout()
    _save(fig, save_path)


def plot_before_after(raw, enhanced, label, save_path):
    fig, axes = _dark_fig_axes(2, (14, 6))
    _imshow(axes[0], raw,      f"Before -- {raw.shape[0]}x{raw.shape[1]}")
    im = _imshow(axes[1], enhanced, f"After -- {enhanced.shape[0]}x{enhanced.shape[1]}")
    _colorbar(fig, im, axes[-1])
    fig.suptitle(f"Before / After -- {label}", color="white", fontsize=12, y=1.01)
    fig.tight_layout()
    _save(fig, save_path)


def plot_difference_map(naive_up, enhanced, save_path):
    diff = enhanced - naive_up
    finite_diff = diff[np.isfinite(diff)]
    vabs = float(np.nanpercentile(np.abs(finite_diff), 98)) if len(finite_diff) > 0 else 1.0

    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0d0d0d")
    ax.set_facecolor("#0d0d0d")
    im = ax.imshow(diff, aspect="auto", origin="upper",
                   vmin=-vabs, vmax=vabs, cmap="RdBu_r", interpolation="nearest")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.92)
    cbar.set_label("dBZ (enhanced - naive upscale)", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=8)
    ax.set_title("Difference Map -- pipeline contribution over naive upscale",
                 color="white", fontsize=11)
    ax.tick_params(colors="white")
    ax.set_xlabel("Column", color="white")
    ax.set_ylabel("Row", color="white")
    fig.tight_layout()
    _save(fig, save_path)


def plot_final_ppi(enhanced, label, used_resnet, save_path):
    """
    Publication-quality circular PPI -- the main output image.
    NaN pixels (background / no-data) render as pure black.
    """
    H, W = enhanced.shape
    cy, cx = H / 2, W / 2
    radius = min(cy, cx) * 0.90
    y_idx, x_idx = np.ogrid[:H, :W]
    display = enhanced.copy()
    display[(y_idx - cy) ** 2 + (x_idx - cx) ** 2 > radius ** 2] = np.nan

    fig, ax = plt.subplots(figsize=(10, 10), facecolor="#000000")
    ax.set_facecolor("#000000")

    cmap = NWS_CMAP.copy()
    cmap.set_bad(color="black")

    im = ax.imshow(display, origin="upper", vmin=VMIN, vmax=VMAX,
                   cmap=cmap, interpolation="bilinear")

    sr_label = "ResNet SR" if used_resnet else "Bicubic"
    ax.set_title(
        f"AEMET Reflectivity -- Enhanced ({sr_label} {SCALE}x)\n{label}",
        color="white", fontsize=13, pad=12,
    )
    ax.axis("off")

    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, shrink=0.80)
    cbar.set_label("Reflectivity (dBZ)", color="white", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=9)

    levels = [12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72]
    for lvl in levels:
        norm_pos = (lvl - VMIN) / (VMAX - VMIN)
        if 0 <= norm_pos <= 1:
            cbar.ax.axhline(norm_pos, color="white", linewidth=0.4, alpha=0.5)

    _save(fig, save_path)


def main():
    raw = None
    label = "Synthetic"

    if args.f:
        filepath = os.path.join(DATA_DIR, args.f)
        try:
            radar = read_radar(filepath)
            raw   = get_field_array(radar, "reflectivity")
            label = os.path.splitext(args.f)[0]
            print(f"[OK] Loaded: {args.f}  shape={raw.shape}")
        except Exception as e:
            print(f"[!] Could not load '{args.f}': {e}")
    else:
        gifs = sorted(
            [f for f in os.listdir(DATA_DIR) if f.startswith("aemet_") and f.endswith(".gif")],
            reverse=True,
        )
        if gifs:
            try:
                radar = read_radar(os.path.join(DATA_DIR, gifs[0]))
                raw   = get_field_array(radar, "reflectivity")
                label = os.path.splitext(gifs[0])[0]
                print(f"[OK] Auto-loaded: {gifs[0]}  shape={raw.shape}")
            except Exception as e:
                print(f"[!] Auto-load failed: {e}")

    if raw is None:
        print("[OK] Using synthetic radar field.")
        raw = generate_synthetic_radar_field(h=760, w=760)
        label = "Synthetic"

    model_path = os.path.join(MODELS_DIR, "resnet_radar.pth")
    resnet_available = os.path.exists(model_path)

    if resnet_available:
        print(f"[OK] ResNet model found -- will use neural super-resolution")
    else:
        print(f"[!] No ResNet model found -- using bicubic upscaling")
        print(f"    Tip: run 04_superresolution.py first for neural SR")

    signal_count = int(np.sum(np.isfinite(raw) & (raw > 0)))
    print(f"\n--- AEMET Radar Enhancement ---")
    print(f"Raw shape      : {raw.shape}")
    print(f"Signal pixels  : {signal_count:,}")
    print(f"Value range    : {np.nanmin(raw):.1f} ... {np.nanmax(raw):.1f} dBZ")
    print(f"SNR (raw)      : {compute_snr(raw):.1f} dB")
    print(f"ResNet SR      : {resnet_available}\n")

    print("Step 1 -- Denoising...")
    denoised = denoise_advanced(raw)

    print("Step 2 -- Bicubic upscale (for step visualisation)...")
    upscaled_viz = upscale(denoised, scale=SCALE)

    print("Step 3 -- Full enhance_field() (ResNet SR if model available)...")
    enhanced = enhance_field(raw, scale=SCALE, use_resnet=resnet_available)

    naive_up = upscale(raw, scale=SCALE, method="linear")

    print(f"\nEnhanced shape : {enhanced.shape}")
    print(f"SNR (enhanced) : {compute_snr(enhanced):.1f} dB\n")

    print("Generating plots...")

    plot_pipeline_steps(raw, denoised, upscaled_viz, enhanced, label,
        save_path=os.path.join(PLOTS_DIR, "enhance_pipeline_steps.png"))

    plot_before_after(raw, enhanced, label,
        save_path=os.path.join(PLOTS_DIR, "enhance_before_after.png"))

    plot_difference_map(naive_up, enhanced,
        save_path=os.path.join(PLOTS_DIR, "enhance_difference_map.png"))

    plot_final_ppi(enhanced, label, used_resnet=resnet_available,
        save_path=os.path.join(PLOTS_DIR, "enhanced_final_ppi.png"))

    print(f"\n[OK] All plots saved to: {PLOTS_DIR}")
    if not resnet_available:
        print("\n  -> Run '04_superresolution.py' to train the ResNet,")
        print("     then re-run this script to get neural SR output.")


if __name__ == "__main__":
    main()
