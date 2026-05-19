"""
03_enhance.py
-------------
Enhancement pipeline for AEMET radar data.

Pipeline (via enhance_field):
    Raw -> Palette Smoother -> Upscale 2x (ResNet if trained, else bicubic) -> Sharpen

Outputs (saved to data/03_enhance/):
    03_PipelineSteps_DDMMYYYY_HHMM.png   -- 4-panel step-by-step view
    03_BeforeAfter_DDMMYYYY_HHMM.png     -- original AEMET image vs enhanced PPI
    03_DifferenceMap_DDMMYYYY_HHMM.png   -- pipeline vs naive upscale
    03_EnhancedPpi_DDMMYYYY_HHMM.png     -- final publication-quality output

Run 04_superresolution.py first to train the ResNet model.
Without it the pipeline falls back to bicubic upscaling.

Usage:
    python 03_enhance.py
    python 03_enhance.py -f 01_AemetNacional_19052026_1043.gif
"""

from __future__ import annotations

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, ENHANCE_DIR, FETCH_DIR, ensure_core_dirs, get_resnet_model_path
from utils.logging_utils import configure_logging
from utils.radar_io import read_radar, get_field_array
from utils.image_radar import AemetRadar
from utils.enhancement import (
    denoise_advanced, upscale, enhance_field,
    compute_snr, generate_synthetic_radar_field,
)
from utils.plot_utils import (
    RADAR_CMAP, DBZ_VMIN, DBZ_VMAX,
    dark_fig, dark_fig_axes, add_colorbar, add_dbz_colorbar,
    save_fig, make_save_path, dbz_imshow,
)

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Enhance AEMET radar data")
parser.add_argument("-f", type=str, default=None,
    help="GIF filename inside data/01_fetch/ (e.g. 01_AemetNacional_19052026_1043.gif)")
args = parser.parse_args()

# ── Paths ─────────────────────────────────────────────────────────────────────
SCALE      = 2
ensure_core_dirs()
OUT_DIR    = ENHANCE_DIR
MODEL_PATH = get_resnet_model_path()


# ── Plot functions ────────────────────────────────────────────────────────────

def PlotPipelineSteps(raw, denoised, upscaled, enhanced, label, save_path):
    """4-panel: raw -> denoised -> upscaled -> enhanced."""
    panels = [
        (raw,      f"(1) Raw\n{raw.shape[0]}x{raw.shape[1]}"),
        (denoised, f"(2) Smoothed\n{denoised.shape[0]}x{denoised.shape[1]}"),
        (upscaled, f"(3) Upscaled {SCALE}x\n{upscaled.shape[0]}x{upscaled.shape[1]}"),
        (enhanced, f"(4) Enhanced\n{enhanced.shape[0]}x{enhanced.shape[1]}"),
    ]
    fig, axes = dark_fig_axes(4, (20, 5))
    im = None
    for ax, (d, t) in zip(axes, panels):
        im = dbz_imshow(ax, d, t)
    if im:
        add_colorbar(fig, im, axes[-1])
    fig.suptitle(f"AEMET Enhancement Pipeline -- {label}",
                 color="white", fontsize=12, y=1.01)
    fig.tight_layout()
    save_fig(fig, save_path)


def PlotBeforeAfter(original_gif_path: str | None, raw_dbz: np.ndarray,
                    enhanced: np.ndarray, label: str, save_path: str):
    """
    2-panel: original AEMET GIF (as-is) on the left,
             enhanced dBZ PPI on the right.
    If no GIF path is available (synthetic fallback), shows raw dBZ on left.
    """
    fig = plt.figure(figsize=(16, 7), facecolor="#0d0d0d")

    # ── Left: original AEMET image ───────────────────────────────────────────
    ax_before = fig.add_subplot(1, 2, 1)
    ax_before.set_facecolor("#0d0d0d")

    if original_gif_path and os.path.exists(original_gif_path):
        orig_rgb = np.array(Image.open(original_gif_path).convert("RGB"))
        ax_before.imshow(orig_rgb, origin="upper", interpolation="nearest")
        ax_before.set_title("Before -- Original AEMET Image",
                            color="white", fontsize=11, pad=6)
    else:
        # Synthetic fallback: show raw dBZ
        ax_before.imshow(raw_dbz, aspect="auto", origin="upper",
                         vmin=DBZ_VMIN, vmax=DBZ_VMAX,
                         cmap=RADAR_CMAP, interpolation="nearest")
        ax_before.set_title("Before -- Raw dBZ (no source image)",
                            color="white", fontsize=11, pad=6)
    ax_before.axis("off")

    # ── Right: enhanced PPI ──────────────────────────────────────────────────
    ax_after = fig.add_subplot(1, 2, 2)
    ax_after.set_facecolor("#0d0d0d")

    H, W   = enhanced.shape
    cy, cx = H / 2, W / 2
    radius = min(cy, cx) * 0.90
    display = enhanced.copy()
    y_idx, x_idx = np.ogrid[:H, :W]
    display[(y_idx - cy) ** 2 + (x_idx - cx) ** 2 > radius ** 2] = np.nan

    im = ax_after.imshow(display, origin="upper",
                         vmin=DBZ_VMIN, vmax=DBZ_VMAX,
                         cmap=RADAR_CMAP, interpolation="bilinear")
    ax_after.set_title(f"After -- Enhanced ({SCALE}x SR)",
                       color="white", fontsize=11, pad=6)
    ax_after.axis("off")
    add_colorbar(fig, im, ax_after)

    fig.suptitle(f"Before / After -- {label}", color="white", fontsize=13, y=1.01)
    fig.tight_layout()
    save_fig(fig, save_path)


def PlotDifferenceMap(naive_up: np.ndarray, enhanced: np.ndarray, save_path: str):
    """Signed difference: full pipeline vs naive bicubic upscale."""
    diff   = enhanced - naive_up
    finite = diff[np.isfinite(diff)]
    vabs   = float(np.nanpercentile(np.abs(finite), 98)) if len(finite) > 0 else 1.0

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
    ax.set_ylabel("Row",    color="white")
    fig.tight_layout()
    save_fig(fig, save_path)


def PlotEnhancedPpi(enhanced: np.ndarray, label: str,
                    used_resnet: bool, save_path: str):
    """Publication-quality circular PPI of the enhanced output."""
    H, W   = enhanced.shape
    cy, cx = H / 2, W / 2
    radius = min(cy, cx) * 0.90

    display = enhanced.copy()
    y_idx, x_idx = np.ogrid[:H, :W]
    display[(y_idx - cy) ** 2 + (x_idx - cx) ** 2 > radius ** 2] = np.nan

    fig, ax = plt.subplots(figsize=(10, 10), facecolor="#000000")
    ax.set_facecolor("#000000")

    im = ax.imshow(display, origin="upper",
                   vmin=DBZ_VMIN, vmax=DBZ_VMAX,
                   cmap=RADAR_CMAP, interpolation="bilinear")

    sr_tag = "ResNet SR" if used_resnet else "Bicubic"
    ax.set_title(
        f"AEMET Reflectivity -- Enhanced ({sr_tag} {SCALE}x)\n{label}",
        color="white", fontsize=13, pad=12,
    )
    ax.axis("off")
    add_dbz_colorbar(fig, im, ax)
    save_fig(fig, save_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    configure_logging("03_enhance")

    raw              = None
    label            = "Synthetic"
    original_gif_path = None

    # Load data
    if args.f:
        filepath = os.path.join(FETCH_DIR, args.f)
        if not os.path.exists(filepath):
            filepath = os.path.join(DATA_DIR, args.f)
        try:
            radar = read_radar(filepath)
            raw   = get_field_array(radar, "reflectivity")
            label = os.path.splitext(args.f)[0]
            if isinstance(radar, AemetRadar):
                original_gif_path = filepath
            print(f"[OK] Loaded: {args.f}  shape={raw.shape}")
        except Exception as e:
            print(f"[!] Could not load '{args.f}': {e}")
    else:
        gifs = sorted(
            [f for f in os.listdir(FETCH_DIR) if f.endswith(".gif")],
            reverse=True,
        ) if os.path.isdir(FETCH_DIR) else []
        if gifs:
            fp = os.path.join(FETCH_DIR, gifs[0])
            try:
                radar = read_radar(fp)
                raw   = get_field_array(radar, "reflectivity")
                label = os.path.splitext(gifs[0])[0]
                if isinstance(radar, AemetRadar):
                    original_gif_path = fp
                print(f"[OK] Auto-loaded: {gifs[0]}  shape={raw.shape}")
            except Exception as e:
                print(f"[!] Auto-load failed: {e}")

    if raw is None:
        print("[OK] Using synthetic radar field.")
        raw   = generate_synthetic_radar_field(h=760, w=760)
        label = "Synthetic"

    resnet_available = os.path.exists(MODEL_PATH)
    if resnet_available:
        print(f"[OK] ResNet model found -- neural super-resolution active")
    else:
        print(f"[!] No ResNet model -- using bicubic upscaling")
        print(f"    Tip: run 04_superresolution.py first")

    print(f"\n--- AEMET Radar Enhancement ---")
    print(f"Raw shape      : {raw.shape}")
    print(f"Signal pixels  : {int(np.sum(np.isfinite(raw) & (raw > 0))):,}")
    print(f"Value range    : {np.nanmin(raw):.1f} ... {np.nanmax(raw):.1f} dBZ")
    print(f"SNR (raw)      : {compute_snr(raw):.1f} dB")
    print(f"ResNet SR      : {resnet_available}\n")

    print("Step 1 -- Palette smoother...")
    denoised = denoise_advanced(raw)

    print("Step 2 -- Bicubic upscale (for step visualisation)...")
    upscaled_viz = upscale(denoised, scale=SCALE)

    print("Step 3 -- Full enhance_field() (ResNet SR if available)...")
    enhanced = enhance_field(raw, scale=SCALE, use_resnet=resnet_available)

    naive_up = upscale(raw, scale=SCALE, method="linear")

    print(f"\nEnhanced shape : {enhanced.shape}")
    print(f"SNR (enhanced) : {compute_snr(enhanced):.1f} dB\n")

    print("Generating plots...")

    PlotPipelineSteps(raw, denoised, upscaled_viz, enhanced, label,
        save_path=make_save_path(OUT_DIR, "03", "PipelineSteps"))

    PlotBeforeAfter(original_gif_path, raw, enhanced, label,
        save_path=make_save_path(OUT_DIR, "03", "BeforeAfter"))

    PlotDifferenceMap(naive_up, enhanced,
        save_path=make_save_path(OUT_DIR, "03", "DifferenceMap"))

    PlotEnhancedPpi(enhanced, label, used_resnet=resnet_available,
        save_path=make_save_path(OUT_DIR, "03", "EnhancedPpi"))

    print(f"\n[OK] All plots saved to: {OUT_DIR}")
    if not resnet_available:
        print("\n  -> Run '04_superresolution.py' to train the ResNet,")
        print("     then re-run this script for neural SR output.")


if __name__ == "__main__":
    main()
