"""
03_enhance.py
-------------
Apply the full enhancement pipeline to an AEMET radar reflectivity field
and save comparison plots to data/plots/.

Pipeline:
    Raw  →  Denoise  →  Upscale 2×  →  Sharpen  →  Enhanced

Outputs (all in data/plots/):
    enhance_pipeline_steps.png   — 4-panel step-by-step view
    enhance_before_after.png     — side-by-side comparison
    enhance_difference_map.png   — what the algorithm added over naive upscale

Usage:
    python 03_enhance.py -f aemet_nacional_20260515_1115.gif
"""

from __future__ import annotations

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")                  # always write to file, never open a window
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── Project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.radar_io import read_radar, get_field_array
from utils.enhancement import (
    denoise_gaussian, denoise_advanced,
    upscale, sharpen_unsharp_mask,
    compute_snr, generate_synthetic_radar_field,
)

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Enhance AEMET radar data")
parser.add_argument(
    "-f", type=str, default=None,
    help="GIF filename inside data/  (e.g. aemet_nacional_20260515_1115.gif)"
)
args = parser.parse_args()

# ── Config ────────────────────────────────────────────────────────────────────
SCALE     = 2
DATA_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
PLOTS_DIR = os.path.join(DATA_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

VMIN, VMAX = -10.0, 75.0

# NWS-style colourmap
_REF_COLORS = [
    "#646464", "#04e9e7", "#019ff4", "#0300f4",
    "#02fd02", "#01c501", "#008e00", "#fdf802",
    "#e5bc00", "#fd9500", "#fd0000", "#d40000",
    "#bc0000", "#f800fd", "#9854c6",
]
NWS_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "nws_ref", list(zip(np.linspace(0, 1, len(_REF_COLORS)), _REF_COLORS))
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _dark_fig_axes(n_panels: int, figsize: tuple):
    fig = plt.figure(figsize=figsize, facecolor="#0d0d0d")
    axes = []
    for i in range(n_panels):
        ax = fig.add_subplot(1, n_panels, i + 1)
        ax.set_facecolor("#0d0d0d")
        axes.append(ax)
    return fig, axes


def _imshow(ax, data, title, vmin=VMIN, vmax=VMAX):
    im = ax.imshow(
        data, aspect="auto", origin="upper",
        vmin=vmin, vmax=vmax, cmap=NWS_CMAP, interpolation="nearest",
    )
    ax.set_title(title, color="white", fontsize=10, pad=6)
    ax.tick_params(colors="white", labelsize=7)
    ax.set_xlabel("Column", color="white", fontsize=8)
    ax.set_ylabel("Row", color="white", fontsize=8)
    return im


def _shared_colorbar(fig, im, axes):
    """Add a single colourbar to the right of the last axis."""
    cbar = fig.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04, shrink=0.92)
    cbar.set_label("Reflectivity (dBZ)", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=8)


def _save(fig, path: str):
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    size_kb = os.path.getsize(path) // 1024
    print(f"  Saved → {path}  ({size_kb} KB)")


# ── Plot functions ────────────────────────────────────────────────────────────

def plot_pipeline_steps(raw, denoised, upscaled, enhanced, save_path: str):
    """4-panel figure: raw → denoised → upscaled → sharpened."""
    panels = [
        (raw,      f"① Raw\n{raw.shape[0]}×{raw.shape[1]}"),
        (denoised, f"② Denoised\n{denoised.shape[0]}×{denoised.shape[1]}"),
        (upscaled, f"③ Upscaled {SCALE}×\n{upscaled.shape[0]}×{upscaled.shape[1]}"),
        (enhanced, f"④ Sharpened\n{enhanced.shape[0]}×{enhanced.shape[1]}"),
    ]
    fig, axes = _dark_fig_axes(4, figsize=(18, 5))
    im = None
    for ax, (data, title) in zip(axes, panels):
        im = _imshow(ax, data, title)
    if im is not None:
        _shared_colorbar(fig, im, axes)
    fig.suptitle(
        "AEMET Radar Enhancement Pipeline",
        color="white", fontsize=13, y=1.01,
    )
    fig.tight_layout()
    _save(fig, save_path)


def plot_before_after(raw, enhanced, save_path: str):
    """2-panel before / after comparison."""
    fig, axes = _dark_fig_axes(2, figsize=(14, 6))
    _imshow(axes[0], raw,      f"Before — raw {raw.shape[0]}×{raw.shape[1]}")
    im = _imshow(axes[1], enhanced, f"After — enhanced {enhanced.shape[0]}×{enhanced.shape[1]}")
    _shared_colorbar(fig, im, axes)
    fig.suptitle("Before / After — AEMET Signal Enhancement",
                 color="white", fontsize=13, y=1.01)
    fig.tight_layout()
    _save(fig, save_path)


def plot_difference_map(naive_up, enhanced, save_path: str):
    """Signed difference: what the full pipeline adds over a naive upscale."""
    diff = enhanced - naive_up
    vabs = float(np.nanpercentile(np.abs(diff), 98))
    if vabs == 0:
        vabs = 1.0

    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0d0d0d")
    ax.set_facecolor("#0d0d0d")
    im = ax.imshow(
        diff, aspect="auto", origin="upper",
        vmin=-vabs, vmax=vabs, cmap="RdBu_r", interpolation="nearest",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.92)
    cbar.set_label("ΔdBZ  (enhanced − naive upscale)", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=8)

    ax.set_title("Difference Map — pipeline contribution over naive upscale",
                 color="white", fontsize=11)
    ax.tick_params(colors="white")
    ax.set_xlabel("Column", color="white")
    ax.set_ylabel("Row", color="white")
    fig.tight_layout()
    _save(fig, save_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    raw = None

    # 1. Load data
    if args.f:
        filepath = os.path.join(DATA_DIR, args.f)
        try:
            radar = read_radar(filepath)
            raw   = get_field_array(radar, "reflectivity")
            print(f"[OK] Loaded: {args.f}  shape={raw.shape}")
        except Exception as e:
            print(f"[!] Could not load '{args.f}': {e}")
    else:
        # Auto-discover newest AEMET gif
        gifs = sorted(
            [f for f in os.listdir(DATA_DIR) if f.startswith("aemet_") and f.endswith(".gif")],
            reverse=True,
        )
        if gifs:
            try:
                radar = read_radar(os.path.join(DATA_DIR, gifs[0]))
                raw   = get_field_array(radar, "reflectivity")
                print(f"[OK] Auto-loaded: {gifs[0]}  shape={raw.shape}")
            except Exception as e:
                print(f"[!] Auto-load failed: {e}")

    if raw is None:
        print("[OK] Using synthetic radar field.")
        raw = generate_synthetic_radar_field(h=760, w=760)

    print(f"\n--- AEMET Radar Enhancement ---")
    print(f"Raw shape  : {raw.shape}")
    print(f"Value range: {np.nanmin(raw):.1f} … {np.nanmax(raw):.1f} dBZ")
    print(f"SNR (raw)  : {compute_snr(raw):.1f} dB\n")

    # 2. Pipeline steps
    print("Step 1 — Denoising…")
    denoised = denoise_advanced(raw)

    print("Step 2 — Upscaling (2×)…")
    upscaled = upscale(denoised, scale=SCALE)

    print("Step 3 — Sharpening…")
    enhanced = sharpen_unsharp_mask(upscaled, sigma=1.5, amount=0.8)

    # Naive upscale for the difference map
    naive_up = upscale(raw, scale=SCALE, method="linear")

    print(f"\nEnhanced shape : {enhanced.shape}")
    print(f"SNR (enhanced) : {compute_snr(enhanced):.1f} dB\n")

    # 3. Plots
    print("Generating plots…")

    plot_pipeline_steps(
        raw, denoised, upscaled, enhanced,
        save_path=os.path.join(PLOTS_DIR, "enhance_pipeline_steps.png"),
    )

    plot_before_after(
        raw, enhanced,
        save_path=os.path.join(PLOTS_DIR, "enhance_before_after.png"),
    )

    plot_difference_map(
        naive_up, enhanced,
        save_path=os.path.join(PLOTS_DIR, "enhance_difference_map.png"),
    )

    print(f"\n[OK] All enhancement plots saved to: {PLOTS_DIR}")
    print("Next step: python 04_superresolution.py")


if __name__ == "__main__":
    main()
