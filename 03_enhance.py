"""
03_enhance.py
-------------
Applies the full enhancement pipeline to the downloaded NEXRAD reflectivity
field and produces a side-by-side before/after comparison plot.

Pipeline:
    Raw data  →  Denoise (Gaussian)  →  Upscale (2×)  →  Sharpen (unsharp mask)

Usage:
    python 03_enhance.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

try:
    import pyart
    HAS_PYART = True
except ImportError:
    HAS_PYART = False

# Add project root to path so utils can be imported
sys.path.insert(0, os.path.dirname(__file__))
from utils.radar_io import read_radar, get_field_array
from utils.enhancement import denoise_gaussian, upscale, sharpen_unsharp_mask, enhance_field, compute_snr, generate_synthetic_radar_field

# ── Config ────────────────────────────────────────────────────────────────────

DATA_DIR  = os.path.join(os.path.dirname(__file__), "data")
PLOTS_DIR = os.path.join(DATA_DIR, "plots")
FILE_PATH = os.path.join(DATA_DIR, "KTLX20130520_195527_V06")
FIELD     = "reflectivity"
SWEEP     = 0          # lowest elevation tilt
SCALE     = 2          # upscaling factor (2 = double resolution)

os.makedirs(PLOTS_DIR, exist_ok=True)


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _imshow(ax, data, title, vmin=-32, vmax=75, cmap="pyart_NWSRef"):
    """Shared imshow call for all panels."""
    cmap_obj = "viridis"
    if HAS_PYART:
        try:
            cmap_obj = pyart.graph.cm.get_cmap(cmap)
        except Exception:
            pass

    im = ax.imshow(
        data, aspect="auto", origin="upper",
        vmin=vmin, vmax=vmax, cmap=cmap_obj, interpolation="nearest",
    )
    ax.set_title(title, color="white", fontsize=10, pad=6)
    ax.tick_params(colors="white", labelsize=7)
    ax.set_xlabel("Gate (range)", color="white", fontsize=8)
    ax.set_ylabel("Ray (azimuth)", color="white", fontsize=8)
    return im


def plot_pipeline_steps(raw, denoised, upscaled, enhanced, save_path=None):
    """
    4-panel figure showing each step of the enhancement pipeline.
    """
    fig = plt.figure(figsize=(18, 8), facecolor="#0d0d0d")
    gs  = gridspec.GridSpec(1, 4, figure=fig, wspace=0.04)

    panels = [
        (raw,      f"① Raw\n{raw.shape[0]}×{raw.shape[1]}"),
        (denoised, f"② Denoised (σ=1.0)\n{denoised.shape[0]}×{denoised.shape[1]}"),
        (upscaled, f"③ Upscaled {SCALE}×\n{upscaled.shape[0]}×{upscaled.shape[1]}"),
        (enhanced, f"④ Sharpened\n{enhanced.shape[0]}×{enhanced.shape[1]}"),
    ]

    ims = []
    for i, (data, title) in enumerate(panels):
        ax = fig.add_subplot(gs[i])
        ax.set_facecolor("#0d0d0d")
        im = _imshow(ax, data, title)
        ims.append(im)
        if i > 0:
            ax.set_ylabel("")  # clean up duplicate labels

    # Shared colourbar on the right
    cbar_ax = fig.add_axes([0.92, 0.12, 0.012, 0.76])
    cbar = fig.colorbar(ims[-1], cax=cbar_ax)
    cbar.set_label("Reflectivity (dBZ)", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=8)

    fig.suptitle(
        "XRAD Enhancement Pipeline — KTLX 2013-05-20  |  Sweep 0 (0.5°)",
        color="white", fontsize=13, y=0.98,
    )

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  Saved: {os.path.abspath(save_path)}")
        plt.close(fig)
    else:
        plt.show()


def plot_before_after(raw, enhanced, save_path=None):
    """
    Simple 2-panel before / after comparison (good for reports).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="#0d0d0d")
    fig.subplots_adjust(wspace=0.05)

    for ax in axes:
        ax.set_facecolor("#0d0d0d")

    _imshow(axes[0], raw,      f"Before  —  raw {raw.shape[0]}×{raw.shape[1]}")
    im = _imshow(axes[1], enhanced, f"After   —  enhanced {enhanced.shape[0]}×{enhanced.shape[1]}")

    cbar_ax = fig.add_axes([0.92, 0.12, 0.012, 0.76])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label("Reflectivity (dBZ)", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=8)

    fig.suptitle(
        "Before / After — XRAD Signal Enhancement",
        color="white", fontsize=13, y=0.98,
    )

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  Saved: {os.path.abspath(save_path)}")
        plt.close(fig)
    else:
        plt.show()


def plot_difference_map(raw_upscaled, enhanced, save_path=None):
    """
    Show the pixel-level difference between the naively upscaled version
    and the full enhancement — this highlights what the algorithm actually adds.
    """
    diff = enhanced - raw_upscaled
    vabs = np.nanpercentile(np.abs(diff), 98)

    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0d0d0d")
    ax.set_facecolor("#0d0d0d")

    im = ax.imshow(
        diff, aspect="auto", origin="upper",
        vmin=-vabs, vmax=vabs, cmap="RdBu_r", interpolation="nearest",
    )
    cbar = plt.colorbar(im, ax=ax, pad=0.01, fraction=0.02)
    cbar.set_label("ΔdBZ  (enhanced − naive upscale)", color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=8)

    ax.set_title("Difference Map — what the enhancement algorithm contributes",
                 color="white", fontsize=11)
    ax.tick_params(colors="white")
    ax.set_xlabel("Gate", color="white")
    ax.set_ylabel("Ray", color="white")

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  Saved: {os.path.abspath(save_path)}")
        plt.close(fig)
    else:
        plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    radar = None
    raw = None
    
    # 1. Try to load real data
    if os.path.exists(FILE_PATH):
        try:
            radar = read_radar(FILE_PATH)
            raw   = get_field_array(radar, FIELD, sweep=SWEEP)
            print(f"\n[OK] Loaded real radar data from {os.path.basename(FILE_PATH)}")
        except Exception as e:
            print(f"\n[!] Failed to read real radar file: {e}")
            print("    Falling back to synthetic data.")
    else:
        print(f"\n[!] Radar file not found at {FILE_PATH}")
        print("    Falling back to synthetic data.")

    # 2. Fallback to synthetic if real load failed
    if raw is None:
        print("\n--- Radar Signal Enhancement Demo (Synthetic) ---")
        raw = generate_synthetic_radar_field()
        print("[OK] Synthetic radar field generated.")
    else:
        print("\n--- Radar Signal Enhancement (Real Data) ---")

    print(f"\nRaw field shape : {raw.shape}  (rays x gates)")
    print(f"Value range     : {np.nanmin(raw):.1f} ... {np.nanmax(raw):.1f} dBZ")
    print(f"SNR estimate    : {compute_snr(raw):.1f} dB\n")

    # 2. Run each pipeline step individually (for the 4-panel plot)
    print("Step 1 — Denoising…")
    denoised = denoise_gaussian(raw, sigma=1.0)

    print("Step 2 — Upscaling (2×)…")
    upscaled = upscale(denoised, scale=SCALE)

    print("Step 3 — Sharpening…")
    enhanced = sharpen_unsharp_mask(upscaled, sigma=1.5, amount=1.2)

    # Naive upscale (no denoising / sharpening) — for the difference map
    naive_up = upscale(raw, scale=SCALE, method="linear")

    print(f"\nEnhanced shape  : {enhanced.shape}")
    print(f"SNR after enh.  : {compute_snr(enhanced):.1f} dB\n")

    # 3. Generate plots
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
