"""
02_visualize.py
---------------
Visualise an AEMET radar GIF as a publication-quality reflectivity plot.

Produces (all saved to data/plots/):
    • aemet_ppi.png          — circular radar disc (true PPI appearance)
    • aemet_raw_array.png    — flat 2-D array view (rays × gates / H × W)

Usage:
    python 02_visualize.py -f aemet_nacional_20260515_1115.gif
"""

from __future__ import annotations

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")                      # non-interactive — always saves to file
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── Project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.radar_io import read_radar, get_field_array
from utils.enhancement import generate_synthetic_radar_field

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Visualise AEMET radar data")
parser.add_argument(
    "-f", type=str, default=None,
    help="GIF filename inside data/  (e.g. aemet_nacional_20260515_1115.gif)"
)
args = parser.parse_args()

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
PLOTS_DIR = os.path.join(DATA_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Colour map (NWS-style reflectivity) ──────────────────────────────────────
_REF_COLORS = [
    "#646464", "#04e9e7", "#019ff4", "#0300f4",
    "#02fd02", "#01c501", "#008e00", "#fdf802",
    "#e5bc00", "#fd9500", "#fd0000", "#d40000",
    "#bc0000", "#f800fd", "#9854c6",
]
_REF_NODES  = np.linspace(0, 1, len(_REF_COLORS))
NWS_REF_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "nws_ref", list(zip(_REF_NODES, _REF_COLORS))
)
VMIN, VMAX = -10.0, 75.0


# ── Plotting ──────────────────────────────────────────────────────────────────

def _dark_fig(figsize=(9, 8)):
    fig, ax = plt.subplots(figsize=figsize, facecolor="#0d0d0d")
    ax.set_facecolor("#0d0d0d")
    return fig, ax


def _style_ax(ax):
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")


def _colorbar(fig, im, ax, label="Reflectivity (dBZ)"):
    """Attach a colourbar to the right of ax."""
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.92)
    cbar.set_label(label, color="white", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=8)
    return cbar


def _save(fig, path: str):
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  Saved → {path}")
    plt.close(fig)


def plot_ppi(data: np.ndarray, title: str, save_path: str):
    """
    Draw the data as a circular PPI disc — masks everything outside the
    radar circle so it looks like a real radar product.
    """
    H, W = data.shape
    cy, cx = H / 2, W / 2
    radius = min(cy, cx) * 0.90

    # Build a masked array: NaN outside the disc
    y, x    = np.ogrid[:H, :W]
    outside = (y - cy) ** 2 + (x - cx) ** 2 > radius ** 2
    display = data.copy()
    display[outside] = np.nan

    fig, ax = _dark_fig(figsize=(9, 9))
    im = ax.imshow(
        display,
        origin="upper",
        vmin=VMIN, vmax=VMAX,
        cmap=NWS_REF_CMAP,
        interpolation="bilinear",
    )
    ax.set_title(title, color="white", fontsize=13, pad=10)
    ax.axis("off")                     # clean — no pixel coordinates shown
    _colorbar(fig, im, ax)
    _save(fig, save_path)


def plot_raw_array(data: np.ndarray, title: str, save_path: str):
    """Rectangular 2-D view of the raw dBZ array."""
    fig, ax = _dark_fig(figsize=(12, 5))
    im = ax.imshow(
        data,
        aspect="auto", origin="upper",
        vmin=VMIN, vmax=VMAX,
        cmap=NWS_REF_CMAP,
        interpolation="nearest",
    )
    ax.set_title(title, color="white", fontsize=12)
    ax.set_xlabel("Column (x / gate)", color="white")
    ax.set_ylabel("Row (y / ray)", color="white")
    _style_ax(ax)
    _colorbar(fig, im, ax)
    _save(fig, save_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    data = None
    label = "Synthetic"

    # 1. Try to load the requested file
    if args.f:
        filepath = os.path.join(DATA_DIR, args.f)
        try:
            radar = read_radar(filepath)
            data  = get_field_array(radar, "reflectivity")
            label = os.path.splitext(args.f)[0]
            print(f"[OK] Loaded: {args.f}  shape={data.shape}")
        except Exception as e:
            print(f"[!] Could not load '{args.f}': {e}")
            print("    Falling back to synthetic data.")
    else:
        # Auto-discover newest AEMET gif in data/
        gifs = sorted(
            [f for f in os.listdir(DATA_DIR) if f.startswith("aemet_") and f.endswith(".gif")],
            reverse=True,
        )
        if gifs:
            filepath = os.path.join(DATA_DIR, gifs[0])
            try:
                radar = read_radar(filepath)
                data  = get_field_array(radar, "reflectivity")
                label = os.path.splitext(gifs[0])[0]
                print(f"[OK] Auto-loaded newest file: {gifs[0]}  shape={data.shape}")
            except Exception as e:
                print(f"[!] Auto-load failed: {e}")

    # 2. Synthetic fallback
    if data is None:
        print("[OK] Using synthetic radar field.")
        data  = generate_synthetic_radar_field(h=760, w=760)
        label = "Synthetic"

    print(f"Value range: {np.nanmin(data):.1f} … {np.nanmax(data):.1f} dBZ")
    print(f"Signal pixels (>0 dBZ): {np.sum(data > 0):,}")

    # 3. Produce plots
    print("\nGenerating plots…")

    plot_ppi(
        data,
        title=f"AEMET Reflectivity — {label}",
        save_path=os.path.join(PLOTS_DIR, "aemet_ppi.png"),
    )

    plot_raw_array(
        data,
        title=f"Raw dBZ Array — {label}  (shape {data.shape[0]}×{data.shape[1]})",
        save_path=os.path.join(PLOTS_DIR, "aemet_raw_array.png"),
    )

    print(f"\n[OK] Plots saved to: {PLOTS_DIR}")
    print("Next step: python 03_enhance.py")


if __name__ == "__main__":
    main()
