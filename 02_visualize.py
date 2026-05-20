"""
02_visualize.py
---------------
Visualise an AEMET radar GIF as publication-quality plots.

Outputs (saved to data/02_visualize/):
    02_Ppi_DDMMYYYY_HHMM.png       -- circular radar disc
    02_RawArray_DDMMYYYY_HHMM.png  -- flat 2-D dBZ array

Usage:
    python 02_visualize.py                                      # auto-loads newest from data/01_fetch/
    python 02_visualize.py -f 01_AemetNacional_19052026_1043.gif
"""

from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import DATA_DIR, FETCH_DIR, VISUALIZE_DIR, ensure_core_dirs
from utils.logging_utils import configure_logging
from utils.metrics import (
    compare_metrics,
    compute_dbz_metrics,
    format_dbz_metrics_block,
    read_metrics_json,
    write_metrics_json,
    write_summary_text,
)
from utils.radar_io import read_radar, get_field_array
from utils.enhancement import generate_synthetic_radar_field
from utils.plot_utils import (
    RADAR_CMAP, DBZ_VMIN, DBZ_VMAX,
    dark_fig, style_ax, add_colorbar, save_fig,
    make_save_path, dbz_imshow,
)

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Visualise AEMET radar data")
parser.add_argument("-f", type=str, default=None,
    help="GIF filename inside data/01_fetch/ (e.g. 01_AemetNacional_19052026_1043.gif)")
args = parser.parse_args()

# ── Paths ─────────────────────────────────────────────────────────────────────
ensure_core_dirs()
OUT_DIR = Path(os.getenv("RADAR_STEP_OUT_DIR", str(VISUALIZE_DIR)))
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Plot functions ────────────────────────────────────────────────────────────

def PlotPpi(data: np.ndarray, title: str, save_path: str):
    """Circular PPI disc — NaN background renders as black."""
    H, W   = data.shape
    cy, cx = H / 2, W / 2
    radius = min(cy, cx) * 0.90

    display = data.copy()
    y, x    = np.ogrid[:H, :W]
    display[(y - cy) ** 2 + (x - cx) ** 2 > radius ** 2] = np.nan

    fig, ax = dark_fig(figsize=(9, 9))
    im = ax.imshow(display, origin="upper",
                   vmin=DBZ_VMIN, vmax=DBZ_VMAX,
                   cmap=RADAR_CMAP, interpolation="bilinear")
    ax.set_title(title, color="white", fontsize=13, pad=10)
    ax.axis("off")
    add_colorbar(fig, im, ax)
    save_fig(fig, save_path)


def PlotRawArray(data: np.ndarray, title: str, save_path: str):
    """Rectangular 2-D view of the dBZ array."""
    fig, ax = dark_fig(figsize=(12, 5))
    im = dbz_imshow(ax, data, title)
    style_ax(ax)
    add_colorbar(fig, im, ax)
    save_fig(fig, save_path)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    configure_logging("02_visualize")

    data  = None
    label = "Synthetic"

    if args.f:
        filepath = os.path.join(FETCH_DIR, args.f)
        if not os.path.exists(filepath):
            # Try root data/ as fallback for manually placed files
            filepath = os.path.join(DATA_DIR, args.f)
        try:
            radar = read_radar(filepath)
            data  = get_field_array(radar, "reflectivity")
            label = os.path.splitext(args.f)[0]
            print(f"[OK] Loaded: {args.f}  shape={data.shape}")
        except Exception as e:
            print(f"[!] Could not load '{args.f}': {e}")
    else:
        gifs = sorted(
            [f for f in os.listdir(FETCH_DIR) if f.endswith(".gif")],
            reverse=True,
        )
        if gifs:
            try:
                radar = read_radar(os.path.join(FETCH_DIR, gifs[0]))
                data  = get_field_array(radar, "reflectivity")
                label = os.path.splitext(gifs[0])[0]
                print(f"[OK] Auto-loaded: {gifs[0]}  shape={data.shape}")
            except Exception as e:
                print(f"[!] Auto-load failed: {e}")

    if data is None:
        print("[OK] Using synthetic radar field.")
        data  = generate_synthetic_radar_field(h=760, w=760)
        label = "Synthetic"

    metrics = compute_dbz_metrics(data)
    metrics["source_file"] = label
    summary_parts = [format_dbz_metrics_block("\n--- AEMET Visualize Decode ---", metrics)]

    compare_path = os.getenv("RADAR_COMPARE_METRICS_JSON")
    if compare_path and os.path.exists(compare_path):
        fetch_metrics = read_metrics_json(compare_path)
        comparison = compare_metrics(fetch_metrics, metrics)
        comparison_lines = [
            "\n--- Compare To Fetch ---",
            f"Same shape     : {comparison['same_shape']}",
            f"Exact match    : {comparison['exact_match']}",
            f"Delta signal   : {comparison['delta_signal_pixels']:+,}",
            f"Delta finite   : {comparison['delta_finite_pixels']:+,}",
        ]
        if comparison["delta_class_counts_dbz"]:
            class_delta = ", ".join(
                f"{key}:{value:+,}" for key, value in comparison["delta_class_counts_dbz"].items()
            )
            comparison_lines.append(f"Delta classes  : {class_delta}")
        else:
            comparison_lines.append("Delta classes  : none")
        summary_parts.append("\n".join(comparison_lines))
        metrics["comparison_to_fetch"] = comparison

    summary = "\n".join(summary_parts)
    print(summary)

    print("\nGenerating plots...")

    PlotPpi(data,
        title=f"AEMET Reflectivity -- {label}",
        save_path=make_save_path(OUT_DIR, "02", "Ppi"))

    PlotRawArray(data,
        title=f"Raw dBZ Array -- {label}  ({data.shape[0]}x{data.shape[1]})",
        save_path=make_save_path(OUT_DIR, "02", "RawArray"))

    metrics_path = os.getenv("RADAR_STEP_METRICS_JSON")
    summary_path = os.getenv("RADAR_STEP_SUMMARY_TXT")
    if metrics_path:
        write_metrics_json(metrics_path, metrics)
    if summary_path:
        write_summary_text(summary_path, summary)

    print(f"\n[OK] Plots saved to: {OUT_DIR}")
    print("Next step: python 03_enhance.py")


if __name__ == "__main__":
    main()
