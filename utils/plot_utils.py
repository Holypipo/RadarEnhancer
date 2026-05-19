"""
utils/plot_utils.py
-------------------
Shared plotting helpers used across all pipeline scripts:
  - Colourmap definition (set once, used everywhere)
  - Timestamped filename builder
  - Dark-themed figure / axes factory
  - Colourbar helper
  - Save helper (prints file size)
"""

from __future__ import annotations
import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


# ── Colourmap ─────────────────────────────────────────────────────────────────
# Anchored to AEMET dBZ palette levels (12, 18, 24 … 72 dBZ).
# NaN (background / no-data) always renders black via set_bad.

_AEMET_COLORS = [
    "#000080",   # 0  dBZ — dark navy (near-zero, rarely seen)
    "#0000FC",   # 12 dBZ — blue
    "#0094FC",   # 18 dBZ — sky blue
    "#00FCFC",   # 24 dBZ — cyan
    "#00C000",   # 30 dBZ — dark green
    "#00FC00",   # 36 dBZ — bright green
    "#FCFC00",   # 42 dBZ — yellow
    "#FCAC00",   # 48 dBZ — orange
    "#FC7400",   # 54 dBZ — deep orange
    "#FC0000",   # 60 dBZ — red
    "#BC0000",   # 66 dBZ — dark red
    "#FC00FC",   # 72 dBZ — magenta
]

RADAR_CMAP: mcolors.LinearSegmentedColormap = (
    mcolors.LinearSegmentedColormap.from_list(
        "aemet_dbz",
        list(zip(np.linspace(0, 1, len(_AEMET_COLORS)), _AEMET_COLORS)),
    )
)
RADAR_CMAP.set_bad(color="black")   # NaN → black, no blue tint

DBZ_VMIN = 0.0
DBZ_VMAX = 72.0


# ── File naming ───────────────────────────────────────────────────────────────

def make_filename(step: str, name: str, ext: str = "png") -> str:
    """
    Build a timestamped CamelCase filename.

    Parameters
    ----------
    step : step prefix, e.g. '02'
    name : CamelCase plot name, e.g. 'RawArray'
    ext  : file extension without dot

    Returns
    -------
    e.g. '02_RawArray_19052026_1043.png'
    """
    ts = datetime.now().strftime("%d%m%Y_%H%M")
    return f"{step}_{name}_{ts}.{ext}"


def make_save_path(out_dir: str, step: str, name: str, ext: str = "png") -> str:
    """Full path for make_filename rooted at out_dir."""
    return os.path.join(out_dir, make_filename(step, name, ext))


# ── Figure / axes factory ─────────────────────────────────────────────────────

def dark_fig(figsize: tuple = (9, 8)):
    """Single-axes dark figure."""
    fig, ax = plt.subplots(figsize=figsize, facecolor="#0d0d0d")
    ax.set_facecolor("#0d0d0d")
    return fig, ax


def dark_fig_axes(n: int, figsize: tuple):
    """Row of n dark axes."""
    fig = plt.figure(figsize=figsize, facecolor="#0d0d0d")
    axes = [fig.add_subplot(1, n, i + 1) for i in range(n)]
    for ax in axes:
        ax.set_facecolor("#0d0d0d")
    return fig, axes


def style_ax(ax):
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")


# ── Colourbar ─────────────────────────────────────────────────────────────────

def add_colorbar(fig, im, ax, label: str = "Reflectivity (dBZ)", fontsize: int = 9):
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, shrink=0.92)
    cbar.set_label(label, color="white", fontsize=fontsize)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=fontsize - 1)
    return cbar


def add_dbz_colorbar(fig, im, ax):
    """Colourbar with dBZ level tick marks at each palette level."""
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, shrink=0.80)
    cbar.set_label("Reflectivity (dBZ)", color="white", fontsize=10)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white", fontsize=9)
    for lvl in [12, 18, 24, 30, 36, 42, 48, 54, 60, 66, 72]:
        pos = (lvl - DBZ_VMIN) / (DBZ_VMAX - DBZ_VMIN)
        if 0 <= pos <= 1:
            cbar.ax.axhline(pos, color="white", linewidth=0.4, alpha=0.5)
    return cbar


# ── Save ──────────────────────────────────────────────────────────────────────

def save_fig(fig, path: str):
    """Save figure, print path and file size, close."""
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    kb = os.path.getsize(path) // 1024
    print(f"  Saved -> {os.path.basename(path)}  ({kb} KB)")


# ── Shared imshow ─────────────────────────────────────────────────────────────

def dbz_imshow(ax, data: np.ndarray, title: str, interpolation: str = "nearest"):
    """Standard dBZ imshow on a dark axis."""
    im = ax.imshow(
        data, aspect="auto", origin="upper",
        vmin=DBZ_VMIN, vmax=DBZ_VMAX,
        cmap=RADAR_CMAP,
        interpolation=interpolation,
    )
    ax.set_title(title, color="white", fontsize=10, pad=6)
    ax.tick_params(colors="white", labelsize=7)
    ax.set_xlabel("Column", color="white", fontsize=8)
    ax.set_ylabel("Row", color="white", fontsize=8)
    return im
