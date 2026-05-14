"""
02_visualize.py
---------------
Read the downloaded NEXRAD file and produce publication-quality plots of:
    • Reflectivity (dBZ)
    • Radial velocity (m/s)
    • Spectrum width

All plots are saved as PNGs in  data/plots/.

Usage:
    python 02_visualize.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

try:
    import pyart
    HAS_PYART = True
except ImportError:
    HAS_PYART = False

# Add project root to path so utils can be imported
sys.path.insert(0, os.path.dirname(__file__))
from utils.radar_io import read_radar, get_field_array
from utils.enhancement import generate_synthetic_radar_field

# ── Paths ─────────────────────────────────────────────────────────────────────

DATA_DIR  = os.path.join(os.path.dirname(__file__), "data")
PLOTS_DIR = os.path.join(DATA_DIR, "plots")
FILE_NAME = "KTLX20130520_195527_V06"
FILE_PATH = os.path.join(DATA_DIR, FILE_NAME)

os.makedirs(PLOTS_DIR, exist_ok=True)


# ── Colour maps ───────────────────────────────────────────────────────────────

# NWS-style reflectivity colourmap (approximation)
NWS_REF_COLORS = [
    "#646464", "#04e9e7", "#019ff4", "#0300f4",
    "#02fd02", "#01c501", "#008e00", "#fdf802",
    "#e5bc00", "#fd9500", "#fd0000", "#d40000",
    "#bc0000", "#f800fd", "#9854c6",
]
NWS_REF_BOUNDS = [-32, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 75]
nws_ref_cmap = mcolors.LinearSegmentedColormap.from_list(
    "nws_ref", list(zip(np.linspace(0, 1, len(NWS_REF_COLORS)), NWS_REF_COLORS))
)


# ── Plotting helpers ──────────────────────────────────────────────────────────

def plot_ppi(
    radar: pyart.core.Radar,
    field: str,
    sweep: int = 0,
    title: str | None = None,
    vmin: float = -32,
    vmax: float = 75,
    cmap: str = "pyart_NWSRef",
    save_path: str | None = None,
) -> None:
    """
    Plot a Plan Position Indicator (PPI) scan — the classic circular radar image.

    Parameters
    ----------
    radar     : Py-ART Radar object
    field     : field name to plot
    sweep     : sweep (elevation) index
    title     : plot title (auto-generated if None)
    vmin/vmax : colour scale range
    cmap      : matplotlib or PyART colourmap name
    save_path : if given, save PNG here; otherwise show interactively
    """
    fig, ax = plt.subplots(figsize=(9, 8), facecolor="#0d0d0d")
    ax.set_facecolor("#0d0d0d")

    if HAS_PYART:
        display = pyart.graph.RadarDisplay(radar)
        display.plot(
            field,
            sweep=sweep,
            ax=ax,
            title="",
            colorbar_label="",
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )
    else:
        # Fallback: plot the raw array as an approximate PPI if we can't use RadarDisplay
        # Note: This won't be a true circular PPI without coordinate transforms,
        # but it lets us see the data.
        data = get_field_array(radar, field, sweep=sweep)
        im = ax.imshow(data, aspect='auto', vmin=vmin, vmax=vmax, cmap='viridis')
        plt.colorbar(im, ax=ax)
        print(f"  [NOTE] Standard imshow used for {field} (pyart missing)")

    elev = 0.5 # Default fallback
    if hasattr(radar, "fixed_angle"):
        elev = radar.fixed_angle["data"][sweep]
    elif hasattr(radar, "sweeps") and len(radar.sweeps) > sweep:
        # MetPy sweep header has target_elevation
        try: elev = radar.sweeps[sweep][0].header.target_elevation
        except: pass

    auto_title = title or f"{field.replace('_', ' ').title()}  |  Elevation {elev:.1f}°"
    ax.set_title(auto_title, color="white", fontsize=13, pad=10)
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333333")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  Saved: {os.path.abspath(save_path)}")
        plt.close(fig)
    else:
        plt.show()


def plot_raw_array(
    data: np.ndarray,
    title: str = "Raw 2-D Field",
    vmin: float = -32,
    vmax: float = 75,
    cmap: str = "pyart_NWSRef",
    save_path: str | None = None,
) -> None:
    """
    Plot a raw 2-D NumPy array (rays × gates) as a rectangular image.
    Useful for seeing the raw data grid before polar coordinate transformations.
    """
    fig, ax = plt.subplots(figsize=(12, 5), facecolor="#0d0d0d")
    ax.set_facecolor("#0d0d0d")

    # Try to get the colormap from pyart, otherwise viridis
    cmap_obj = "viridis"
    if HAS_PYART:
        try:
            cmap_obj = pyart.graph.cm.get_cmap(cmap)
        except Exception:
            pass

    im = ax.imshow(
        data, aspect="auto", origin="upper",
        vmin=vmin, vmax=vmax, cmap=cmap_obj, interpolation="nearest"
    )
    cbar = plt.colorbar(im, ax=ax, pad=0.01, fraction=0.02)
    cbar.ax.yaxis.set_tick_params(color="white")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    ax.set_title(title, color="white", fontsize=12)
    ax.set_xlabel("Gate (range bin)", color="white")
    ax.set_ylabel("Ray (azimuth)", color="white")
    ax.tick_params(colors="white")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"  Saved: {os.path.abspath(save_path)}")
        plt.close(fig)
    else:
        plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    radar = None
    raw_np = None

    # 1. Try to load real data
    if os.path.exists(FILE_PATH):
        try:
            radar = read_radar(FILE_PATH)
            print(f"\n[OK] Loaded real radar data from {os.path.basename(FILE_PATH)}")
        except Exception as e:
            print(f"\n[!] Failed to read real radar file: {e}")
            print("    Falling back to synthetic data for visualization.")
    else:
        print(f"\n[!] Radar file not found at {FILE_PATH}")
        print("    Falling back to synthetic data for visualization.")

    # 2. Generate synthetic fallback if real load failed
    if radar is None:
        raw_np = generate_synthetic_radar_field()
        print("[OK] Synthetic radar field generated.")
        
        # Plot the synthetic array
        plot_raw_array(
            raw_np,
            title="Synthetic Reflectivity — Demonstration Field",
            vmin=-32, vmax=75,
            save_path=os.path.join(PLOTS_DIR, "synthetic_reflectivity_array.png"),
        )
        print(f"\n[OK] All demonstration plots saved to: {PLOTS_DIR}")
        return

    # 3. Proceed with real radar plots if loaded
    print("Generating plots…")

    # Reflectivity
    try:
        plot_ppi(
            radar, "reflectivity", sweep=0,
            vmin=-32, vmax=75, cmap="pyart_NWSRef",
            save_path=os.path.join(PLOTS_DIR, "ppi_reflectivity.png"),
        )
    except Exception as e:
        print(f"  Error plotting reflectivity: {e}")

    # Velocity
    try:
        plot_ppi(
            radar, "velocity", sweep=0,
            vmin=-32, vmax=32, cmap="pyart_NWSVel",
            save_path=os.path.join(PLOTS_DIR, "ppi_velocity.png"),
        )
    except Exception as e:
        print(f"  Error plotting velocity: {e}")

    # Spectrum width
    try:
        plot_ppi(
            radar, "spectrum_width", sweep=0,
            vmin=0, vmax=10, cmap="pyart_NWS_SPW",
            save_path=os.path.join(PLOTS_DIR, "ppi_spectrum_width.png"),
        )
    except Exception as e:
        # Only log if it's not a 'field not found' error
        if "not found" not in str(e).lower():
            print(f"  Error plotting spectrum width: {e}")

    # ── Raw 2-D array view ─────────────────────────────────────────────────
    print("\nGenerating raw 2-D array view…")
    try:
        raw_np = get_field_array(radar, "reflectivity", sweep=0)
        plot_raw_array(
            raw_np,
            title=f"Raw Reflectivity — Sweep 0 (rays × gates)  shape={raw_np.shape}",
            vmin=-32, vmax=75,
            save_path=os.path.join(PLOTS_DIR, "raw_reflectivity_array.png"),
        )
    except Exception as e:
        print(f"  Error generating raw array view: {e}")

    print(f"\n[OK] All plots saved to: {PLOTS_DIR}")
    print("Next step: python 03_enhance.py")


if __name__ == "__main__":
    main()
