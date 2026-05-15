"""
utils/radar_io.py
-----------------
Radar file reading and field extraction — AEMET GIF edition.

Supported inputs
----------------
  • AEMET radar GIF / PNG images  → ImageRadarMock  (always available)
  • NEXRAD / ODIM / CF-Radial     → Py-ART Radar    (if pyart installed)
"""

from __future__ import annotations

import os
import numpy as np

from .image_radar import ImageRadarMock, is_image_file

try:
    import pyart
    HAS_PYART = True
except ImportError:
    HAS_PYART = False


# ── Reading ───────────────────────────────────────────────────────────────────

def read_radar(filepath: str):
    """
    Auto-detect format and return a radar data object.

    Returns
    -------
    ImageRadarMock  for AEMET GIF/PNG files (no extra dependencies needed)
    pyart.Radar     for NEXRAD/ODIM/CfRadial (requires pyart)

    Raises
    ------
    FileNotFoundError  if the file does not exist
    ImportError        if no reader can handle the file
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    print(f"Reading: {os.path.basename(filepath)}")

    # ── Image-based (AEMET) ───────────────────────────────────────────────
    if is_image_file(filepath):
        print("  Format: Image-based (AEMET)")
        return ImageRadarMock(filepath)

    # ── Binary radar formats (NEXRAD / ODIM / …) ─────────────────────────
    if HAS_PYART:
        try:
            radar = pyart.io.read(filepath)
            print("  Format: Py-ART")
            return radar
        except Exception as e:
            raise ImportError(f"Py-ART could not read '{filepath}': {e}") from e

    raise ImportError(
        "No radar reader available for this file type. "
        "Install pyart ('pip install arm-pyart') for binary radar formats."
    )


# ── Field extraction ──────────────────────────────────────────────────────────

def get_field_array(
    radar,
    field_name: str = "reflectivity",
    sweep: int = 0,
) -> np.ndarray:
    """
    Extract a radar field as a clean 2-D float32 NumPy array.

    Parameters
    ----------
    radar      : ImageRadarMock or pyart.Radar
    field_name : field to extract (default 'reflectivity')
    sweep      : sweep / elevation index (0 = lowest)

    Returns
    -------
    2-D float32 array of shape (rays, gates) or (H, W) for image mocks
    """
    # ── ImageRadarMock ────────────────────────────────────────────────────
    if isinstance(radar, ImageRadarMock):
        if field_name not in radar.fields:
            raise KeyError(
                f"Field '{field_name}' not in ImageRadarMock. "
                f"Available: {list(radar.fields.keys())}"
            )
        return radar.fields[field_name]["data"].copy()

    # ── Py-ART Radar ──────────────────────────────────────────────────────
    if HAS_PYART and hasattr(radar, "fields"):
        if field_name not in radar.fields:
            raise KeyError(
                f"Field '{field_name}' not found. "
                f"Available: {list(radar.fields.keys())}"
            )
        start, end = radar.get_start_end(sweep)
        data = radar.fields[field_name]["data"][start: end + 1]
        if isinstance(data, np.ma.MaskedArray):
            data = data.filled(np.nan)
        return data.astype(np.float32)

    raise ValueError(
        "Unrecognised radar object. Pass an ImageRadarMock or a pyart.Radar."
    )
