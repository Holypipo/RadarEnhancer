"""
utils/radar_io.py
-----------------
Radar file reading and field extraction.

Supported sources
-----------------
  AEMET  (.gif / .png starting with 'aemet_')  → AemetRadar  (always available)
  NEXRAD / ODIM / CF-Radial (binary)           → pyart.Radar (requires arm-pyart)
"""

from __future__ import annotations
import os
import numpy as np

from .image_radar import AemetRadar, is_aemet_file

try:
    import pyart
    HAS_PYART = True
except ImportError:
    HAS_PYART = False


def read_radar(filepath: str):
    """
    Auto-detect source and return a radar data object.

    Returns
    -------
    AemetRadar   for AEMET GIF/PNG files
    pyart.Radar  for NEXRAD / ODIM / CfRadial (requires pyart)

    Raises
    ------
    FileNotFoundError  – file does not exist
    ImportError        – no reader available for this format
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    print(f"Reading: {os.path.basename(filepath)}")

    if is_aemet_file(filepath):
        print("  Source: AEMET image")
        return AemetRadar(filepath)

    if HAS_PYART:
        try:
            radar = pyart.io.read(filepath)
            print("  Source: NEXRAD / Py-ART")
            return radar
        except Exception as e:
            raise ImportError(
                f"Py-ART could not read '{os.path.basename(filepath)}': {e}"
            ) from e

    raise ImportError(
        "No reader available for this file type. "
        "Install arm-pyart ('pip install arm-pyart') for NEXRAD / binary formats."
    )


def get_field_array(radar, field_name: str = "reflectivity", sweep: int = 0) -> np.ndarray:
    """
    Extract a radar field as a clean 2-D float32 array.

    Parameters
    ----------
    radar      : AemetRadar or pyart.Radar
    field_name : field to extract
    sweep      : elevation index (0 = lowest, AEMET always 0)

    Returns
    -------
    float32 (H, W) array — NaN where no signal / background
    """
    if isinstance(radar, AemetRadar):
        if field_name not in radar.fields:
            raise KeyError(
                f"Field '{field_name}' not in AemetRadar. "
                f"Available: {list(radar.fields.keys())}"
            )
        return radar.fields[field_name]["data"].copy()

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

    raise ValueError("Unrecognised radar object. Pass AemetRadar or pyart.Radar.")
