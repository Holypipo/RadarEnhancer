"""
utils/radar_io.py
-----------------
Wrappers around Py-ART for reading radar files in various formats and
extracting fields as clean NumPy arrays.

Supported formats (via Py-ART auto-detection):
    • NEXRAD Level-II  (.bzip2 / no extension)
    • ODIM HDF5        (.h5, .hdf5)
    • CF/Radial NetCDF (.nc)
    • UF, Sigmet, DORADE …
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

try:
    from metpy.io import Level2File
    HAS_METPY = True
except ImportError:
    HAS_METPY = False


# ── Reading ───────────────────────────────────────────────────────────────────

def read_radar(filepath: str) -> any:
    """
    Auto-detect format and return a radar data object.
    Supports Py-ART (Radar) and MetPy (Level2File) fallback.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Radar file not found: {filepath}")

    print(f"Reading: {os.path.basename(filepath)}")
    
    if HAS_PYART:
        try:
            radar = pyart.io.read(filepath)
            print("  Format: Py-ART")
            return radar
        except Exception as e:
            # If it's an image file, don't worry about Py-ART failing
            if not is_image_file(filepath):
                print(f"  Py-ART read failed: {e}")

    # ── Image-based Radar ──────────────────────────────────────────────────
    if is_image_file(filepath):
        try:
            print("  Format: Image-based (AEMET/Meteocat)")
            return ImageRadarMock(filepath)
        except Exception as e:
            print(f"  Image load failed: {e}")

    if HAS_METPY:
        try:
            # Check if it's a NEXRAD L2 file
            radar = Level2File(filepath)
            print("  Format: MetPy Level2File")
            return radar
        except Exception as e:
            print(f"  MetPy read failed: {e}")

    raise ImportError("No compatible radar reader (Py-ART or MetPy) installed or file format unsupported.")


# ── Field extraction ──────────────────────────────────────────────────────────

def get_field_array(
    radar: any,
    field_name: str,
    sweep: int = 0,
) -> np.ndarray:
    """
    Extract a single sweep from a radar field as a 2-D NumPy array.
    """
    # ── Py-ART Branch ────────────────────────────────────────────────────────
    if HAS_PYART and hasattr(radar, 'fields'):
        if field_name not in radar.fields:
            available = list(radar.fields.keys())
            raise KeyError(f"Field '{field_name}' not found. Available: {available}")

        start, end = radar.get_start_end(sweep)
        data = radar.fields[field_name]["data"][start : end + 1]
        if isinstance(data, np.ma.MaskedArray):
            data = data.filled(np.nan)
        return data.astype(np.float32)

    # ── Image Mock Branch ──────────────────────────────────────────────────
    if isinstance(radar, ImageRadarMock):
        if field_name not in radar.fields:
             raise KeyError(f"Field '{field_name}' not available in ImageMock.")
        return radar.fields[field_name]["data"]

    # ── MetPy Branch ─────────────────────────────────────────────────────────
    if HAS_METPY and hasattr(radar, 'sweeps'):
        # Map common field names to MetPy's byte-string keys
        field_map = {
            'reflectivity': b'REF',
            'velocity': b'VEL',
            'spectrum_width': b'SW'
        }
        internal_key = field_map.get(field_name, field_name)
        if isinstance(internal_key, str):
            internal_key = internal_key.encode('ascii')
        
        # MetPy Level2File structure: radar.sweeps is a list
        # Each sweep has a list of 'rays' (Radial namedtuples)
        try:
            s = radar.sweeps[sweep]
            # Filter rays that have the requested field in their 'moments' dict
            data_rays = []
            for r in s:
                if hasattr(r, 'moments') and internal_key in r.moments:
                    # MetPy stores (header, data_array)
                    data_rays.append(np.array(r.moments[internal_key][1]))
            
            if not data_rays:
                raise KeyError(f"Field {internal_key} not found in MetPy sweep.")
            
            # Pad rays to consistent length
            max_gates = max(len(r) for r in data_rays)
            grid = np.full((len(data_rays), max_gates), np.nan, dtype=np.float32)
            for i, r in enumerate(data_rays):
                grid[i, :len(r)] = r
            return grid
        except Exception as e:
            raise ValueError(f"Error extracting field via MetPy: {e}")

    raise ValueError("Radar object type not recognized or reader not installed.")


# ── Metadata helpers ──────────────────────────────────────────────────────────

def radar_summary(radar: pyart.core.Radar) -> dict:
    """Return a compact summary dict of radar metadata."""
    return {
        "instrument_name": radar.metadata.get("instrument_name", "unknown"),
        "latitude_deg": float(radar.latitude["data"][0]),
        "longitude_deg": float(radar.longitude["data"][0]),
        "altitude_m": float(radar.altitude["data"][0]),
        "nsweeps": radar.nsweeps,
        "nrays": radar.nrays,
        "ngates": radar.ngates,
        "fields": list(radar.fields.keys()),
        "sweep_elevations_deg": radar.fixed_angle["data"].tolist(),
    }
