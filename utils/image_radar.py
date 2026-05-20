"""
utils/image_radar.py
--------------------
Converts AEMET radar GIF images into numerical reflectivity fields (dBZ).
Supports AEMET nacional and regional products only.
"""

from __future__ import annotations
import os
import numpy as np
from PIL import Image


# ── AEMET Colour Palette ──────────────────────────────────────────────────────
# Exact RGB values verified by sampling real AEMET GIF pixels.
# Each entry: ([R, G, B], dBZ_value)
# Background / non-signal colours map to NaN so they render as black.

AEMET_PALETTE = [
    # Background / ignore
    ([0,   0,   0  ], np.nan),  # Black — no signal inside disc
    ([255, 255, 255], np.nan),  # White / grid lines
    ([127, 127, 127], np.nan),  # Mid grey — outer background
    ([150, 150, 150], np.nan),  # Light grey variant
    ([100, 100, 100], np.nan),  # Dark grey variant
    ([200, 0,   90 ], np.nan),  # Boundary / radar-circle outline
    # Signal (low → high dBZ)
    ([0,   0,   252], 12.0),
    ([0,   148, 252], 18.0),
    ([0,   252, 252], 24.0),
    ([67,  131, 35 ], 30.0),   # Dark green in real echoes on AEMET national GIFs
    ([0,   192, 0  ], 30.0),   # Dark green — verified fix (was 0,160,0)
    ([0,   252, 0  ], 36.0),
    ([252, 252, 0  ], 42.0),
    ([252, 172, 0  ], 48.0),
    ([252, 116, 0  ], 54.0),
    ([252, 0,   0  ], 60.0),
    ([188, 0,   0  ], 66.0),
    ([252, 0,   252], 72.0),
]

_PALETTE_COLORS = np.array([c for c, _ in AEMET_PALETTE], dtype=np.int32)
_PALETTE_VALUES = np.array([v for _, v in AEMET_PALETTE], dtype=np.float32)

# Tight threshold: palette entries differ by ~50+ per channel; 15 handles
# minor GIF compression artefacts without cross-palette bleeding.
_MATCH_THRESHOLD = 15


def image_to_dbz(image_path: str) -> np.ndarray:
    """
    Load an AEMET radar GIF and map every pixel to a dBZ value.

    Returns
    -------
    float32 array (H, W).
    Signal pixels  → dBZ (12–72).
    Background / unmatched → np.nan.
    """
    with Image.open(image_path) as img:
        data = np.array(img.convert("RGB"), dtype=np.int32)

    H, W = data.shape[:2]
    dbz  = np.full((H, W), np.nan, dtype=np.float32)

    # Vectorised L-inf distance to every palette entry
    diff      = np.abs(data[:, :, None, :] - _PALETTE_COLORS[None, None, :, :])
    linf_dist = diff.max(axis=-1)          # (H, W, N)
    min_dist  = linf_dist.min(axis=-1)    # (H, W)
    best_idx  = linf_dist.argmin(axis=-1) # (H, W)

    matched       = min_dist <= _MATCH_THRESHOLD
    dbz[matched]  = _PALETTE_VALUES[best_idx[matched]]
    # NaN palette entries stay NaN even when matched

    return _apply_circular_mask(dbz)


def _apply_circular_mask(dbz: np.ndarray) -> np.ndarray:
    """Set everything outside the main radar disc to NaN."""
    H, W   = dbz.shape
    cy, cx = H // 2, W // 2
    radius = min(cy, cx) * 0.90
    y, x   = np.ogrid[:H, :W]
    dbz[(y - cy) ** 2 + (x - cx) ** 2 > radius ** 2] = np.nan
    return dbz


class AemetRadar:
    """
    Minimal radar-like object built from an AEMET GIF image.
    Holds the raw PIL image (for display) and the dBZ field (for processing).
    """

    def __init__(self, filepath: str):
        self.filepath   = filepath
        self.source     = "aemet"
        with Image.open(filepath) as img:
            self.pil_image = img.convert("RGB")   # original for display
        dbz_field       = image_to_dbz(filepath)
        self.fields     = {"reflectivity": {"data": dbz_field}}
        self.metadata   = {"instrument_name": "AEMET Image Radar"}
        self.latitude   = {"data": [40.0]}
        self.longitude  = {"data": [-3.7]}
        self.nsweeps    = 1

    def get_start_end(self, sweep: int = 0):
        n = self.fields["reflectivity"]["data"].shape[0]
        return 0, n - 1


def is_aemet_file(filepath: str) -> bool:
    """
    Returns True for AEMET radar GIF/PNG files.
    Accepts both old naming (aemet_nacional_...) and new (01_AemetNacional_...).
    """
    name = os.path.basename(filepath).lower()
    ext  = os.path.splitext(name)[1]
    if ext not in {".gif", ".png"}:
        return False
    # New timestamped naming: 01_AemetNacional_... or 01_AemetBa_... etc.
    if name.startswith("01_aemet"):
        return True
    # Legacy naming: aemet_nacional_... / aemet_ba_...
    if name.startswith("aemet_"):
        return True
    return False
