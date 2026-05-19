"""
utils/image_radar.py
--------------------
Converts AEMET radar GIF images into numerical reflectivity fields (dBZ).
"""

from __future__ import annotations
import os
import numpy as np
from PIL import Image


# ── AEMET Palette ─────────────────────────────────────────────────────────────
# Each entry: (RGB tuple, dBZ value)
# Background / non-signal colours all map to -10.0 (treated as no-data).
AEMET_PALETTE = [
    # --- Background / non-signal colours → NaN (rendered as transparent/black) ---
    ([0,   0,   0  ], np.nan),  # Black — inside disc but no signal
    ([255, 255, 255], np.nan),  # White / grid lines
    ([127, 127, 127], np.nan),  # Mid grey — outer background
    ([150, 150, 150], np.nan),  # Light grey variant
    ([100, 100, 100], np.nan),  # Dark grey variant
    ([200, 0,   90 ], np.nan),  # Boundary / radar-circle outline — ignore
    # --- Signal colours (low → high dBZ) ---
    # Exact RGB values sampled from real AEMET GIFs (verified against pixel data).
    ([0,   0,   252], 12.0),   # Blue
    ([0,   148, 252], 18.0),   # Sky blue
    ([0,   252, 252], 24.0),   # Cyan
    ([0,   192, 0  ], 30.0),   # Dark green  ← was (0,160,0) — FIXED
    ([0,   252, 0  ], 36.0),   # Bright green
    ([252, 252, 0  ], 42.0),   # Yellow
    ([252, 172, 0  ], 48.0),   # Orange
    ([252, 116, 0  ], 54.0),   # Deep orange
    ([252, 0,   0  ], 60.0),   # Red
    ([188, 0,   0  ], 66.0),   # Dark red
    ([252, 0,   252], 72.0),   # Magenta / purple
]

# Pre-build arrays for vectorised matching (avoids per-pixel Python loops)
_PALETTE_COLORS = np.array([c for c, _ in AEMET_PALETTE], dtype=np.int32)   # (N, 3)
_PALETTE_VALUES = np.array([v for _, v in AEMET_PALETTE], dtype=np.float32)  # (N,)

# Maximum L-inf distance to accept a colour match.
# 15 is tight enough to separate palette entries (which differ by ~50+ per channel)
# while still handling minor JPEG/GIF compression artefacts.
_MATCH_THRESHOLD = 15


def image_to_dbz(image_path: str) -> np.ndarray:
    """
    Load an AEMET radar image and map every pixel to a dBZ value.

    Returns
    -------
    2-D float32 array shaped (H, W).
    Signal pixels  → dBZ value (12 … 72).
    Background / unmatched pixels → np.nan (renders as transparent/black).
    """
    img  = Image.open(image_path).convert("RGB")
    data = np.array(img, dtype=np.int32)          # (H, W, 3)
    H, W = data.shape[:2]

    # Start fully NaN — unmatched pixels stay NaN (background)
    dbz = np.full((H, W), np.nan, dtype=np.float32)

    # Vectorised L-inf distance to every palette colour simultaneously
    diff      = np.abs(data[:, :, None, :] - _PALETTE_COLORS[None, None, :, :])  # (H,W,N,3)
    linf_dist = diff.max(axis=-1)                                                  # (H,W,N)
    min_dist  = linf_dist.min(axis=-1)                                             # (H,W)
    best_idx  = linf_dist.argmin(axis=-1)                                          # (H,W)

    matched = min_dist <= _MATCH_THRESHOLD
    dbz[matched] = _PALETTE_VALUES[best_idx[matched]]
    # NaN palette entries (background colours) stay NaN even when matched

    # Mask outside the radar disc → NaN
    dbz = _apply_circular_mask(dbz)

    return dbz


def _apply_circular_mask(dbz: np.ndarray) -> np.ndarray:
    """Set everything outside the main radar disc to NaN."""
    H, W = dbz.shape
    cy, cx = H // 2, W // 2
    radius = min(cy, cx) * 0.90
    y, x = np.ogrid[:H, :W]
    outside = (y - cy) ** 2 + (x - cx) ** 2 > radius ** 2
    dbz[outside] = np.nan
    return dbz


# ── ImageRadarMock ────────────────────────────────────────────────────────────

class ImageRadarMock:
    """
    Minimal stand-in for a Py-ART Radar object built from an AEMET image.
    Exposes only what the pipeline scripts actually use.
    """

    def __init__(self, filepath: str):
        self.filepath = filepath
        dbz_field = image_to_dbz(filepath)

        self.fields = {
            "reflectivity": {"data": dbz_field}
        }
        self.metadata  = {"instrument_name": "AEMET Image Radar"}
        self.latitude  = {"data": [40.0]}   # Approx centre of Spain
        self.longitude = {"data": [-3.7]}
        self.altitude  = {"data": [0]}
        self.nsweeps   = 1

    def get_start_end(self, sweep: int = 0):
        n = self.fields["reflectivity"]["data"].shape[0]
        return 0, n - 1


def is_image_file(filepath: str) -> bool:
    return os.path.splitext(filepath)[1].lower() in {".gif", ".png", ".jpg", ".jpeg"}
