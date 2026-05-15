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
    # --- Background / ignore colours ---
    ([0,   0,   0  ], -10.0),  # Black background
    ([255, 255, 255], -10.0),  # White / grid lines
    ([127, 127, 127], -10.0),  # Mid grey
    ([150, 150, 150], -10.0),  # Light grey
    ([100, 100, 100], -10.0),  # Dark grey
    ([200, 0,   90 ], -10.0),  # Boundary pink — ignore
    # --- Signal colours (low → high dBZ) ---
    ([0,   0,   252], 12.0),   # Blue
    ([0,   148, 252], 18.0),   # Sky blue
    ([0,   252, 252], 24.0),   # Cyan
    ([0,   160, 0  ], 30.0),   # Dark green
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
    2-D float32 array shaped (H, W).  Pixels that do not match any palette
    colour are set to -10.0 (background / no-data).
    """
    img  = Image.open(image_path).convert("RGB")
    data = np.array(img, dtype=np.int32)          # (H, W, 3)
    H, W = data.shape[:2]

    dbz = np.full((H, W), -10.0, dtype=np.float32)

    # Vectorised: compute L-inf distance to every palette colour simultaneously
    # data[:,:,None,:] shape: (H, W, 1, 3)
    # _PALETTE_COLORS shape:  (N, 3) → broadcast to (1, 1, N, 3)
    diff      = np.abs(data[:, :, None, :] - _PALETTE_COLORS[None, None, :, :])  # (H,W,N,3)
    linf_dist = diff.max(axis=-1)                                                  # (H,W,N)
    min_dist  = linf_dist.min(axis=-1)                                             # (H,W)
    best_idx  = linf_dist.argmin(axis=-1)                                          # (H,W)

    matched = min_dist <= _MATCH_THRESHOLD
    dbz[matched] = _PALETTE_VALUES[best_idx[matched]]

    # Remove legend box, range rings, and text by masking outside the radar circle
    dbz = _apply_circular_mask(dbz)

    return dbz


def _apply_circular_mask(dbz: np.ndarray) -> np.ndarray:
    """Zero out everything outside the main radar disc."""
    H, W = dbz.shape
    cy, cx = H // 2, W // 2
    # Use 0.90 of the shorter half-axis — keeps the full scan but removes the
    # legend strip that AEMET places in the corners.
    radius = min(cy, cx) * 0.90

    y, x = np.ogrid[:H, :W]
    outside = (y - cy) ** 2 + (x - cx) ** 2 > radius ** 2
    dbz[outside] = -10.0
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
