"""
utils/image_radar.py
--------------------
Utilities for converting processed radar images (AEMET/Meteocat)
back into numerical reflectivity fields (dBZ).
"""

import os
import numpy as np
from PIL import Image

# ── AEMET Palette Mapping (Approximate) ──────────────────────────────────────
# Based on common AEMET reflectivity scales.
AEMET_PALETTE = [
    ([0, 0, 0],       -10.0), # Black Background
    ([127, 127, 127], -10.0), # Grey Background
    ([255, 255, 255], -10.0), # White Background / Grid
    ([200, 0, 90],    -10.0), # Boundary color (ignore)
    ([150, 150, 150], -10.0), # Alt Grey
    ([100, 100, 100], -10.0), # Dark Grey
    ([0, 0, 252],     12.0),  # Light Blue
    ([0, 148, 252],   18.0),  # Mid Blue
    ([0, 252, 252],   24.0),  # Cyan
    ([0, 160, 0],     30.0),  # Dark Green
    ([0, 252, 0],     36.0),  # Bright Green
    ([252, 252, 0],   42.0),  # Yellow
    ([252, 172, 0],   48.0),  # Orange
    ([252, 116, 0],   54.0),  # Deep Orange
    ([252, 0, 0],     60.0),  # Red
    ([188, 0, 0],     66.0),  # Dark Red
    ([252, 0, 252],   72.0),  # Purple/Magenta
]

def image_to_dbz(image_path: str, source: str = "aemet") -> np.ndarray:
    """
    Loads an image and maps its RGB colors to dBZ values.
    Returns a 2D NumPy array.
    """
    img = Image.open(image_path).convert("RGB")
    data = np.array(img)
    h, w, _ = data.shape
    
    # Initialize with NaN or low value
    dbz = np.full((h, w), -10.0, dtype=np.float32)
    
    palette = AEMET_PALETTE # Default
    
    # Vectorized color matching (find closest color in palette)
    # For small palettes, this is fast enough.
    for color, val in palette:
        # [v3.7 Fix] Cast to int to prevent uint8 overflow/underflow during subtraction
        diff = np.abs(data.astype(int) - np.array(color).astype(int))
        mask = np.all(diff < 30, axis=-1)
        dbz[mask] = val
        
    # Apply circular mask to remove legend and text artifacts
    dbz = apply_circular_mask(dbz)
        
    return dbz

def apply_circular_mask(dbz):
    """
    Sets everything outside the central radar circle to -10.0 (background).
    """
    h, w = dbz.shape
    center_y, center_x = h // 2, w // 2
    # Radius is usually ~90-95% of the half-width
    radius = min(center_y, center_x) * 0.92
    
    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((y - center_y)**2 + (x - center_x)**2)
    
    mask = dist_from_center > radius
    dbz[mask] = -10.0
    return dbz

class ImageRadarMock:
    """
    A mock class that mimics a Py-ART Radar object but contains data 
    loaded from an image.
    """
    def __init__(self, filepath: str, source: str = "aemet"):
        self.filepath = filepath
        self.source = source
        self.fields = {
            "reflectivity": {
                "data": image_to_dbz(filepath, source)
            }
        }
        # Mock metadata
        self.metadata = {"instrument_name": f"{source.upper()} Image-based Radar"}
        self.latitude = {"data": [41.38]} # Default to Barcelona-ish
        self.longitude = {"data": [2.17]}
        self.altitude = {"data": [0]}
        self.nsweeps = 1
        
    def get_start_end(self, sweep):
        # For image-based, we only have 1 sweep which is the whole image
        return 0, self.fields["reflectivity"]["data"].shape[0] - 1

def is_image_file(filepath: str) -> bool:
    ext = os.path.splitext(filepath)[1].lower()
    return ext in [".gif", ".png", ".jpg", ".jpeg", ".dat"] # .dat for the raw downloads we did
