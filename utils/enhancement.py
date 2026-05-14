"""
utils/enhancement.py
--------------------
Core radar signal enhancement functions.

Pipeline overview
-----------------
  Raw data  →  [1] Denoise  →  [2] Upscale  →  [3] Sharpen  →  Enhanced data

Each step can be used independently or chained via `enhance_field()`.
"""

from __future__ import annotations

import os
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter, uniform_filter, median_filter
from scipy.interpolate import RegularGridInterpolator
from skimage.restoration import denoise_nl_means, estimate_sigma


# ── 1. Denoising ──────────────────────────────────────────────────────────────

def denoise_gaussian(
    data: np.ndarray,
    sigma: float = 1.0,
    preserve_nan: bool = True,
) -> np.ndarray:
    """
    Apply a Gaussian-weighted neighbourhood average to reduce noise.

    Each output gate is a weighted sum of itself and its neighbours.
    Neighbours closer to the centre gate contribute more (Gaussian weight).
    This is equivalent to a low-pass filter in the frequency domain.

    Parameters
    ----------
    data         : 2-D float array (rays × gates), NaN = missing
    sigma        : standard deviation of Gaussian kernel (gates)
    preserve_nan : if True, NaN positions in the input remain NaN in the output

    Returns
    -------
    Denoised 2-D float array
    """
    nan_mask = np.isnan(data)

    # Replace NaN with local mean for the convolution, then restore mask
    filled = np.where(nan_mask, 0.0, data)
    weight = np.where(nan_mask, 0.0, 1.0)

    smoothed = gaussian_filter(filled, sigma=sigma)
    norm     = gaussian_filter(weight, sigma=sigma)

    with np.errstate(invalid="ignore"):
        result = np.where(norm > 0, smoothed / norm, np.nan)

    if not preserve_nan:
        result = np.where(nan_mask, np.nan, result)

    return result.astype(np.float32)


def denoise_median_bilateral(
    data: np.ndarray,
    window: int = 3,
) -> np.ndarray:
    """
    Simple uniform (box) filter — faster than Gaussian, less edge-aware.
    Good for a quick sanity check.
    """
    nan_mask = np.isnan(data)
    filled   = np.where(nan_mask, 0.0, data)
    weight   = np.where(nan_mask, 0.0, 1.0)

    smoothed = uniform_filter(filled, size=window)
    norm     = uniform_filter(weight, size=window)

    with np.errstate(invalid="ignore"):
        result = np.where(norm > 0, smoothed / norm, np.nan)

    return result.astype(np.float32)


# ── 2. Super-resolution upscaling (Deep Learning v3) ──────────────────────────

def upscale(
    data: np.ndarray,
    scale: int = 2,
    method: str = "linear",
) -> np.ndarray:
    """
    Upscale a 2-D radar field by an integer factor using interpolation.

    "Predicts" intermediate gate / ray values that the radar never sampled,
    by fitting a smooth surface through the observed points.

    Parameters
    ----------
    data   : 2-D float array (rays × gates)
    scale  : integer upscaling factor (2 = double the resolution)
    method : 'linear' or 'nearest'

    Returns
    -------
    2-D float array of shape (rays*scale, gates*scale)
    """
    h, w = data.shape
    nan_mask = np.isnan(data)

    # Build the interpolator on the original grid
    y_old = np.linspace(0.0, 1.0, h)
    x_old = np.linspace(0.0, 1.0, w)

    # Fill NaN so the interpolator doesn't break
    filled = np.where(nan_mask, 0.0, data)
    interp = RegularGridInterpolator(
        (y_old, x_old), filled, method=method, bounds_error=False, fill_value=np.nan
    )

    # Query on the new (finer) grid
    y_new = np.linspace(0.0, 1.0, h * scale)
    x_new = np.linspace(0.0, 1.0, w * scale)
    yy, xx = np.meshgrid(y_new, x_new, indexing="ij")
    upscaled = interp((yy, xx))

    # Also upscale the NaN mask so missing regions stay missing
    nan_interp = RegularGridInterpolator(
        (y_old, x_old), nan_mask.astype(float), method="nearest",
        bounds_error=False, fill_value=1.0
    )
    upscaled_nan = nan_interp((yy, xx)) > 0.5
    upscaled[upscaled_nan] = np.nan

    return upscaled.astype(np.float32)


# ── 3. Sharpening ────────────────────────────────────────────────────────────

def sharpen_unsharp_mask(
    data: np.ndarray,
    sigma: float = 1.5,
    amount: float = 1.5,
) -> np.ndarray:
    """
    Unsharp masking: amplify fine-scale structure that survived upscaling.

    Formula:  output = data + amount * (data - blur(data))

    Parameters
    ----------
    data   : 2-D float array
    sigma  : blur radius used to compute the "unsharp" mask
    amount : how strongly to boost edge detail (1.0 = moderate)
    """
    nan_mask = np.isnan(data)
    filled   = np.where(nan_mask, 0.0, data)

    blur   = gaussian_filter(filled, sigma=sigma)
    result = filled + amount * (filled - blur)
    result[nan_mask] = np.nan

    return result.astype(np.float32)


# ── 4. Full pipeline ──────────────────────────────────────────────────────────

def denoise_advanced(data: np.ndarray, h: float = 0.8) -> np.ndarray:
    """
    Apply Non-Local Means (NLM) denoising.
    Unlike local filters, NLM searches for similar patches to preserve 
    the complex textures of storm cells.
    """
    # NLM expects 0-1 range for best performance
    valid_data = np.nan_to_num(data, nan=-32.0)
    d_min, d_max = -32.0, 75.0
    norm = (valid_data - d_min) / (d_max - d_min)
    
    # Fast NLM
    sigma_est = np.mean(estimate_sigma(norm))
    patch_kw = dict(patch_size=5, patch_distance=6)
    denoised = denoise_nl_means(norm, h=h * sigma_est, fast_mode=True, **patch_kw)
    
    return (denoised * (d_max - d_min) + d_min).astype(np.float32)


class ChannelAttention(nn.Module):
    """
    Channel Attention Module (Squeeze-and-Excitation).
    Allows the network to adaptively recalibrate channel-wise feature 
    responses by explicitly modelling interdependencies between channels.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class RadarResBlock(nn.Module):
    """
    Residual Block with Channel Attention (RCAB).
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1)
        )
        self.ca = ChannelAttention(channels)

    def forward(self, x):
        res = self.conv(x)
        res = self.ca(res)
        return x + res

class RadarResNet(nn.Module):
    """
    RadarResNet-v3: Deep Residual Channel Attention Network.
    Optimised for high-fidelity radar reflectivity super-resolution.
    """
    def __init__(self, scale: int = 2):
        super().__init__()
        self.scale = scale
        # Initial feature extraction
        self.conv_in = nn.Conv2d(1, 64, 3, padding=1)
        
        # Deep Residual Architecture (8 Attention Blocks)
        self.res_blocks = nn.Sequential(*[RadarResBlock(64) for _ in range(8)])
        
        # Global Skip Connection refinement
        self.conv_after_res = nn.Conv2d(64, 64, 3, padding=1)
        
        # Upsampling via PixelShuffle
        if scale == 2:
            self.upsample = nn.Sequential(
                nn.Conv2d(64, 256, 3, padding=1),
                nn.PixelShuffle(2)
            )
        else:
            self.upsample = nn.Upsample(scale_factor=scale, mode='bicubic', align_corners=False)
            
        # Final reconstruction
        self.conv_out = nn.Conv2d(64, 1, 3, padding=1)

    def forward(self, x):
        feat = self.conv_in(x)
        res = self.res_blocks(feat)
        res = self.conv_after_res(res)
        feat = feat + res # Global skip connection
        up = self.upsample(feat)
        return self.conv_out(up)


def local_contrast_enhancement(data: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """
    Adaptive reflectivity stretching. 
    Brings out faint stratiform details while preserving storm cores.
    """
    nan_mask = np.isnan(data)
    # Use a slightly higher floor for LCE to avoid background noise amplification
    bg_threshold = -5.0
    
    # Mask out background for stretching calculation
    signal_mask = data > bg_threshold
    if not np.any(signal_mask):
        return data
        
    valid = np.nan_to_num(data, nan=-10.0)
    
    # Adaptive sigmoid stretching
    mean = np.mean(valid[signal_mask])
    std = np.std(valid[signal_mask])
    std = max(std, 0.5)
    
    # Soft thresholding and stretching
    normalized = (valid - mean) / std
    stretched = 1 / (1 + np.exp(-0.7 * normalized))
    
    # Map back to dBZ range
    result = stretched * (np.max(valid) - np.min(valid)) + np.min(valid)
    
    # Blend back: only apply LCE where there is signal, keep background flat
    final = np.where(signal_mask, result, valid)
    final[nan_mask] = np.nan
    return final.astype(np.float32)

def enhance_field(
    data: np.ndarray,
    denoise_h: float = 0.6,
    scale: int = 2,
    sharpen_amount: float = 0.8,
    use_resnet: bool = True
) -> np.ndarray:
    """
    Radar Enhancement Pipeline v3:
    1. Non-Local Means (NLM) Denoising
    2. RadarResNet-v3 Deep Super-Resolution
    3. Local Contrast Enhancement (LCE)
    4. Adaptive Edge Sharpening
    """
    # [1] NLM Denoise
    denoised = denoise_advanced(data, h=denoise_h)

    # [2] Super-resolution
    # [v3.9 Fix] Only execute ResNet if weights are available. 
    # Random weights destroy meteorological signal integrity.
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "models", "resnet_radar.pth")
    if use_resnet and os.path.exists(model_path):
        try:
            enhanced = _run_resnet(denoised, scale=scale)
        except Exception as e:
            print(f"ResNet Execution Error: {e}. Falling back to Bicubic.")
            enhanced = upscale(denoised, scale=scale)
    else:
        # High-fidelity bicubic fallback
        enhanced = upscale(denoised, scale=scale)

    # [3] Post-ResNet Smoothing (To remove dots/grid artifacts)
    from scipy.ndimage import gaussian_filter
    enhanced = gaussian_filter(enhanced, sigma=0.6)

    # [4] Reflective Conservatism (Prevent saturation/overestimation)
    # Ensure enhanced peak does not exceed original peak significantly
    orig_max = np.nanmax(denoised)
    enh_max = np.nanmax(enhanced)
    if enh_max > orig_max and orig_max > 0:
        # Scale back to original peak if it overestimates signal
        enhanced = enhanced * (orig_max / enh_max)

    # [5] Subtle Edge Sharpening
    # Very conservative amount to avoid amplifying noise
    signal_mask = enhanced > 5.0
    sharpened = sharpen_unsharp_mask(enhanced, sigma=0.8, amount=sharpen_amount * 0.4)
    enhanced = np.where(signal_mask, sharpened, enhanced)
    
    return enhanced


_MODEL_CACHE = {}

def _run_resnet(data: np.ndarray, scale: int = 2) -> np.ndarray:
    """Inference helper for RadarResNet."""
    global _MODEL_CACHE
    
    cache_key = (scale,)
    model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "models", "resnet_radar.pth")
    
    if cache_key not in _MODEL_CACHE:
        model = RadarResNet(scale=scale)
        model.load_state_dict(torch.load(model_path, map_location='cpu', weights_only=True))
        model.eval()
        _MODEL_CACHE[cache_key] = model
    
    model = _MODEL_CACHE[cache_key]
    
    # Normalise
    d_min, d_max = -32.0, 75.0
    norm = (data - d_min) / (d_max - d_min)
    inp = torch.from_numpy(norm[np.newaxis, np.newaxis]).float()
    
    with torch.no_grad():
        pred = model(inp).numpy()[0, 0]
        
    # [v3.1 Fix] Post-upsampling smoothing to remove checkerboard/grid artifacts
    # A very subtle blur breaks the PixelShuffle grid without losing storm detail
    pred = gaussian_filter(pred, sigma=0.4)
        
    return (pred * (d_max - d_min) + d_min).astype(np.float32)


# ── 5. Quality metrics ────────────────────────────────────────────────────────

def compute_snr(data: np.ndarray, noise_floor_dBZ: float = -32.0) -> float:
    """
    Estimate the signal-to-noise ratio by comparing high-intensity areas
    (signal) to the low-intensity background (noise).
    """
    valid = data[~np.isnan(data)]
    if len(valid) == 0: return 0.0
    
    # Simple estimate: Signal is top 10% of reflectivity, noise is mean of rest
    sorted_v = np.sort(valid)
    idx = int(len(sorted_v) * 0.9)
    signal = np.mean(sorted_v[idx:])
    noise  = np.mean(sorted_v[:idx])
    
    return float(signal - noise)


# ── 6. Synthetic generation ──────────────────────────────────────────────────

def generate_synthetic_radar_field(h: int = 256, w: int | None = None, seed: int | None = None, size: int | None = None) -> np.ndarray:
    """
    Generate a synthetic reflectivity field for testing and training.
    Supports both rectangular (h, w) and square (size) grids.
    """
    if size is not None:
        h = w = size
    if w is None:
        w = h

    field = np.full((h, w), -15.0, dtype=np.float32)
    rng = np.random.default_rng(seed)
    
    # Stratiform background (smooth gradients)
    xx, yy = np.meshgrid(np.linspace(0, 10, w), np.linspace(0, 10, h))
    field += 25 * (0.5 + 0.5 * np.sin(xx * 0.5) * np.cos(yy * 0.3))
    
    # Convective cells (Gaussian blobs)
    for _ in range(12):
        ry, rx = rng.integers(0, h), rng.integers(0, w)
        amp = rng.uniform(30, 60)
        sigma = rng.uniform(min(h, w)/50, min(h, w)/10)
        y, x = np.ogrid[:h, :w]
        dist = (y - ry)**2 + (x - rx)**2
        field += amp * np.exp(-dist / (2 * sigma**2))
        
    # Add some "noise"
    field += rng.normal(0, 2.0, (h, w))
    
    # Optional mask for small patches (training)
    if h > 100:
        field[:, :20] = np.nan
    
    return np.clip(field, -32.0, 75.0).astype(np.float32)
