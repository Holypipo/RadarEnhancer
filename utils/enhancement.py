"""
utils/enhancement.py
--------------------
Radar signal enhancement functions — AEMET edition.

Pipeline
--------
  Raw (dBZ)  →  [1] Denoise  →  [2] Upscale  →  [3] Sharpen  →  Enhanced (dBZ)

Torch / ResNet are imported lazily so the module loads even when PyTorch is
not installed; the pipeline gracefully falls back to scipy interpolation.
"""

from __future__ import annotations

import os
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator

from config import get_resnet_model_path


# ── 1. Denoising ──────────────────────────────────────────────────────────────

def denoise_gaussian(
    data: np.ndarray,
    sigma: float = 1.0,
    preserve_nan: bool = True,
) -> np.ndarray:
    """
    Gaussian-weighted neighbourhood average (low-pass filter).

    NaN pixels are excluded from the kernel average so they don't pull
    neighbouring values toward zero.

    Parameters
    ----------
    data         : 2-D float array (H × W), NaN = missing / background
    sigma        : kernel standard deviation in pixels
    preserve_nan : keep input NaN positions NaN in output
    """
    nan_mask = np.isnan(data)
    filled   = np.where(nan_mask, 0.0, data)
    weight   = np.where(nan_mask, 0.0, 1.0)

    smoothed = gaussian_filter(filled, sigma=sigma)
    norm     = gaussian_filter(weight, sigma=sigma)

    with np.errstate(invalid="ignore"):
        result = np.where(norm > 0, smoothed / norm, np.nan)

    if not preserve_nan:
        result = np.where(nan_mask, np.nan, result)

    return result.astype(np.float32)


def denoise_advanced(data: np.ndarray, h: float = 0.8) -> np.ndarray:
    """
    Boundary-preserving palette smoother for AEMET radar images.

    AEMET GIFs are palette-quantised: every signal pixel is exactly 12, 18,
    24 … 72 dBZ with no spatial noise.  Classic denoisers (Gaussian, NLM)
    spread the kernel across the NaN background, inflating each 1-pixel dot
    into a large blob.  This function smooths ONLY within the existing signal
    footprint — never expanding it into background pixels.

    What it does:
      • Softens the hard step-edges between adjacent palette levels
        (e.g. a 12 dBZ pixel next to an 18 dBZ pixel gets blended to ~15 dBZ)
      • Zero boundary expansion — the signal footprint is preserved exactly
      • Fast: single Gaussian pass with a small sigma

    Parameters
    ----------
    data : 2-D float32 array in dBZ, NaN = background / no-data
    h    : unused (kept for API compatibility)
    """
    nan_mask    = np.isnan(data)
    signal_mask = ~nan_mask           # pixels that actually have signal

    filled   = np.where(nan_mask, 0.0, data)
    weight   = np.where(nan_mask, 0.0, 1.0)

    # Small sigma: enough to blend adjacent palette levels, not enough to expand
    smoothed = gaussian_filter(filled, sigma=0.8)
    norm     = gaussian_filter(weight, sigma=0.8)

    with np.errstate(invalid="ignore", divide="ignore"):
        blended = np.where(norm > 0, smoothed / norm, data)

    # Write blended values ONLY where there was already signal — strict boundary
    result = data.copy()
    result[signal_mask] = blended[signal_mask]
    return result.astype(np.float32)


# ── 2. Upscaling ─────────────────────────────────────────────────────────────

def upscale(
    data: np.ndarray,
    scale: int = 2,
    method: str = "linear",
) -> np.ndarray:
    """
    Upscale a 2-D radar field by an integer factor using grid interpolation.

    NaN pixels (background) are handled carefully: the interpolator fills NaN
    regions with the nearest signal value so that linear blending at signal
    edges doesn't fade toward zero — the NaN mask is then re-applied strictly
    to restore the correct background.

    Parameters
    ----------
    data   : 2-D float array (H × W), NaN = background
    scale  : integer upscaling factor  (2 → double resolution)
    method : 'linear' (smooth) or 'nearest' (crisp edges)
    """
    H, W     = data.shape
    nan_mask = np.isnan(data)

    # Fill NaN with nearest-neighbour signal values so the interpolator
    # doesn't blend toward 0 at signal boundaries
    if np.any(nan_mask) and np.any(~nan_mask):
        from scipy.ndimage import distance_transform_edt
        _, idx = distance_transform_edt(nan_mask, return_indices=True)
        filled = data[idx[0], idx[1]]          # nearest-signal fill
        filled = np.where(nan_mask, filled, data)
    else:
        filled = np.where(nan_mask, 0.0, data)

    y_old = np.linspace(0.0, 1.0, H)
    x_old = np.linspace(0.0, 1.0, W)
    interp = RegularGridInterpolator(
        (y_old, x_old), filled, method=method,
        bounds_error=False, fill_value=np.nan,
    )

    y_new = np.linspace(0.0, 1.0, H * scale)
    x_new = np.linspace(0.0, 1.0, W * scale)
    yy, xx = np.meshgrid(y_new, x_new, indexing="ij")
    upscaled = interp((yy, xx))

    # Re-apply NaN mask: use nearest interpolation of original mask
    nan_interp = RegularGridInterpolator(
        (y_old, x_old), nan_mask.astype(float), method="nearest",
        bounds_error=False, fill_value=1.0,
    )
    upscaled[nan_interp((yy, xx)) > 0.5] = np.nan

    return upscaled.astype(np.float32)


# ── 3. Sharpening ────────────────────────────────────────────────────────────

def sharpen_unsharp_mask(
    data: np.ndarray,
    sigma: float = 1.5,
    amount: float = 1.0,
) -> np.ndarray:
    """
    Unsharp masking: amplify fine-scale edges that survived upscaling.

    output = data + amount × (data − blur(data))

    Parameters
    ----------
    sigma  : blur radius for the unsharp mask
    amount : edge-boost strength (0.5–1.5 is typical; >2 amplifies noise)
    """
    nan_mask = np.isnan(data)
    filled   = np.where(nan_mask, 0.0, data)
    blur     = gaussian_filter(filled, sigma=sigma)
    result   = filled + amount * (filled - blur)
    result[nan_mask] = np.nan
    return result.astype(np.float32)


# ── 4. Full pipeline ──────────────────────────────────────────────────────────

def enhance_field(
    data: np.ndarray,
    denoise_h: float = 0.6,
    scale: int = 2,
    sharpen_amount: float = 0.5,
    use_resnet: bool = True,
) -> np.ndarray:
    """
    Full AEMET radar enhancement pipeline:
      1. Palette smoother  (softens hard dBZ level edges, no boundary expansion)
      2. Super-resolution  (ResNet if trained weights exist, else bicubic)
      3. Edge sharpening   (unsharp mask, signal pixels only)
      4. dBZ clip          (prevents values leaving physical range)

    Parameters
    ----------
    data           : raw dBZ array from AEMET GIF (NaN = background)
    denoise_h      : unused — kept for API compat
    scale          : upscaling factor (2 = double resolution)
    sharpen_amount : unsharp-mask boost strength
    use_resnet     : use trained ResNet if weights are present
    """
    # [1] Palette smoother — softens step edges, strict boundary preservation
    denoised = denoise_advanced(data, h=denoise_h)

    # [2] Super-resolution
    model_path = get_resnet_model_path()
    if use_resnet and os.path.exists(model_path):
        try:
            enhanced = _run_resnet(denoised, scale=scale)
        except Exception as e:
            print(f"  [ResNet] Inference failed ({e}). Falling back to bicubic.")
            enhanced = upscale(denoised, scale=scale)
    else:
        enhanced = upscale(denoised, scale=scale)

    # [3] Subtle edge sharpening — signal pixels only, conservative amount
    nan_mask_e = np.isnan(enhanced)
    filled_e   = np.where(nan_mask_e, 0.0, enhanced)
    blur_e     = gaussian_filter(filled_e, sigma=1.2)
    sharpened  = filled_e + sharpen_amount * (filled_e - blur_e)
    sharpened[nan_mask_e] = np.nan

    # [4] Clip to physical dBZ range, restore NaN background
    result = np.clip(sharpened, 0.0, 75.0)
    result[nan_mask_e] = np.nan

    return result


# ── 5. ResNet (lazy import) ───────────────────────────────────────────────────

_MODEL_CACHE: dict = {}


def _get_torch_model(scale: int):
    """Load and cache the ResNet. Returns None if torch is not installed."""
    global _MODEL_CACHE
    if scale in _MODEL_CACHE:
        return _MODEL_CACHE[scale]

    try:
        import torch
        from .resnet import RadarResNet  # imported here to avoid top-level torch dep
    except ImportError:
        return None

    model_path = get_resnet_model_path()
    if not os.path.exists(model_path):
        return None

    model = RadarResNet(scale=scale)
    model.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=True)
    )
    model.eval()
    _MODEL_CACHE[scale] = model
    return model


def _run_resnet(data: np.ndarray, scale: int = 2) -> np.ndarray:
    """Run ResNet super-resolution. Raises RuntimeError if unavailable."""
    model = _get_torch_model(scale)
    if model is None:
        raise RuntimeError("torch or model weights not available")

    import torch

    d_min, d_max = -32.0, 75.0
    norm = (np.nan_to_num(data, nan=-10.0) - d_min) / (d_max - d_min)
    inp  = torch.from_numpy(norm[None, None]).float()

    with torch.no_grad():
        pred = model(inp).numpy()[0, 0]

    pred = gaussian_filter(pred, sigma=0.4)
    result = (pred * (d_max - d_min) + d_min).astype(np.float32)
    result[_resample_mask_nearest(np.isnan(data), result.shape)] = np.nan
    return result


def _resample_mask_nearest(mask: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    """Resize a boolean mask with nearest-neighbour sampling."""
    H, W = mask.shape
    target_h, target_w = target_shape

    y_old = np.linspace(0.0, 1.0, H)
    x_old = np.linspace(0.0, 1.0, W)
    y_new = np.linspace(0.0, 1.0, target_h)
    x_new = np.linspace(0.0, 1.0, target_w)
    yy, xx = np.meshgrid(y_new, x_new, indexing="ij")

    interp = RegularGridInterpolator(
        (y_old, x_old),
        mask.astype(float),
        method="nearest",
        bounds_error=False,
        fill_value=1.0,
    )
    return interp((yy, xx)) > 0.5


# ── 6. Quality metric ─────────────────────────────────────────────────────────

def compute_snr(data: np.ndarray) -> float:
    """
    Estimate SNR: mean of top-10% signal pixels minus mean of bottom-90%.
    Works correctly with NaN backgrounds — only real signal pixels are used.
    Returns 0.0 for empty / all-NaN arrays.
    """
    valid = data[np.isfinite(data)]
    if len(valid) < 10:
        return 0.0
    sorted_v = np.sort(valid)
    idx = int(len(sorted_v) * 0.9)
    return float(sorted_v[idx:].mean() - sorted_v[:idx].mean())


# ── 7. Synthetic field generator ──────────────────────────────────────────────

def generate_synthetic_radar_field(
    h: int = 256,
    w: int | None = None,
    seed: int | None = None,
    size: int | None = None,
) -> np.ndarray:
    """
    Generate a plausible synthetic reflectivity field for testing / training.

    Parameters
    ----------
    h, w  : output height and width (use `size` for a square patch)
    seed  : random seed for reproducibility
    size  : if given, overrides h and w with a square patch
    """
    if size is not None:
        h = w = size
    if w is None:
        w = h

    field = np.full((h, w), -15.0, dtype=np.float32)
    rng   = np.random.default_rng(seed)

    # Stratiform background
    xx, yy = np.meshgrid(np.linspace(0, 10, w), np.linspace(0, 10, h))
    field += 25.0 * (0.5 + 0.5 * np.sin(xx * 0.5) * np.cos(yy * 0.3))

    # Convective cells
    for _ in range(12):
        ry    = rng.integers(0, h)
        rx    = rng.integers(0, w)
        amp   = rng.uniform(30, 60)
        sigma = rng.uniform(min(h, w) / 50, min(h, w) / 10)
        y, x  = np.ogrid[:h, :w]
        field += amp * np.exp(-((y - ry) ** 2 + (x - rx) ** 2) / (2 * sigma ** 2))

    field += rng.normal(0, 2.0, (h, w))

    return np.clip(field, -32.0, 75.0).astype(np.float32)
