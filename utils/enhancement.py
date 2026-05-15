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
    Enhanced denoising for AEMET images.

    Uses Non-Local Means when skimage is available and the array is small
    enough to be practical (<= 512 px on the shorter side).  Falls back to
    a two-pass Gaussian for large images (760×760 NLM takes several minutes).

    Parameters
    ----------
    data : 2-D float array in dBZ
    h    : denoising strength (higher = smoother, 0.6–1.0 is a good range)
    """
    short_side = min(data.shape)

    if short_side <= 512:
        try:
            from skimage.restoration import denoise_nl_means, estimate_sigma
            valid = np.nan_to_num(data, nan=-10.0)
            d_min, d_max = -32.0, 75.0
            norm = (valid - d_min) / (d_max - d_min)
            sigma_est = float(np.mean(estimate_sigma(norm)))
            denoised  = denoise_nl_means(
                norm, h=h * sigma_est, fast_mode=True,
                patch_size=5, patch_distance=6,
            )
            return (denoised * (d_max - d_min) + d_min).astype(np.float32)
        except Exception:
            pass  # fall through to Gaussian

    # Fast fallback: two-pass Gaussian (coarse + fine)
    result = denoise_gaussian(data, sigma=1.2)
    result = denoise_gaussian(result, sigma=0.6)
    return result


# ── 2. Upscaling ─────────────────────────────────────────────────────────────

def upscale(
    data: np.ndarray,
    scale: int = 2,
    method: str = "linear",
) -> np.ndarray:
    """
    Upscale a 2-D radar field by an integer factor using grid interpolation.

    Parameters
    ----------
    data   : 2-D float array (H × W)
    scale  : integer upscaling factor  (2 → double resolution)
    method : 'linear' (smooth) or 'nearest' (crisp, for masks)
    """
    H, W     = data.shape
    nan_mask = np.isnan(data)

    y_old = np.linspace(0.0, 1.0, H)
    x_old = np.linspace(0.0, 1.0, W)
    filled = np.where(nan_mask, 0.0, data)

    interp = RegularGridInterpolator(
        (y_old, x_old), filled, method=method,
        bounds_error=False, fill_value=np.nan,
    )

    y_new = np.linspace(0.0, 1.0, H * scale)
    x_new = np.linspace(0.0, 1.0, W * scale)
    yy, xx = np.meshgrid(y_new, x_new, indexing="ij")
    upscaled = interp((yy, xx))

    # Propagate NaN mask
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
    sharpen_amount: float = 0.8,
    use_resnet: bool = True,
) -> np.ndarray:
    """
    Full AEMET radar enhancement pipeline:
      1. Denoising        (Gaussian / NLM)
      2. Super-resolution (ResNet if trained weights exist, else bicubic)
      3. Post-upscaling smooth  (removes checkerboard / grid artefacts)
      4. Peak conservatism      (prevents overestimation)
      5. Edge sharpening        (subtle, signal-only)

    Parameters
    ----------
    data           : raw dBZ array from AEMET GIF
    denoise_h      : NLM strength (ignored when Gaussian fallback is used)
    scale          : upscaling factor
    sharpen_amount : unsharp-mask boost
    use_resnet     : attempt ResNet SR if trained weights are present
    """
    # [1] Denoise
    denoised = denoise_advanced(data, h=denoise_h)

    # [2] Super-resolution
    model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "models", "resnet_radar.pth",
    )
    if use_resnet and os.path.exists(model_path):
        try:
            enhanced = _run_resnet(denoised, scale=scale)
        except Exception as e:
            print(f"  [ResNet] Inference failed ({e}). Falling back to bicubic.")
            enhanced = upscale(denoised, scale=scale)
    else:
        enhanced = upscale(denoised, scale=scale)

    # [3] Post-upsampling smoothing (removes PixelShuffle checkerboard)
    enhanced = gaussian_filter(enhanced, sigma=0.6)

    # [4] Reflectivity conservatism — don't let peak exceed original
    orig_max = np.nanmax(denoised)
    enh_max  = np.nanmax(enhanced)
    if enh_max > orig_max > 0:
        enhanced = enhanced * (orig_max / enh_max)

    # [5] Subtle edge sharpening (signal pixels only)
    signal_mask = enhanced > 5.0
    sharpened   = sharpen_unsharp_mask(enhanced, sigma=0.8, amount=sharpen_amount * 0.4)
    enhanced    = np.where(signal_mask, sharpened, enhanced)

    return enhanced


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

    model_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "models", "resnet_radar.pth",
    )
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
    return (pred * (d_max - d_min) + d_min).astype(np.float32)


# ── 6. Quality metric ─────────────────────────────────────────────────────────

def compute_snr(data: np.ndarray) -> float:
    """
    Simple SNR estimate: mean of top-10% reflectivity minus mean of bottom-90%.
    Returns 0.0 for empty / all-NaN arrays.
    """
    valid = data[~np.isnan(data)]
    if len(valid) == 0:
        return 0.0
    sorted_v = np.sort(valid)
    idx      = int(len(sorted_v) * 0.9)
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
