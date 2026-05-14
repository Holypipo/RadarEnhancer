# Technical Documentation: Radar Signal Enhancement Pipeline (v3.2)

This document outlines the multi-stage digital signal processing (DSP) and artificial intelligence (AI) pipeline used to transform raw meteorological radar feeds into high-fidelity research products.

---

## 1. Signal Ingestion & Numerical Extraction
The pipeline begins by ingesting secondary radar products (e.g., AEMET GIFs or NEXRAD Level-II volumes). For image-based sources:
- **Colorimetric Mapping**: A discrete palette matching algorithm identifies specific RGB values and maps them to their physical reflectivity equivalents (dBZ).
- **Coordinate Normalization**: The resulting 2D raster is normalized to a standard [-10, 75] dBZ range, establishing a consistent noise floor for downstream kernels.

## 2. Advanced Denoising (Non-Local Means)
To separate meteorological signals from electronic noise and atmospheric clutter, we implement the **Non-Local Means (NLM)** algorithm:
- **Spatial Redundancy**: Unlike local filters (Gaussian/Median), NLM searches for similar patches across the entire image.
- **Weighted Averaging**: Pixels with similar neighborhoods are averaged together, effectively suppressing stochastic noise while preserving the sharp gradients of convective cells.
- **Denoise Strength (H)**: A user-tunable parameter controlling the filtering aggressiveness.

## 3. AI Super-Resolution (RadarResNet-v3.2)
The core enhancement is performed by a Deep Residual Network (ResNet) optimized for meteorological signal upscaling:
- **Architecture**: The model utilizes 8 Residual Blocks with Channel Attention (CA) to prioritize significant meteorological features over background noise.
- **PixelShuffle Reconstruction**: For 2x upscaling, the model rearranges low-resolution channel features into high-resolution spatial features, avoiding the "blurring" artifacts common in bilinear interpolation.
- **Deterministic Seeding**: Global random seeds (42) are applied to the inference engine to ensure scientific reproducibility.

## 4. Reflective Conservatism & Validation
To maintain meteorological truth, the system applies a **Conservatism Guard**:
- **Intensity Clipping**: The enhanced signal is compared against the original L0 peak intensity. If the AI kernel overestimates the reflectivity, the signal is scaled back to ensure the product does not suggest non-physical weather intensity.
- **Background Restoration**: The original background mask is re-applied post-enhancement to prevent "signal bleed" into empty space.

## 5. Post-Processing & Adaptive Sharpening
Final refinement is performed via:
- **Gaussian Smoothing**: A subtle sigma=0.4 filter is applied to remove any grid-pattern artifacts introduced by the upsampling layer.
- **Unsharp Masking**: An adaptive sharpening kernel enhances the edges of storm cells, improving localized feature identification for research analysis.

---
**Version**: 3.2.0  
**Authors**: Radar Signal Processing Laboratory  
**Documentation Date**: 2026-05-14
