# Radar_v1 — XRAD / NEXRAD Signal Enhancement

A Python pipeline for reading, visualising, and enhancing weather radar data
using neighbourhood-based signal processing and super-resolution techniques.

---

## Project Structure

```
Radar_v1/
├── requirements.txt          # Python dependencies
├── 01_fetch_data.py          # Download a sample NEXRAD file (USA)
├── 01_fetch_aemet.py         # Fetch latest AEMET radar (Spain)
├── 01_fetch_meteocat.py      # Fetch latest Meteocat radar (Catalonia)
├── 02_visualize.py           # Read and plot radar fields
├── 03_enhance.py             # Enhancement algorithms (denoise → upscale → sharpen)
├── 04_superresolution.py     # Advanced: CNN-based super-resolution skeleton
├── web_app.py                # Interactive Streamlit Dashboard
├── utils/
│   ├── radar_io.py           # Multi-source reader (NEXRAD, AEMET, Meteocat)
│   ├── enhancement.py        # Core enhancement functions
│   └── image_radar.py        # Image-to-dBZ mapping utilities
└── data/                     # Downloaded radar files land here
```

---

## Features

- **Multi-Source Support**: 
  - **NEXRAD (USA)**: Support for Level-II binary data.
  - **AEMET (Spain)**: Live fetch via OpenData API (GIF to dBZ mapping).
  - **Meteocat (Catalonia)**: Support for XEMA API (PNG to dBZ mapping).
- **Enhancement Engine**: Adaptive NLM Denoising + Unsharp Masking.
- **Super-Resolution**: RadarResNet-v2 (SRCNN variant) for high-fidelity upscaling.
- **Interactive Dashboard**: Real-time analysis and ROI zooming.

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Dashboard
```bash
streamlit run web_app.py
```

### 3. Fetch Data (CLI)
For NEXRAD:
```bash
python 01_fetch_data.py
```
For AEMET (Live):
```bash
python 01_fetch_aemet.py --radar ba
```

---

## How the Enhancement Works

```
Raw radar signal (rays × gates)
        │
        ▼
  [1] Denoising          ← Non-Local Means (NLM) or Gaussian filter
        │
        ▼
  [2] SR Upscaling       ← RadarResNet-v2 (Residual Learning)
        │
        ▼
  [3] Sharpening         ← Adaptive Unsharp masking
        │
        ▼
Enhanced radar field
```

The pipeline preserves meteorological structure while removing noise artifacts. For image-based sources (AEMET/Meteocat), the system reverse-engineers the color palette back into numerical dBZ values before processing.

### Roadmap
- [x] Multi-source API integration (AEMET)
- [x] Residual Learning for Super-Resolution
- [ ] Attenuation correction (Z-PHI method)
- [ ] Kalman filter for temporal smoothing
- [ ] Export to GeoTIFF for GIS use
