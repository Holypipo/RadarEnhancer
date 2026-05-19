# AEMET Radar Enhancement Pipeline

A 4-step pipeline to fetch, visualise, enhance, and super-resolve AEMET radar data.

## Setup

```bash
pip install -r requirements.txt

# For neural super-resolution (step 4) — CPU:
pip install torch --index-url https://download.pytorch.org/whl/cpu

# For neural super-resolution (step 4) — NVIDIA GPU:
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## Pipeline

### Step 1 — Fetch radar image from AEMET
```bash
python 01_fetch_aemet.py --radar nacional
python 01_fetch_aemet.py --radar ba     # Barcelona
python 01_fetch_aemet.py --radar se     # Sevilla
```
Saves a `.gif` to `data/`.

### Step 2 — Visualise raw data
```bash
python 02_visualize.py
python 02_visualize.py -f aemet_nacional_20260515_1146.gif
```
Saves to `data/plots/`:
- `aemet_ppi.png` — circular PPI disc
- `aemet_raw_array.png` — flat 2-D array view

### Step 3 — Enhance
```bash
python 03_enhance.py
python 03_enhance.py -f aemet_nacional_20260515_1146.gif
```
Saves to `data/plots/`:
- `enhance_pipeline_steps.png` — 4-panel step view
- `enhance_before_after.png` — before / after comparison
- `enhance_difference_map.png` — what the algorithm contributes
- `enhanced_final_ppi.png` — **main output** (uses ResNet if trained)

> If you run step 4 first to train the model, re-running step 3 will
> automatically use neural super-resolution instead of bicubic.

### Step 4 — Train neural super-resolution model
```bash
python 04_superresolution.py --epochs 20 --patches 500
```
Saves to `data/models/resnet_radar.pth` and `data/plots/resnet_training.png`.

After training, re-run step 3 to get ResNet-enhanced output.

## Recommended order for best results

```bash
python 01_fetch_aemet.py
python 04_superresolution.py --epochs 30 --patches 1000   # train first
python 02_visualize.py
python 03_enhance.py                                        # now uses ResNet
```

## File structure

```
RadarEnhancer/
  01_fetch_aemet.py
  02_visualize.py
  03_enhance.py
  04_superresolution.py
  requirements.txt
  utils/
    __init__.py
    image_radar.py      # GIF -> dBZ conversion
    radar_io.py         # file reading
    enhancement.py      # denoise / upscale / sharpen pipeline
    resnet.py           # RadarResNet architecture (requires torch)
  data/
    *.gif               # downloaded AEMET images (you create these)
    plots/              # all output plots saved here
    models/             # trained ResNet weights saved here
```
