# AEMET Radar Enhancement Pipeline

4-step pipeline to fetch, visualise, enhance, and super-resolve AEMET radar data.

## Setup

```bash
pip install -r requirements.txt

# For neural super-resolution (step 4) - CPU:
pip install torch --index-url https://download.pytorch.org/whl/cpu

# For neural super-resolution (step 4) - NVIDIA GPU:
pip install torch --index-url https://download.pytorch.org/whl/cu121
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
    image_radar.py     # AEMET GIF -> dBZ conversion
    radar_io.py        # file reader (AEMET + NEXRAD)
    enhancement.py     # denoise / upscale / sharpen pipeline
    resnet.py          # RadarResNet architecture (requires torch)
    plot_utils.py      # shared colourmap, naming, figure helpers
  data/
    01_fetch/          # downloaded AEMET GIFs go here
    02_visualize/      # raw PPI and array plots
    03_enhance/        # pipeline, before/after, difference, final PPI
    04_ResNet/         # trained model weights and training plots
```

## File naming convention

All output files follow: `{StepNum}_{CamelCaseName}_{DDMMYYYY}_{HHMM}.ext`

Examples:
- `01_AemetNacional_19052026_1043.gif`
- `02_Ppi_19052026_1044.png`
- `03_BeforeAfter_19052026_1045.png`
- `04_TrainingCurve_19052026_1200.png`

Each run creates new timestamped files — old outputs are never overwritten.

## Pipeline

### Step 1 — Fetch radar image
```bash
python 01_fetch_aemet.py --radar nacional
python 01_fetch_aemet.py --radar ba     # Barcelona
python 01_fetch_aemet.py --radar se     # Sevilla
```
Saves `01_AemetNacional_DDMMYYYY_HHMM.gif` to `data/01_fetch/`.

### Step 2 — Visualise raw data
```bash
python 02_visualize.py
python 02_visualize.py -f 01_AemetNacional_19052026_1043.gif
```
Saves to `data/02_visualize/`:
- `02_Ppi_...png` — circular PPI disc
- `02_RawArray_...png` — flat 2-D dBZ array

### Step 3 — Enhance
```bash
python 03_enhance.py
python 03_enhance.py -f 01_AemetNacional_19052026_1043.gif
```
Saves to `data/03_enhance/`:
- `03_PipelineSteps_...png` — 4-panel step view
- `03_BeforeAfter_...png`   — original AEMET image vs enhanced PPI
- `03_DifferenceMap_...png` — what the algorithm contributes
- `03_EnhancedPpi_...png`   — final output (ResNet SR if trained)

### Step 4 — Train neural super-resolution
```bash
python 04_superresolution.py --epochs 20 --patches 500
```
Saves to `data/04_ResNet/`:
- `resnet_radar.pth`          — trained model weights (used by step 3)
- `04_TrainingCurve_...png`   — loss curves
- `04_Verification_...png`    — LR vs HR vs SR patch comparison

After training, re-run step 3 to use ResNet super-resolution.

## Recommended order

```bash
python 01_fetch_aemet.py
python 04_superresolution.py --epochs 30 --patches 1000   # train first
python 02_visualize.py
python 03_enhance.py                                        # uses ResNet
```
