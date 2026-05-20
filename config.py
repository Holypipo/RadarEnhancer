"""
Shared configuration for the radar enhancement project.

Centralising paths and external-service settings keeps the command-line
scripts, Streamlit app, and utility modules from drifting apart.
"""

from __future__ import annotations

import os
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
LOG_DIR = ROOT_DIR / "logs"
RUNS_DIR = DATA_DIR / "runs"

FETCH_DIR = DATA_DIR / "01_fetch"
VISUALIZE_DIR = DATA_DIR / "02_visualize"
ENHANCE_DIR = DATA_DIR / "03_enhance"
RESNET_DIR = DATA_DIR / "04_ResNet"

LEGACY_MODEL_DIR = DATA_DIR / "models"
RESNET_MODEL_PATH = RESNET_DIR / "resnet_radar.pth"
LEGACY_RESNET_MODEL_PATH = LEGACY_MODEL_DIR / "resnet_radar.pth"

AEMET_BASE_URL = "https://opendata.aemet.es/opendata/api"
AEMET_DEFAULT_RADAR = os.getenv("AEMET_DEFAULT_RADAR", "nacional").strip().lower()
AEMET_API_KEY = os.getenv(
    "AEMET_API_KEY",
    "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJiZXJuYXQubW9udGFuYUBnbWFpbC5jb20iLCJqdGkiOiIwMjQ1MDI3OC0wYTg5LTQ1NzktYjYzNS1kMDJlNWQwYWQ3OTAiLCJpc3MiOiJBRU1FVCIsImlhdCI6MTc3MzgzMTM5NiwidXNlcklkIjoiMDI0NTAyNzgtMGE4OS00NTc5LWI2MzUtZDAyZTVkMGFkNzkwIiwicm9sZSI6IiJ9.G_GtDuBRxV-3aXJFYkyZV-A9aJ473jEGYZa2uWAGgYw",
)


def ensure_core_dirs() -> None:
    """Create the project output directories used by the pipeline."""
    for path in (
        DATA_DIR,
        LOG_DIR,
        RUNS_DIR,
        FETCH_DIR,
        VISUALIZE_DIR,
        ENHANCE_DIR,
        RESNET_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def get_resnet_model_path(prefer_existing: bool = True) -> Path:
    """
    Resolve the ResNet weights path.

    The documented/training path is data/04_ResNet/resnet_radar.pth. A legacy
    data/models/resnet_radar.pth fallback is kept so older checkouts keep
    working until the model is regenerated or moved.
    """
    if prefer_existing:
        if RESNET_MODEL_PATH.exists():
            return RESNET_MODEL_PATH
        if LEGACY_RESNET_MODEL_PATH.exists():
            return LEGACY_RESNET_MODEL_PATH
    return RESNET_MODEL_PATH
