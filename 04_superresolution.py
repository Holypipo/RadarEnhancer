"""
04_superresolution.py
---------------------
Train RadarResNet on synthetic radar patches, then verify on a real AEMET file.

Architecture : RadarResNet-v3 (Residual Channel-Attention + PixelShuffle 2x)
Outputs      : data/04_ResNet/resnet_radar.pth
               data/04_ResNet/04_TrainingCurve_DDMMYYYY_HHMM.png
               data/04_ResNet/04_Verification_DDMMYYYY_HHMM.png

Usage:
    python 04_superresolution.py --epochs 20 --patches 500
    python 04_superresolution.py --epochs 20 --patches 500 -f 01_AemetNacional_19052026_1043.gif
"""

from __future__ import annotations

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import FETCH_DIR, RESNET_DIR, RESNET_MODEL_PATH, ensure_core_dirs
from utils.logging_utils import configure_logging
from utils.enhancement import generate_synthetic_radar_field, compute_snr
from utils.radar_io import read_radar, get_field_array
from utils.plot_utils import (
    RADAR_CMAP, DBZ_VMIN, DBZ_VMAX,
    dark_fig_axes, add_colorbar, save_fig, make_save_path,
)

# ── Paths ─────────────────────────────────────────────────────────────────────
ensure_core_dirs()
OUT_DIR    = RESNET_DIR
MODEL_PATH = RESNET_MODEL_PATH

# ── Hyperparameters ───────────────────────────────────────────────────────────
SCALE         = 2
PATCH_SIZE_HR = 64
BATCH_SIZE    = 16
LR_INIT       = 1e-3


def _RequireTorch():
    try:
        import torch
        return torch
    except ImportError:
        print(
            "\n[ERROR] PyTorch is not installed.\n"
            "Install with:\n"
            "    pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
        )
        sys.exit(1)


# ── Dataset ───────────────────────────────────────────────────────────────────

def BuildDataset(n_patches: int):
    """Generate LR/HR patch pairs from synthetic radar fields."""
    print(f"Generating {n_patches} synthetic radar patches...")
    lr_list, hr_list = [], []
    d_min, d_max = -32.0, 75.0

    for i in range(n_patches):
        hr      = generate_synthetic_radar_field(size=PATCH_SIZE_HR, seed=i)
        lr      = hr[::2, ::2]
        hr_norm = (hr - d_min) / (d_max - d_min)
        lr_norm = (lr - d_min) / (d_max - d_min)
        hr_list.append(hr_norm[np.newaxis])
        lr_list.append(lr_norm[np.newaxis])
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{n_patches}")

    return (np.array(lr_list, dtype=np.float32),
            np.array(hr_list, dtype=np.float32))


# ── Training ──────────────────────────────────────────────────────────────────

def Train(lr_data: np.ndarray, hr_data: np.ndarray, epochs: int):
    torch = _RequireTorch()
    from utils.resnet import RadarResNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}  ({epochs} epochs, {len(lr_data)} patches)")

    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(lr_data), torch.from_numpy(hr_data))
    n_train  = int(0.85 * len(dataset))
    n_val    = len(dataset) - n_train
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = torch.utils.data.DataLoader(val_ds,   batch_size=BATCH_SIZE)

    model     = RadarResNet(scale=SCALE).to(device)
    loss_fn   = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_INIT)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5)

    history = {"train": [], "val": []}

    for ep in range(1, epochs + 1):
        model.train()
        t_loss = 0.0
        for lrb, hrb in train_dl:
            optimizer.zero_grad()
            pred = model(lrb.to(device))
            loss = loss_fn(pred, hrb.to(device))
            loss.backward()
            optimizer.step()
            t_loss += loss.item() * len(lrb)
        t_loss /= n_train

        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for lrb, hrb in val_dl:
                pred    = model(lrb.to(device))
                v_loss += loss_fn(pred, hrb.to(device)).item() * len(lrb)
        v_loss /= n_val

        history["train"].append(t_loss)
        history["val"].append(v_loss)
        scheduler.step(v_loss)
        print(f"  Epoch {ep:3d}/{epochs}  train={t_loss:.6f}  val={v_loss:.6f}")

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\n[OK] Model saved -> {MODEL_PATH}")

    _PlotTrainingCurve(history)
    return model


def _PlotTrainingCurve(history: dict):
    fig, ax = plt.subplots(figsize=(10, 5), facecolor="#0e1117")
    ax.set_facecolor("#0e1117")
    ax.plot(history["train"], label="Train loss", color="#4fc3f7")
    ax.plot(history["val"],   label="Val loss",   color="#ff5252")
    ax.set_title("RadarResNet Training Convergence", color="white", fontsize=12)
    ax.set_xlabel("Epoch", color="white")
    ax.set_ylabel("MSE Loss (log scale)", color="white")
    ax.set_yscale("log")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#1a1a2e", labelcolor="white")
    ax.grid(alpha=0.2)
    save_fig(fig, make_save_path(OUT_DIR, "04", "TrainingCurve"))


# ── Verification ──────────────────────────────────────────────────────────────

def VerifyOnAemet(model, gif_path: str):
    """Run the trained model on a central patch from a real AEMET image."""
    torch = _RequireTorch()

    try:
        radar = read_radar(gif_path)
        field = get_field_array(radar, "reflectivity")
    except Exception as e:
        print(f"[!] Could not load AEMET file for verification: {e}")
        return

    H, W   = field.shape
    cy, cx = H // 2, W // 2
    half   = PATCH_SIZE_HR // 2
    patch  = field[cy - half: cy + half, cx - half: cx + half]
    patch  = np.nan_to_num(patch, nan=0.0)

    if patch.shape != (PATCH_SIZE_HR, PATCH_SIZE_HR):
        print("[!] Patch extraction failed -- image may be too small.")
        return

    lr_patch = patch[::2, ::2]
    d_min, d_max = -32.0, 75.0
    norm = (lr_patch - d_min) / (d_max - d_min)
    inp  = torch.from_numpy(norm[None, None]).float()

    model.eval()
    with torch.no_grad():
        pred = model(inp).numpy()[0, 0]
    sr_patch = pred * (d_max - d_min) + d_min

    fig, axes = dark_fig_axes(3, (15, 5))
    panels = [
        (lr_patch, f"LR Input\n{lr_patch.shape[0]}x{lr_patch.shape[1]}"),
        (patch,    f"Ground Truth (HR)\n{patch.shape[0]}x{patch.shape[1]}"),
        (sr_patch, f"ResNet SR Output\n{sr_patch.shape[0]}x{sr_patch.shape[1]}"),
    ]
    im = None
    for ax, (arr, title) in zip(axes, panels):
        im = ax.imshow(arr, vmin=DBZ_VMIN, vmax=DBZ_VMAX,
                       cmap=RADAR_CMAP, interpolation="nearest")
        ax.set_title(title, color="white", fontsize=11)
        ax.tick_params(colors="white")
    if im:
        add_colorbar(fig, im, axes[-1], label="dBZ")

    fig.suptitle(f"RadarResNet Verification -- {os.path.basename(gif_path)}",
                 color="white", fontsize=13)
    fig.tight_layout()
    save_fig(fig, make_save_path(OUT_DIR, "04", "Verification"))


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    configure_logging("04_superresolution")

    parser = argparse.ArgumentParser(description="Train RadarResNet")
    parser.add_argument("-f", type=str, default=None,
        help="AEMET GIF in data/01_fetch/ for post-training verification")
    parser.add_argument("--epochs",  type=int, default=20)
    parser.add_argument("--patches", type=int, default=500)
    args = parser.parse_args()

    lr_data, hr_data = BuildDataset(args.patches)
    model = Train(lr_data, hr_data, args.epochs)

    # Verification
    gif_path = None
    if args.f:
        gif_path = os.path.join(FETCH_DIR, args.f)
    elif os.path.isdir(FETCH_DIR):
        gifs = sorted(
            [f for f in os.listdir(FETCH_DIR) if f.endswith(".gif")],
            reverse=True,
        )
        if gifs:
            gif_path = os.path.join(FETCH_DIR, gifs[0])

    if gif_path and os.path.exists(gif_path):
        print(f"\nVerifying on: {os.path.basename(gif_path)}")
        VerifyOnAemet(model, gif_path)
    else:
        print("\n[!] No AEMET GIF found for verification. Run 01_fetch_aemet.py first.")

    print(f"\n[OK] All outputs saved to: {OUT_DIR}")
    print(f"  Model  -> {MODEL_PATH}")
    print(f"  Plots  -> {OUT_DIR}/")


if __name__ == "__main__":
    main()
