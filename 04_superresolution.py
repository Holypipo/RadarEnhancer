"""
04_superresolution.py
---------------------
Train RadarResNet on synthetic radar patches, then verify on a real AEMET file.

Architecture : RadarResNet-v3 (Residual Channel-Attention + PixelShuffle)
Outputs      : data/models/resnet_radar.pth   — trained weights
               data/plots/resnet_training.png  — loss curves
               data/plots/resnet_verification.png — before/after on real data

Usage:
    python 04_superresolution.py --epochs 20 --patches 500
    python 04_superresolution.py --epochs 20 --patches 500 -f aemet_nacional_20260515_1115.gif
"""

from __future__ import annotations

import os
import sys
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# ── Project imports ───────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.enhancement import generate_synthetic_radar_field, compute_snr
from utils.radar_io import read_radar, get_field_array

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
MODELS_DIR = os.path.join(DATA_DIR, "models")
PLOTS_DIR  = os.path.join(DATA_DIR, "plots")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────
SCALE         = 2
PATCH_SIZE_HR = 64      # HR patch side (pixels)
PATCH_SIZE_LR = PATCH_SIZE_HR // SCALE
BATCH_SIZE    = 16
LR_INIT       = 1e-3


# ── Check torch availability ──────────────────────────────────────────────────

def _require_torch():
    try:
        import torch
        return torch
    except ImportError:
        print(
            "\n[ERROR] PyTorch is not installed.\n"
            "Install it with:\n"
            "    pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
            "Then re-run this script."
        )
        sys.exit(1)


# ── Dataset builder ───────────────────────────────────────────────────────────

def build_dataset(n_patches: int):
    """Generate LR/HR patch pairs from synthetic radar fields."""
    print(f"Generating {n_patches} synthetic radar patches…")
    lr_list, hr_list = [], []

    for i in range(n_patches):
        hr = generate_synthetic_radar_field(size=PATCH_SIZE_HR, seed=i)
        lr = hr[::2, ::2]                           # simple 2× downscale

        # Normalise to [0, 1] for training stability
        d_min, d_max = -32.0, 75.0
        hr_norm = (hr - d_min) / (d_max - d_min)
        lr_norm = (lr - d_min) / (d_max - d_min)

        hr_list.append(hr_norm[np.newaxis])         # (1, H, W)
        lr_list.append(lr_norm[np.newaxis])

        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{n_patches}")

    return np.array(lr_list, dtype=np.float32), np.array(hr_list, dtype=np.float32)


# ── Training ──────────────────────────────────────────────────────────────────

def train(lr_data: np.ndarray, hr_data: np.ndarray, epochs: int) -> object:
    torch = _require_torch()
    from utils.resnet import RadarResNet

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}  ({epochs} epochs, {len(lr_data)} patches)")

    # Dataset / DataLoader
    dataset = torch.utils.data.TensorDataset(
        torch.from_numpy(lr_data),
        torch.from_numpy(hr_data),
    )
    n_train   = int(0.85 * len(dataset))
    n_val     = len(dataset) - n_train
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val])
    train_dl = torch.utils.data.DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = torch.utils.data.DataLoader(val_ds,   batch_size=BATCH_SIZE)

    model     = RadarResNet(scale=SCALE).to(device)
    loss_fn   = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR_INIT)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, factor=0.5
    )

    history = {"train": [], "val": []}

    for ep in range(1, epochs + 1):
        # ── Train ──
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

        # ── Validate ──
        model.eval()
        v_loss = 0.0
        with torch.no_grad():
            for lrb, hrb in val_dl:
                pred   = model(lrb.to(device))
                v_loss += loss_fn(pred, hrb.to(device)).item() * len(lrb)
        v_loss /= n_val

        history["train"].append(t_loss)
        history["val"].append(v_loss)
        scheduler.step(v_loss)

        print(f"  Epoch {ep:3d}/{epochs}  train={t_loss:.6f}  val={v_loss:.6f}")

    # ── Save weights ──
    model_path = os.path.join(MODELS_DIR, "resnet_radar.pth")
    torch.save(model.state_dict(), model_path)
    print(f"\n[OK] Model saved → {model_path}")

    _plot_training(history)
    return model


def _plot_training(history: dict):
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
    path = os.path.join(PLOTS_DIR, "resnet_training.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  Saved → {path}")


# ── Verification on real AEMET data ──────────────────────────────────────────

def verify_on_aemet(model, gif_path: str):
    """Run the trained model on a patch from a real AEMET image and save a plot."""
    torch = _require_torch()

    try:
        radar = read_radar(gif_path)
        field = get_field_array(radar, "reflectivity")
    except Exception as e:
        print(f"[!] Could not load AEMET file for verification: {e}")
        return

    H, W = field.shape
    # Pick a central patch
    cy, cx = H // 2, W // 2
    half   = PATCH_SIZE_HR // 2
    patch  = field[cy - half: cy + half, cx - half: cx + half]
    patch  = np.nan_to_num(patch, nan=-10.0)

    if patch.shape != (PATCH_SIZE_HR, PATCH_SIZE_HR):
        print("[!] Patch extraction failed — image may be too small.")
        return

    # Downscale to simulate LR input
    lr_patch = patch[::2, ::2]

    d_min, d_max = -32.0, 75.0
    norm = (lr_patch - d_min) / (d_max - d_min)
    inp  = torch.from_numpy(norm[None, None]).float()

    model.eval()
    with torch.no_grad():
        pred = model(inp).numpy()[0, 0]

    sr_patch = pred * (d_max - d_min) + d_min

    # Plot
    _REF_COLORS = [
        "#646464", "#04e9e7", "#019ff4", "#0300f4",
        "#02fd02", "#01c501", "#008e00", "#fdf802",
        "#e5bc00", "#fd9500", "#fd0000", "#d40000",
        "#bc0000", "#f800fd", "#9854c6",
    ]
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "nws_ref", list(zip(np.linspace(0, 1, len(_REF_COLORS)), _REF_COLORS))
    )

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor="#0d0d0d")
    titles = [
        f"LR Input\n{lr_patch.shape[0]}×{lr_patch.shape[1]}",
        f"Ground Truth (HR)\n{patch.shape[0]}×{patch.shape[1]}",
        f"ResNet SR Output\n{sr_patch.shape[0]}×{sr_patch.shape[1]}",
    ]
    arrays = [lr_patch, patch, sr_patch]

    im = None
    for ax, arr, title in zip(axes, arrays, titles):
        ax.set_facecolor("#0d0d0d")
        im = ax.imshow(arr, vmin=-10, vmax=75, cmap=cmap, interpolation="nearest")
        ax.set_title(title, color="white", fontsize=11)
        ax.tick_params(colors="white")

    if im is not None:
        cbar = fig.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04, shrink=0.92)
        cbar.set_label("dBZ", color="white")
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

    fig.suptitle(
        f"RadarResNet Verification — {os.path.basename(gif_path)}",
        color="white", fontsize=13,
    )
    fig.tight_layout()
    path = os.path.join(PLOTS_DIR, "resnet_verification.png")
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    size_kb = os.path.getsize(path) // 1024
    print(f"  Saved → {path}  ({size_kb} KB)")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train RadarResNet on AEMET data")
    parser.add_argument(
        "-f", type=str, default=None,
        help="AEMET GIF filename in data/ for post-training verification"
    )
    parser.add_argument("--epochs",  type=int, default=20)
    parser.add_argument("--patches", type=int, default=500)
    args = parser.parse_args()

    # Build dataset and train
    lr_data, hr_data = build_dataset(args.patches)
    model = train(lr_data, hr_data, args.epochs)

    # Verification: use -f if given, otherwise auto-discover newest AEMET gif
    gif_path = None
    if args.f:
        gif_path = os.path.join(DATA_DIR, args.f)
    else:
        gifs = sorted(
            [f for f in os.listdir(DATA_DIR) if f.startswith("aemet_") and f.endswith(".gif")],
            reverse=True,
        )
        if gifs:
            gif_path = os.path.join(DATA_DIR, gifs[0])

    if gif_path and os.path.exists(gif_path):
        print(f"\nVerifying on real AEMET data: {os.path.basename(gif_path)}")
        verify_on_aemet(model, gif_path)
    else:
        print("\n[!] No AEMET GIF found for verification. Run 01_fetch_aemet.py first.")

    print(f"\n[OK] All outputs saved.")
    print(f"  Model  → {MODELS_DIR}/resnet_radar.pth")
    print(f"  Plots  → {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
