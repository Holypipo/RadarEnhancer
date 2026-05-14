"""
04_superresolution.py
---------------------
Research-grade Radar Super-Resolution training script.
Trains a Residual Super-Resolution Network (RadarResNet) on synthetic 
radar signatures to learn high-fidelity upscaling.

Architecture: RadarResNet-v2 (Residual Blocks + PixelShuffle)
Usage:
    python 04_superresolution.py --epochs 20 --patches 500
"""

import os, sys, argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Add root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.enhancement import RadarResNet, generate_synthetic_radar_field

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(__file__), "data")
MODELS_DIR = os.path.join(DATA_DIR, "models")
PLOTS_DIR  = os.path.join(DATA_DIR, "plots")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR,  exist_ok=True)

# ── Hyperparameters ───────────────────────────────────────────────────────────
SCALE      = 2
PATCH_SIZE_HR = 64
PATCH_SIZE_LR = 32
BATCH_SIZE = 16
LR         = 1e-3

# ── Dataset ───────────────────────────────────────────────────────────────────
class RadarPatchDataset(Dataset):
    def __init__(self, lr, hr):
        self.lr = torch.from_numpy(lr).float()
        self.hr = torch.from_numpy(hr).float()
    def __len__(self): return len(self.lr)
    def __getitem__(self, i): return self.lr[i], self.hr[i]

def build_dataset(n_patches: int):
    print(f"Generating {n_patches} synthetic radar patches...")
    lr_data, hr_data = [], []
    for i in range(n_patches):
        # Generate Ground Truth (HR)
        hr = generate_synthetic_radar_field(size=PATCH_SIZE_HR)
        # Generate Input (LR) by downsampling
        lr = hr[::2, ::2]
        
        # Normalise to approx [-1, 1] range for training stability
        hr_norm = (hr - 30.0) / 40.0
        lr_norm = (lr - 30.0) / 40.0
        
        hr_data.append(hr_norm[np.newaxis])
        lr_data.append(lr_norm[np.newaxis])
        if (i+1) % 100 == 0: print(f"  {i+1}/{n_patches}")

    return np.array(lr_data), np.array(hr_data)

# ── Training ──────────────────────────────────────────────────────────────────
def train(lr_data, hr_data, epochs: int):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    ds = RadarPatchDataset(lr_data, hr_data)
    n_train = int(0.85 * len(ds))
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, len(ds)-n_train])
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE)

    model = RadarResNet(scale=SCALE).to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
    
    history = {"train": [], "val": []}

    for ep in range(1, epochs + 1):
        model.train()
        train_loss = 0
        for lrb, hrb in train_dl:
            optimizer.zero_grad()
            pred = model(lrb.to(device))
            loss = loss_fn(pred, hrb.to(device))
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(lrb)
        train_loss /= n_train

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for lrb, hrb in val_dl:
                pred = model(lrb.to(device))
                loss = loss_fn(pred, hrb.to(device))
                val_loss += loss.item() * len(lrb)
        val_loss /= (len(ds) - n_train)
        
        history["train"].append(train_loss)
        history["val"].append(val_loss)
        scheduler.step(val_loss)
        
        print(f"Epoch {ep:2d}/{epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f}")

    # Save model
    model_path = os.path.join(MODELS_DIR, "resnet_radar.pth")
    torch.save(model.state_dict(), model_path)
    print(f"\n[OK] Model saved to: {model_path}")
    
    _plot_history(history)
    return model

def _plot_history(h):
    plt.figure(figsize=(10, 5), facecolor="#0e1117")
    plt.plot(h['train'], label='Train Loss', color='#4fc3f7')
    plt.plot(h['val'], label='Val Loss', color='#ff5252')
    plt.title("RadarResNet Training Convergence", color="white")
    plt.yscale('log')
    plt.legend()
    plt.grid(alpha=0.2)
    plt.savefig(os.path.join(PLOTS_DIR, "resnet_training.png"), facecolor="#0e1117")
    plt.close()

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patches", type=int, default=500)
    args = parser.parse_args()

    lr_data, hr_data = build_dataset(args.patches)
    model = train(lr_data, hr_data, args.epochs)

    # Final visual check on real data if available
    from utils.radar_io import read_radar, get_field_array
    real_file = os.path.join(DATA_DIR, "KTLX20130520_195527_V06")
    if os.path.exists(real_file):
        try:
            radar = read_radar(real_file)
            raw = get_field_array(radar, "reflectivity", sweep=0)
            # Pick a 64x64 patch from the core
            patch = raw[116-32:116+32, 98-32:98+32]
            patch = np.nan_to_num(patch, nan=-32.0)
            
            # Predict
            model.eval()
            inp = torch.from_numpy(((patch - 30.0) / 40.0)[np.newaxis, np.newaxis]).float()
            with torch.no_grad():
                pred = model(inp).numpy()[0, 0]
            
            res = (pred * 40.0 + 30.0)
            
            fig, ax = plt.subplots(1, 2, figsize=(12, 6), facecolor="#0e1117")
            ax[0].imshow(patch, vmin=-10, vmax=70, cmap='viridis')
            ax[0].set_title("Input (LR)", color="white")
            ax[1].imshow(res, vmin=-10, vmax=70, cmap='viridis')
            ax[1].set_title("RadarResNet Output (SR)", color="white")
            plt.savefig(os.path.join(PLOTS_DIR, "resnet_real_test.png"), facecolor="#0e1117")
            print(f"[OK] Real-data verification plot saved to data/plots/resnet_real_test.png")
        except Exception as e:
            print(f"[!] Real data test failed: {e}")

if __name__ == "__main__":
    main()
