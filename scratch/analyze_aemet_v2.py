import numpy as np
from PIL import Image
from collections import Counter
import os

def analyze_gif(path):
    if not os.path.exists(path):
        print(f"File not found: {path}")
        return
    img = Image.open(path).convert("RGB")
    data = np.array(img)
    pixels = data.reshape(-1, 3)
    
    # Count frequencies
    counts = Counter([tuple(p) for p in pixels])
    most_common = counts.most_common(10)
    
    print(f"File: {os.path.basename(path)}")
    print(f"Total pixels: {len(pixels)}")
    for i, (color, count) in enumerate(most_common):
        print(f" Rank {i}: Color {color} -> {count} pixels ({100*count/len(pixels):.2f}%)")

if __name__ == "__main__":
    # Test both if they exist
    analyze_gif("data/aemet_test.dat")
    # Also check if any recent aemet_*.gif exists
    for f in os.listdir("data"):
        if f.startswith("aemet") and f.endswith(".gif"):
            analyze_gif(os.path.join("data", f))
            break
