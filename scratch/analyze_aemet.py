import numpy as np
from PIL import Image
import os

def analyze_gif(path):
    img = Image.open(path)
    # Convert to RGB to get values
    img_rgb = img.convert("RGB")
    data = np.array(img_rgb)
    
    # Get unique colors
    unique_colors = np.unique(data.reshape(-1, 3), axis=0)
    print(f"Found {len(unique_colors)} unique colors in {os.path.basename(path)}")
    
    # Print some to see
    for i, color in enumerate(unique_colors):
        if i > 20: break
        print(f" Color {i}: {color}")

if __name__ == "__main__":
    analyze_gif("data/aemet_ba_test.dat")
