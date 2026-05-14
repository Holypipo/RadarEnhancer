import numpy as np
import sys
import os

# Add root directory to path
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root)

from utils.enhancement import enhance_field

def test_enhancement():
    # Create a synthetic field with a storm cell (40 dBZ)
    data = np.full((128, 128), -10.0, dtype=np.float32)
    data[60:70, 60:70] = 40.0
    
    print(f"Input Max: {np.nanmax(data)}")
    
    enhanced = enhance_field(data, denoise_h=0.6, scale=2, sharpen_amount=1.0, use_resnet=True)
    
    print(f"Output Shape: {enhanced.shape}")
    print(f"Output Max: {np.nanmax(enhanced)}")
    print(f"Output Min: {np.nanmin(enhanced)}")
    
    # Check if signal survived
    if np.nanmax(enhanced) < 0:
        print("FAIL: Signal was lost!")
    else:
        print("SUCCESS: Signal preserved.")

if __name__ == "__main__":
    test_enhancement()
