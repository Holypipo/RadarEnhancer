import os
import sys
import numpy as np

# Add parent dir to path
sys.path.append(os.getcwd())

from utils.radar_io import read_radar, get_field_array

def test_load_aemet():
    # Find the latest aemet gif
    files = [f for f in os.listdir("data") if f.startswith("aemet") and f.endswith(".gif")]
    if not files:
        files = ["aemet_ba_test.dat"]
        
    for f in files:
        path = os.path.join("data", f)
        if not os.path.exists(path): continue
        
        print(f"\nTesting file: {path}")
        try:
            radar = read_radar(path)
            raw = get_field_array(radar, "reflectivity")
            print(f"  OK: Loaded array of shape {raw.shape}")
            print(f"  Max dBZ: {np.nanmax(raw)}")
            print(f"  Min dBZ: {np.nanmin(raw)}")
            valid_data = raw[raw > -10]
            if len(valid_data) > 0:
                print(f"  Mean dBZ (valid): {np.nanmean(valid_data)}")
            else:
                print("  No signal detected (> -10dBZ)")
        except Exception as e:
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_load_aemet()
