"""
01_fetch_data.py
----------------
Downloads NEXRAD Level-II radar data.

This version downloads an hourly .tar bundle from Google Cloud Public Data
and extracts the specific scan file we need.

GCS Bucket: gcp-public-data-nexrad-l2
Hour: 2013-05-20 19:00-19:59 UTC (contains the Moore EF5 tornado peak)

Usage:
    python 01_fetch_data.py
"""

import os
import sys
import urllib.request
import tarfile
import shutil
import gzip

# ── Configuration ─────────────────────────────────────────────────────────────

STATION   = "KTLX"
YEAR      = "2013"
MONTH     = "05"
DAY       = "20"
HOUR_TAR  = "NWS_NEXRAD_NXL2DP_KTLX_20130520190000_20130520195959.tar"
TARGET_FILE = "KTLX20130520_195527_V06"

# GCS Public Media Link for the TAR
TAR_URL = (
    "https://storage.googleapis.com/download/storage/v1/b/gcp-public-data-nexrad-l2/o/"
    f"2013%2F05%2F20%2FKTLX%2F{HOUR_TAR}?alt=media"
)

DATA_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
TAR_PATH   = os.path.join(DATA_DIR, HOUR_TAR)
EXTRACT_DIR = os.path.join(DATA_DIR, "extracted_temp")
FINAL_PATH = os.path.join(DATA_DIR, TARGET_FILE)

os.makedirs(DATA_DIR, exist_ok=True)

# ── Download ──────────────────────────────────────────────────────────────────

def download_tar():
    if os.path.exists(FINAL_PATH):
        print(f"Target file already exists: {FINAL_PATH}")
        return True

    print(f"Downloading hourly bundle (~100MB)...")
    print(f"URL: {TAR_URL}")
    
    try:
        req = urllib.request.Request(TAR_URL, headers={"User-Agent": "Radar_v1/1.0"})
        with urllib.request.urlopen(req, timeout=60) as resp:
            total = int(resp.headers.get("Content-Length", 0))
            downloaded = 0
            chunk = 1024 * 1024 # 1MB
            with open(TAR_PATH, "wb") as f:
                while True:
                    block = resp.read(chunk)
                    if not block:
                        break
                    f.write(block)
                    downloaded += len(block)
                    if total:
                        pct = downloaded / total * 100
                        mb  = downloaded / (1024 * 1024)
                        print(f"\r  {pct:5.1f}%  {mb:.1f} MB", end="", flush=True)
            print()
        print(f"  OK: Downloaded to {TAR_PATH}")
        return True
    except Exception as e:
        print(f"  FAIL: {e}")
        return False

# ── Extract ───────────────────────────────────────────────────────────────────

def extract_target():
    if os.path.exists(FINAL_PATH):
        return True

    print(f"Extracting {TARGET_FILE} from bundle...")
    try:
        os.makedirs(EXTRACT_DIR, exist_ok=True)
        with tarfile.open(TAR_PATH, "r") as tar:
            # Look for the target file in the tar
            members = tar.getmembers()
            target_member = next((m for m in members if TARGET_FILE in m.name), None)
            
            if not target_member:
                print(f"  FAIL: Could not find {TARGET_FILE} in the archive.")
                # List first few files to help debug
                print("  Files in archive (first 5):")
                for m in members[:5]:
                    print(f"    - {m.name}")
                return False
            
            # Extract the specific member
            tar.extract(target_member, path=EXTRACT_DIR)
            extracted_path = os.path.join(EXTRACT_DIR, target_member.name)
            
            # ── Handle GZIP compression ──
            # Many NEXRAD archives are gzipped internally.
            if not os.path.exists(extracted_path):
                 print(f"  FAIL: Extracted path not found: {extracted_path}")
                 return False

            with open(extracted_path, 'rb') as f:
                header = f.read(3)
            
            if header == b'\x1f\x8b\x08':
                print(f"  Decompressing GZIP file...")
                with gzip.open(extracted_path, 'rb') as f_in:
                    with open(FINAL_PATH, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
            else:
                shutil.move(extracted_path, FINAL_PATH)
                
            print(f"  OK: Extracted to {FINAL_PATH}")
            return True
    except Exception as e:
        print(f"  FAIL: {e}")
        return False
    finally:
        # Cleanup
        if os.path.exists(EXTRACT_DIR):
            shutil.rmtree(EXTRACT_DIR)
        if os.path.exists(TAR_PATH):
            os.remove(TAR_PATH)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if os.path.exists(FINAL_PATH):
        print(f"File ready: {FINAL_PATH}")
    else:
        if download_tar():
            if extract_target():
                print("\n[OK] Success! Data is ready for analysis.")
            else:
                print("\n[ERROR] Extraction failed.")
        else:
            print("\n[ERROR] Download failed.")

    if os.path.exists(FINAL_PATH):
        print("\nNext step: py 02_visualize.py")

if __name__ == "__main__":
    main()
