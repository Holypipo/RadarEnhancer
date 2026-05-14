"""
01_fetch_aemet.py
-----------------
Downloads AEMET radar data (GIF images) using their OpenData API.
Stores the data in the data/ directory.

Usage:
    python 01_fetch_aemet.py --radar ba
"""

import os
import sys
import requests
import argparse
from datetime import datetime

# ── Configuration ─────────────────────────────────────────────────────────────

API_KEY = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJiZXJuYXQubW9udGFuYUBnbWFpbC5jb20iLCJqdGkiOiIwMjQ1MDI3OC0wYTg5LTQ1NzktYjYzNS1kMDJlNWQwYWQ3OTAiLCJpc3MiOiJBRU1FVCIsImlhdCI6MTc3MzgzMTM5NiwidXNlcklkIjoiMDI0NTAyNzgtMGE4OS00NTc5LWI2MzUtZDAyZTVkMGFkNzkwIiwicm9sZSI6IiJ9.G_GtDuBRxV-3aXJFYkyZV-A9aJ473jEGYZa2uWAGgYw"
BASE_URL = "https://opendata.aemet.es/opendata/api"

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)

# ── API Logic ─────────────────────────────────────────────────────────────────

def fetch_aemet_data(endpoint_path, retries=3):
    """Generic 2-step AEMET API fetcher with retry logic."""
    url = f"{BASE_URL}{endpoint_path}"
    headers = {
        'accept': 'application/json',
        'api_key': API_KEY
    }
    
    for attempt in range(retries):
        print(f"Contacting AEMET API (Attempt {attempt+1}/{retries}): {url}")
        try:
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
            
            info = response.json()
            if info.get("estado") != 200:
                print(f"  RETRY: API returned state {info.get('estado')} - {info.get('descripcion')}")
                import time
                time.sleep(2)
                continue
                
            data_url = info.get("datos")
            if not data_url:
                print("  RETRY: No 'datos' URL in response.")
                import time
                time.sleep(2)
                continue
                
            print(f"  OK: Data URL found. Downloading...")
            data_resp = requests.get(data_url, timeout=20)
            data_resp.raise_for_status()
            
            content = data_resp.content
            if content.startswith(b"<!DOCTYPE html>") or content.startswith(b"<html"):
                print("  RETRY: API returned an HTML error page.")
                import time
                time.sleep(2)
                continue
                
            if not (content.startswith(b"GIF") or content.startswith(b"\x89PNG")):
                print(f"  RETRY: Content is not a recognized image.")
                import time
                time.sleep(2)
                continue
                
            return content
        except Exception as e:
            print(f"  RETRY ERROR: {e}")
            import time
            time.sleep(2)
            
    return None

def main():
    parser = argparse.ArgumentParser(description="Fetch AEMET Radar Data")
    parser.add_argument("--radar", type=str, default="nacional", help="Radar ID (ba, ma, va, se, za, etc. or 'nacional')")
    args = parser.parse_args()

    if args.radar == "nacional":
        endpoint = "/red/radar/nacional"
        filename = f"aemet_nacional_{datetime.now().strftime('%Y%m%d_%H%M')}.gif"
    else:
        endpoint = f"/red/radar/regional/{args.radar}"
        filename = f"aemet_{args.radar}_{datetime.now().strftime('%Y%m%d_%H%M')}.gif"

    content = fetch_aemet_data(endpoint)
    if content:
        target_path = os.path.join(DATA_DIR, filename)
        with open(target_path, "wb") as f:
            f.write(content)
        print(f"\n[OK] Success! Radar image saved to: {target_path}")
        print(f"Note: This is a GIF file. You'll need a specialized loader to process it as dBZ values.")
    else:
        print("\n[ERROR] Failed to fetch data.")

if __name__ == "__main__":
    main()
