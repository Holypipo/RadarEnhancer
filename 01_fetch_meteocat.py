"""
01_fetch_meteocat.py
--------------------
Downloads Meteocat radar data (images) using their API.
Note: Requires a Meteocat API Key.

Usage:
    python 01_fetch_meteocat.py --key YOUR_KEY
"""

import os
import sys
import requests
import argparse
from datetime import datetime
from config import DATA_DIR, ensure_core_dirs
from utils.logging_utils import configure_logging

# ── Configuration ─────────────────────────────────────────────────────────────

BASE_URL = "https://api.meteo.cat/radar/v1/imatge"
ensure_core_dirs()

def fetch_meteocat(api_key, radar_id="composicio"):
    url = f"{BASE_URL}/{radar_id}"
    headers = {
        'X-Api-Key': api_key,
        'accept': 'application/json'
    }
    
    print(f"Contacting Meteocat API: {url}")
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        # Meteocat might return the image directly or a JSON with a URL
        content_type = response.headers.get('Content-Type', '')
        if 'application/json' in content_type:
            info = response.json()
            # Handle JSON response if applicable
            return None 
        else:
            return response.content
            
    except Exception as e:
        print(f"  ERROR: {e}")
        return None

def main():
    configure_logging("01_fetch_meteocat")

    parser = argparse.ArgumentParser(description="Fetch Meteocat Radar Data")
    parser.add_argument("--key", type=str, required=True, help="Meteocat API Key")
    parser.add_argument("--radar", type=str, default="composicio", help="Radar ID or 'composicio'")
    args = parser.parse_args()

    content = fetch_meteocat(args.key, args.radar)
    if content:
        filename = f"meteocat_{args.radar}_{datetime.now().strftime('%Y%m%d_%H%M')}.png"
        target_path = os.path.join(DATA_DIR, filename)
        with open(target_path, "wb") as f:
            f.write(content)
        print(f"\n[OK] Success! Meteocat image saved to: {target_path}")
    else:
        print("\n[ERROR] Failed to fetch data. Check your API key and permissions.")

if __name__ == "__main__":
    main()
