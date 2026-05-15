"""
01_fetch_aemet.py
-----------------
Downloads the latest AEMET radar image via their OpenData API.
Saves a .gif to data/.

Usage:
    python 01_fetch_aemet.py --radar nacional
    python 01_fetch_aemet.py --radar ba       (Barcelona)
    python 01_fetch_aemet.py --radar se       (Sevilla)
    python 01_fetch_aemet.py --radar ma       (Málaga)

Available regional codes: ba, ma, va, se, za, bi, ml, pv, an
"""

import os
import sys
import time
import requests
import argparse
from datetime import datetime

# ── Config ────────────────────────────────────────────────────────────────────

API_KEY  = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJiZXJuYXQubW9udGFuYUBnbWFpbC5jb20iLCJqdGkiOiIwMjQ1MDI3OC0wYTg5LTQ1NzktYjYzNS1kMDJlNWQwYWQ3OTAiLCJpc3MiOiJBRU1FVCIsImlhdCI6MTc3MzgzMTM5NiwidXNlcklkIjoiMDI0NTAyNzgtMGE4OS00NTc5LWI2MzUtZDAyZTVkMGFkNzkwIiwicm9sZSI6IiJ9.G_GtDuBRxV-3aXJFYkyZV-A9aJ473jEGYZa2uWAGgYw"
BASE_URL = "https://opendata.aemet.es/opendata/api"
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

os.makedirs(DATA_DIR, exist_ok=True)

# ── Fetcher ───────────────────────────────────────────────────────────────────

def fetch_aemet_image(endpoint_path: str, retries: int = 3) -> bytes | None:
    """
    2-step AEMET API fetch with retry logic.
    Step 1: metadata request → get 'datos' URL
    Step 2: download actual image bytes from 'datos' URL
    """
    url     = f"{BASE_URL}{endpoint_path}"
    headers = {"accept": "application/json", "api_key": API_KEY}

    for attempt in range(1, retries + 1):
        print(f"  Contacting AEMET API (attempt {attempt}/{retries})…")
        try:
            r = requests.get(url, headers=headers, timeout=15)
            r.raise_for_status()
            info = r.json()

            if info.get("estado") != 200:
                print(f"  API state {info.get('estado')}: {info.get('descripcion')}")
                time.sleep(2)
                continue

            data_url = info.get("datos")
            if not data_url:
                print("  No 'datos' URL in response.")
                time.sleep(2)
                continue

            print("  Downloading image…")
            img_r = requests.get(data_url, timeout=20)
            img_r.raise_for_status()
            content = img_r.content

            # Reject HTML error pages
            if content[:15].lower().startswith(b"<!doctype"):
                print("  Got an HTML error page — retrying.")
                time.sleep(2)
                continue

            # Accept GIF or PNG
            if not (content[:3] == b"GIF" or content[:4] == b"\x89PNG"):
                print(f"  Unexpected content type — retrying.")
                time.sleep(2)
                continue

            return content

        except Exception as e:
            print(f"  Error: {e}")
            time.sleep(2)

    return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fetch AEMET radar image")
    parser.add_argument(
        "--radar", type=str, default="nacional",
        help="Radar scope: 'nacional' or a regional code (ba, ma, va, se, za…)"
    )
    args = parser.parse_args()

    if args.radar == "nacional":
        endpoint = "/red/radar/nacional"
        stem     = "aemet_nacional"
    else:
        endpoint = f"/red/radar/regional/{args.radar}"
        stem     = f"aemet_{args.radar}"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    filename  = f"{stem}_{timestamp}.gif"
    out_path  = os.path.join(DATA_DIR, filename)

    print(f"\nFetching AEMET radar: {args.radar.upper()}")
    content = fetch_aemet_image(endpoint)

    if content:
        with open(out_path, "wb") as f:
            f.write(content)
        size_kb = len(content) // 1024
        print(f"\n[OK] Saved → {out_path}  ({size_kb} KB)")
        print(f"Next step: python 02_visualize.py -f {filename}")
    else:
        print("\n[ERROR] Failed to fetch radar image after all retries.")
        sys.exit(1)


if __name__ == "__main__":
    main()
