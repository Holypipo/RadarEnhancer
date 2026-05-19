"""
01_fetch_aemet.py
-----------------
Downloads the latest AEMET radar image via their OpenData API.
Saves to data/01_fetch/ with timestamped CamelCase filename.

Usage:
    python 01_fetch_aemet.py --radar nacional
    python 01_fetch_aemet.py --radar ba       (Barcelona)
    python 01_fetch_aemet.py --radar se       (Sevilla)
    python 01_fetch_aemet.py --radar ma       (Malaga)

Available regional codes: ba, ma, va, se, za, bi, ml, pv, an
"""

import os
import sys
import time
import requests
import argparse
from datetime import datetime
from config import AEMET_API_KEY, AEMET_BASE_URL, FETCH_DIR, ensure_core_dirs
from utils.logging_utils import configure_logging

# ── Config ────────────────────────────────────────────────────────────────────

ensure_core_dirs()
OUT_DIR = FETCH_DIR

# ── Fetcher ───────────────────────────────────────────────────────────────────

def FetchAemetImage(endpoint_path: str, retries: int = 3) -> bytes | None:
    """
    2-step AEMET API fetch with retry logic.
    Step 1: metadata request -> get 'datos' URL.
    Step 2: download image bytes from 'datos' URL.
    """
    url     = f"{AEMET_BASE_URL}{endpoint_path}"
    headers = {"accept": "application/json", "api_key": AEMET_API_KEY}

    for attempt in range(1, retries + 1):
        print(f"  Contacting AEMET API (attempt {attempt}/{retries})...")
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

            print("  Downloading image...")
            img_r = requests.get(data_url, timeout=20)
            img_r.raise_for_status()
            content = img_r.content

            if content[:15].lower().startswith(b"<!doctype"):
                print("  Got an HTML error page -- retrying.")
                time.sleep(2)
                continue

            if not (content[:3] == b"GIF" or content[:4] == b"\x89PNG"):
                print("  Unexpected content type -- retrying.")
                time.sleep(2)
                continue

            return content

        except Exception as e:
            print(f"  Error: {e}")
            time.sleep(2)

    return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    configure_logging("01_fetch_aemet")

    parser = argparse.ArgumentParser(description="Fetch AEMET radar image")
    parser.add_argument(
        "--radar", type=str, default="nacional",
        help="Radar scope: 'nacional' or a regional code (ba, ma, va, se, za...)"
    )
    args = parser.parse_args()

    if args.radar == "nacional":
        endpoint = "/red/radar/nacional"
        scope    = "Nacional"
    else:
        endpoint = f"/red/radar/regional/{args.radar}"
        scope    = args.radar.upper()

    # Filename: 01_AemetNacional_19052026_1043.gif
    ts       = datetime.now().strftime("%d%m%Y_%H%M")
    stem     = f"Aemet{scope.capitalize()}"
    filename = f"01_{stem}_{ts}.gif"
    out_path = os.path.join(OUT_DIR, filename)

    print(f"\nFetching AEMET radar: {scope}")
    content = FetchAemetImage(endpoint)

    if content:
        with open(out_path, "wb") as f:
            f.write(content)
        size_kb = len(content) // 1024
        print(f"\n[OK] Saved -> {out_path}  ({size_kb} KB)")
        print(f"Next step: python 02_visualize.py -f {filename}")
    else:
        print("\n[ERROR] Failed to fetch radar image after all retries.")
        sys.exit(1)


if __name__ == "__main__":
    main()
