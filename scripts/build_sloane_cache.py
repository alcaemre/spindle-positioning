#
# Emre Alca
# University of Pennsylvania
# Created on Thu Mar 06 2026
#
# Run once to download Neil Sloane's best-known sphere packings (N=4..130)
# and save them as .npy files in data/sloane_cache/.
#
# Usage:
#   python scripts/build_sloane_cache.py
#

import urllib.request
import numpy as np
import os

SLOANE_URL = "http://neilsloane.com/packings/dim3/pack.3.{N}.txt"
CACHE_DIR  = os.path.join(os.path.dirname(__file__), '..', 'data', 'sloane_cache')
N_MIN, N_MAX = 4, 130


def download_and_cache(N, cache_dir):
    url = SLOANE_URL.format(N=N)
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            lines = resp.read().decode().strip().splitlines()
        vals = [float(l.strip()) for l in lines if l.strip()]
        pts = np.array(vals).reshape(N, 3)
        if pts.shape != (N, 3):
            print(f"  N={N}: unexpected shape {pts.shape}, skipping")
            return False
        np.save(os.path.join(cache_dir, f"sloane_{N}.npy"), pts)
        return True
    except Exception as e:
        print(f"  N={N}: failed ({e})")
        return False


if __name__ == "__main__":
    os.makedirs(CACHE_DIR, exist_ok=True)
    print(f"Saving to {os.path.abspath(CACHE_DIR)}\n")

    ok, fail = [], []
    for N in range(N_MIN, N_MAX + 1):
        success = download_and_cache(N, CACHE_DIR)
        (ok if success else fail).append(N)
        print(f"  N={N}: {'ok' if success else 'FAILED'}")

    print(f"\nDone. {len(ok)} cached, {len(fail)} failed.")
    if fail:
        print(f"Failed N values: {fail}")