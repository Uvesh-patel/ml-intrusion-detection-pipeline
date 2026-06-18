"""
Download OCPP 1.6 WebSocket IDS dataset from Zenodo.

Source: Dalamagkas et al. (2025) "Federated Detection of Open Charge Point
Protocol 1.6 Cyberattacks" - https://zenodo.org/records/14887131

Downloads the balanced CSV splits (TCP/IP and Application layers).
Uses curl because Zenodo blocks Python's requests library.

Author: Uvesh Patel
"""

import os
import subprocess
import zipfile

DATA_DIR = "data/ocpp"
os.makedirs(DATA_DIR, exist_ok=True)

# Zenodo API download URLs (balanced subsets only, ~10 MB total)
FILES = {
    "Balanced_OCPP16_TCP-IP_Layer.zip":
        "https://zenodo.org/api/records/14887131/files/Balanced_OCPP16_TCP-IP_Layer.zip/content",
    "Balanced_OCPP16_APP_Layer.zip":
        "https://zenodo.org/api/records/14887131/files/Balanced_OCPP16_APP_Layer.zip/content",
}

for fname, url in FILES.items():
    dest = os.path.join(DATA_DIR, fname)
    if os.path.exists(dest):
        print(f"  {fname} already exists, skipping download")
        continue
    print(f"  Downloading {fname}...")
    subprocess.run(["curl", "-L", "-o", dest, url], check=True)
    print(f"    Done.")

# Extract
for fname in FILES:
    zpath = os.path.join(DATA_DIR, fname)
    if not os.path.exists(zpath):
        continue
    extract_dir = os.path.join(DATA_DIR, fname.replace(".zip", ""))
    if os.path.exists(extract_dir):
        print(f"  {fname} already extracted")
        continue
    print(f"  Extracting {fname}...")
    with zipfile.ZipFile(zpath, "r") as z:
        z.extractall(DATA_DIR)
    print(f"    Done.")

print("\nContents:")
for root, dirs, fls in os.walk(DATA_DIR):
    for f in fls:
        if f.endswith(".csv"):
            path = os.path.join(root, f)
            size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"  {os.path.relpath(path, DATA_DIR)} ({size_mb:.1f} MB)")
