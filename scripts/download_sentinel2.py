#!/usr/bin/env python3
"""
scripts/download_sentinel2.py
=============================
Downloads sentinel2.tar.gzaa (24.3 GB) from torchgeo/dynamic_earthnet and extracts
the monthly Sentinel-2 multi-spectral cubes.
"""

import os
import tarfile
from huggingface_hub import hf_hub_download

TARGET_DIR = os.path.abspath("data/datasets/dynamic_earthnet")
os.makedirs(TARGET_DIR, exist_ok=True)

print("Starting download of sentinel2.tar.gzaa (24.3 GB)...")
archive_path = hf_hub_download(
    repo_id="torchgeo/dynamic_earthnet",
    filename="sentinel2.tar.gzaa",
    repo_type="dataset",
    local_dir=TARGET_DIR
)
print(f"Successfully downloaded archive to: {archive_path}")

print(f"Extracting Sentinel-2 cubes to {TARGET_DIR}...")
try:
    with tarfile.open(archive_path, "r:*") as tar:
        tar.extractall(path=TARGET_DIR)
    print("Sentinel-2 extraction complete!")
except Exception as e:
    print(f"Extraction notice: {e}")
