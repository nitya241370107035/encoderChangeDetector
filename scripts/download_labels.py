#!/usr/bin/env python3
"""
scripts/download_labels.py
==========================
Downloads labels.tar.gz (1.32 GB) from torchgeo/dynamic_earthnet and extracts
the monthly 7-class semantic LULC ground truth raster masks.
"""

import os
import tarfile
from huggingface_hub import hf_hub_download

TARGET_DIR = os.path.abspath("data/datasets/dynamic_earthnet")
os.makedirs(TARGET_DIR, exist_ok=True)

print("Starting download of labels.tar.gz (1.32 GB)...")
archive_path = hf_hub_download(
    repo_id="torchgeo/dynamic_earthnet",
    filename="labels.tar.gz",
    repo_type="dataset",
    local_dir=TARGET_DIR
)
print(f"Successfully downloaded archive to: {archive_path}")

print(f"Extracting labels to {TARGET_DIR}...")
with tarfile.open(archive_path, "r:gz") as tar:
    tar.extractall(path=TARGET_DIR)

print("Labels extraction complete!")
