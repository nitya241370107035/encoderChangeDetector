#!/usr/bin/env python3
"""
scripts/prepare_dynamicearthnet.py
==================================
Extracts bi-temporal multi-spectral patch pairs and semantic LULC transition
labels from DynamicEarthNet, computes Prithvi-EO-2.0-300M embeddings, and
caches feature vectors into .npz files for lightning-fast head training.
"""

import os
import sys
import csv
import json
import argparse
from typing import List, Dict, Tuple, Optional
import numpy as np
import rasterio
from rasterio.windows import Window

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)

from backend.services.prithvi_encoder import get_prithvi_encoder

# DynamicEarthNet 7 LULC Classes (Indices 0 to 6):
# 0: Soil, 1: Forest, 2: Water, 3: Agriculture, 4: Built-up, 5: Snow/Ice, 6: Wetland
LULC_CLASSES = ["Soil", "Forest", "Water", "Agriculture", "Built-up", "Snow/Ice", "Wetland"]

# Target 8 Semantic Transition Categories:
TRANSITION_CLASSES = [
    "Stable / No Change",                     # 0: C_T1 == C_T2
    "Vegetation -> Built-up / Industrial",    # 1: Forest/Agri -> Built-up
    "Bare Soil -> Built-up / Infrastructure", # 2: Soil -> Built-up
    "Deforestation / Land Clearing",          # 3: Forest -> Soil
    "Land Inundation / Flooding",             # 4: Any -> Water
    "Water Desiccation / Reclamation",        # 5: Water -> Soil/Built-up
    "Greening / Revegetation",                # 6: Soil -> Agri/Forest
    "Harvest / Agricultural Fallow"           # 7: Agri -> Soil
]

# 1-indexed rasterio band indices for Sentinel-2 -> Prithvi [B02, B03, B04, B08, B11, B12]
S2_PRITHVI_BANDS = [2, 3, 4, 8, 11, 12]


def map_transition(c1: int, c2: int) -> int:
    """Maps a (C_T1, C_T2) LULC pair to one of the 8 transition categories."""
    if c1 == c2:
        return 0  # Stable / No Change
    
    # 0: Soil, 1: Forest, 2: Water, 3: Agriculture, 4: Built-up, 5: Snow/Ice, 6: Wetland
    if c2 == 4:  # Destination is Built-up
        if c1 in [1, 3]:  # From Forest or Agriculture
            return 1
        elif c1 == 0:     # From Bare Soil
            return 2
        else:
            return 1

    if c1 == 1 and c2 == 0:  # Forest to Soil
        return 3

    if c2 == 2 and c1 != 2:  # To Water
        return 4

    if c1 == 2 and c2 != 2:  # From Water
        return 5

    if c1 == 0 and c2 in [1, 3]:  # Soil to Vegetation
        return 6

    if c1 == 3 and c2 == 0:  # Agriculture to Soil (Harvest)
        return 7

    # Default fallback: if destination is Built-up -> 1, otherwise 0
    return 0


def compute_spectral_deltas(crop_t1: np.ndarray, crop_t2: np.ndarray) -> np.ndarray:
    """
    Computes [delta_NDVI, delta_NDBI, delta_NDWI] from 6-band crops.
    crop shape: (6, H, W). Bands: 0:B02(Blue), 1:B03(Green), 2:B04(Red), 3:B08(NIR), 4:B11(SWIR1), 5:B12(SWIR2).
    """
    def get_indices(c: np.ndarray):
        b_green = c[1].astype(np.float32)
        b_red = c[2].astype(np.float32)
        b_nir = c[3].astype(np.float32)
        b_swir1 = c[4].astype(np.float32)

        eps = 1e-6
        ndvi = (b_nir - b_red) / (b_nir + b_red + eps)
        ndbi = (b_swir1 - b_nir) / (b_swir1 + b_nir + eps)
        ndwi = (b_green - b_nir) / (b_green + b_nir + eps)
        return np.nanmean(ndvi), np.nanmean(ndbi), np.nanmean(ndwi)

    ndvi1, ndbi1, ndwi1 = get_indices(crop_t1)
    ndvi2, ndbi2, ndwi2 = get_indices(crop_t2)

    return np.array([ndvi2 - ndvi1, ndbi2 - ndbi1, ndwi2 - ndwi1], dtype=np.float32)


def process_aoi_pairs(
    aoi_id: str,
    pairs_meta: List[Tuple[Dict, Dict]],
    data_dir: str,
    encoder,
    patch_size: int = 64,
    batch_size: int = 64
) -> List[Dict]:
    """Processes bi-temporal pairs for a single AOI and extracts patch features."""
    all_samples = []

    for row_t1, row_t2 in pairs_meta:
        s2_path1 = os.path.join(data_dir, row_t1["s2_path"])
        s2_path2 = os.path.join(data_dir, row_t2["s2_path"])
        lbl_path1 = os.path.join(data_dir, row_t1["label_path"])
        lbl_path2 = os.path.join(data_dir, row_t2["label_path"])

        if not (os.path.exists(s2_path1) and os.path.exists(s2_path2) and os.path.exists(lbl_path1) and os.path.exists(lbl_path2)):
            continue

        try:
            with rasterio.open(s2_path1) as ds_s2_1, rasterio.open(s2_path2) as ds_s2_2, \
                 rasterio.open(lbl_path1) as ds_lbl_1, rasterio.open(lbl_path2) as ds_lbl_2:

                H, W = ds_s2_1.shape
                patches_t1 = []
                patches_t2 = []
                meta_list = []

                # Read full scenes once into RAM (avoids 1,024 disk seek operations per pair)
                full_s2_1 = ds_s2_1.read(S2_PRITHVI_BANDS).astype(np.float32)
                full_s2_2 = ds_s2_2.read(S2_PRITHVI_BANDS).astype(np.float32)
                full_lbl_1 = ds_lbl_1.read()
                full_lbl_2 = ds_lbl_2.read()

                n_rows = H // patch_size
                n_cols = W // patch_size

                for r in range(n_rows):
                    r_start = r * patch_size
                    r_end = r_start + patch_size
                    for c in range(n_cols):
                        c_start = c * patch_size
                        c_end = c_start + patch_size

                        # Read 6 Prithvi bands from memory: (6, 64, 64)
                        c1 = full_s2_1[:, r_start:r_end, c_start:c_end]
                        c2 = full_s2_2[:, r_start:r_end, c_start:c_end]

                        # Check empty / nodata
                        if np.nanmean(c1) < 10.0 or np.nanmean(c2) < 10.0:
                            continue

                        # Read 7-band one-hot labels from memory: (7, 64, 64)
                        m1 = full_lbl_1[:, r_start:r_end, c_start:c_end]
                        m2 = full_lbl_2[:, r_start:r_end, c_start:c_end]

                        # Majority class per patch
                        c1_counts = [(m1[k] > 0).sum() for k in range(7)]
                        c2_counts = [(m2[k] > 0).sum() for k in range(7)]

                        c_t1 = int(np.argmax(c1_counts))
                        c_t2 = int(np.argmax(c2_counts))
                        y_trans = map_transition(c_t1, c_t2)

                        delta_s = compute_spectral_deltas(c1, c2)

                        patches_t1.append(c1)
                        patches_t2.append(c2)
                        meta_list.append({
                            "c_t1": c_t1,
                            "c_t2": c_t2,
                            "transition": y_trans,
                            "delta_s": delta_s
                        })

                if not patches_t1:
                    continue

                # Batch encode with Prithvi
                embs_t1 = encoder.encode_multiband_patches(patches_t1, batch_size=batch_size)
                embs_t2 = encoder.encode_multiband_patches(patches_t2, batch_size=batch_size)

                for i in range(len(patches_t1)):
                    e1 = embs_t1[i]
                    e2 = embs_t2[i]
                    delta_e = e2 - e1
                    d_s = meta_list[i]["delta_s"]

                    # Feature vector: [delta_e (1024) || e1 (1024) || delta_s (3)] = 2051-D
                    feat_vector = np.concatenate([delta_e, e1, d_s], axis=0).astype(np.float32)

                    all_samples.append({
                        "feat": feat_vector,
                        "label": meta_list[i]["transition"],
                        "from_class": meta_list[i]["c_t1"],
                        "to_class": meta_list[i]["c_t2"],
                        "aoi": aoi_id
                    })

        except Exception as e:
            print(f"Warning: error processing AOI {aoi_id} pair: {e}")
            continue

    return all_samples


def main():
    parser = argparse.ArgumentParser(description="Extract and cache DynamicEarthNet multi-spectral change features.")
    parser.add_argument("--data-dir", type=str, default="data/datasets/dynamic_earthnet", help="Path to DynamicEarthNet root.")
    parser.add_argument("--output-dir", type=str, default="data/cache", help="Output directory for cached .npz files.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for Prithvi forward passes.")
    parser.add_argument("--patch-size", type=int, default=64, help="Patch size (default: 64 px).")
    parser.add_argument("--max-aois", type=int, default=None, help="Optional maximum number of AOIs to process.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    splits_file = os.path.join(args.data_dir, "splits.csv")
    if not os.path.exists(splits_file):
        raise FileNotFoundError(f"splits.csv not found at {splits_file}")

    print("=" * 65)
    print("[EO] DYNAMICEARTHNET PREPROCESSING & PRITHVI FEATURE CACHE PIPELINE")
    print("=" * 65)
    print(f"Dataset Root:  {args.data_dir}")
    print(f"Output Cache:  {args.output_dir}")
    print(f"Patch Size:    {args.patch_size}x{args.patch_size} px")
    print(f"Batch Size:    {args.batch_size}")
    print("=" * 65 + "\n")

    # 1. Parse splits.csv and group by AOI
    aoi_rows: Dict[str, List[Dict]] = {}
    with open(splits_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            aoi_name = r["s2_path"].split("/")[1]
            if aoi_name not in aoi_rows:
                aoi_rows[aoi_name] = []
            aoi_rows[aoi_name].append(r)

    valid_aois = []
    for aoi_name, rows in aoi_rows.items():
        has_any = False
        for r in rows:
            s2_p = os.path.join(args.data_dir, r["s2_path"])
            lbl_p = os.path.join(args.data_dir, r["label_path"])
            if os.path.exists(s2_p) and os.path.exists(lbl_p):
                has_any = True
                break
        if has_any:
            valid_aois.append(aoi_name)

    all_aois = sorted(valid_aois)
    if args.max_aois:
        all_aois = all_aois[:args.max_aois]

    print(f"Discovered {len(all_aois)} available AOI scenes with Sentinel-2 and Labels.")

    # 2. Geographic Cross-AOI Train/Val Split (80% Train, 20% Val)
    np.random.seed(42)
    shuffled_aois = list(all_aois)
    np.random.shuffle(shuffled_aois)

    val_count = max(1, int(len(shuffled_aois) * 0.20))
    val_aois = set(shuffled_aois[:val_count])
    train_aois = set(shuffled_aois[val_count:])

    print(f"Split Plan: {len(train_aois)} Train AOIs | {len(val_aois)} Validation AOIs (Unseen Locations)\n")

    # 3. Build Pairs per AOI
    # For each AOI, we take:
    # Pair A: Earliest (2018-01/02) vs Latest (2019-11/12) -> Full 2-year transformation
    # Pair B: Intermediate 1-year shift (2018-06 vs 2019-06) -> Mid-cycle transition
    def get_aoi_pairs(rows: List[Dict]) -> List[Tuple[Dict, Dict]]:
        # Sort by year_month
        sorted_rows = sorted(rows, key=lambda x: x["year_month"])
        pairs = []
        if len(sorted_rows) >= 2:
            pairs.append((sorted_rows[0], sorted_rows[-1]))  # Earliest vs Latest
        if len(sorted_rows) >= 18:
            # 12-month interval pair
            mid_start = sorted_rows[len(sorted_rows) // 4]
            mid_end = sorted_rows[3 * len(sorted_rows) // 4]
            pairs.append((mid_start, mid_end))
        return pairs

    # 4. Load Prithvi Encoder
    print("Initializing Prithvi-EO-2.0-300M encoder...")
    encoder = get_prithvi_encoder()

    train_samples = []
    val_samples = []

    print("\nStarting feature extraction across all AOIs...")
    for idx, aoi in enumerate(all_aois, 1):
        rows = aoi_rows[aoi]
        pairs = get_aoi_pairs(rows)
        is_val = aoi in val_aois
        split_name = "VAL" if is_val else "TRAIN"

        print(f"[{idx:02d}/{len(all_aois):02d}] Processing AOI '{aoi}' ({split_name}) - {len(pairs)} temporal pairs...")
        samples = process_aoi_pairs(aoi, pairs, args.data_dir, encoder, args.patch_size, args.batch_size)

        if is_val:
            val_samples.extend(samples)
        else:
            train_samples.extend(samples)

        print(f"       Extracted {len(samples)} valid patches from '{aoi}' (Cumulative Train: {len(train_samples)}, Val: {len(val_samples)})")

    # 5. Pack and Export Cached Arrays
    def save_split_npz(samples: List[Dict], out_path: str):
        if not samples:
            print(f"Warning: No samples to save for {out_path}")
            return
        X = np.stack([s["feat"] for s in samples], axis=0).astype(np.float32)
        y = np.array([s["label"] for s in samples], dtype=np.int64)
        from_c = np.array([s["from_class"] for s in samples], dtype=np.int64)
        to_c = np.array([s["to_class"] for s in samples], dtype=np.int64)

        np.savez_compressed(out_path, X=X, y=y, from_class=from_c, to_class=to_c)
        print(f"Saved {len(X)} samples to '{out_path}' | X shape: {X.shape}, y shape: {y.shape}")

    print("\n" + "=" * 65)
    print("[SAVE] EXPORTING CACHED FEATURE DATASETS")
    print("=" * 65)
    train_npz = os.path.join(args.output_dir, "dynamicearthnet_train.npz")
    val_npz = os.path.join(args.output_dir, "dynamicearthnet_val.npz")

    save_split_npz(train_samples, train_npz)
    save_split_npz(val_samples, val_npz)

    # Save class mapping metadata
    class_meta_path = os.path.join(args.output_dir, "transition_classes.json")
    with open(class_meta_path, "w", encoding="utf-8") as f:
        json.dump({
            "lulc_classes": LULC_CLASSES,
            "transition_classes": TRANSITION_CLASSES,
            "feature_dim": 2051,
            "train_aois": list(train_aois),
            "val_aois": list(val_aois)
        }, f, indent=2)

    print(f"Saved class metadata to '{class_meta_path}'")
    print("=" * 65)
    print("[SUCCESS] DYNAMIC EARTHNET FEATURE CACHING COMPLETE!")
    print("=" * 65)


if __name__ == "__main__":
    main()
