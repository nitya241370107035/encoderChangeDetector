#!/usr/bin/env python3
"""
scripts/patch_change_detector.py
================================
Prototype script for Patch-wise Semantic Change Detection using RemoteCLIP.

Workflow:
1. Load Before and After images (and corresponding QA masks if available).
2. Divide images into spatial grid patches (e.g. 4x4 or 8x8).
3. Perform per-patch Quality Check (cloud/bad-pixel fraction from mask).
4. Compute 512-dim RemoteCLIP embeddings for all clear/valid patches.
5. Calculate semantic distance (1 - cosine_similarity), robust null statistics (median & MAD), and z-scores.
6. Extract and rank Change Candidate patches.
7. Generate side-by-side composite visualization (OpenCV/PIL) and JSON report.
"""

import os
import sys
import json
import argparse
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import rasterio
except ImportError:
    rasterio = None

from backend.services.encoder import RemoteCLIPEncoder


def load_image_and_mask(img_path: str, mask_path: str = None):
    """Loads RGB image as PIL and mask as numpy array if present."""
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image not found: {img_path}")
    
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    
    mask = None
    if mask_path and os.path.exists(mask_path):
        if rasterio:
            with rasterio.open(mask_path) as ds:
                mask = ds.read(1)
        else:
            mask = np.array(Image.open(mask_path))
        
        # Ensure mask dimensions match image
        if mask.shape != (h, w):
            mask_img = Image.fromarray(mask).resize((w, h), Image.NEAREST)
            mask = np.array(mask_img)
            
    return img, mask


def analyze_patches(
    before_img_path: str,
    after_img_path: str,
    before_mask_path: str = None,
    after_mask_path: str = None,
    grid_size: int = 8,
    quality_thresh: float = 0.20,
    z_threshold: float = 2.0,
    output_dir: str = "data/runs/patch_change_demo"
):
    print(f"\n=======================================================")
    print(f"🛰️  PATCH-WISE SEMANTIC CHANGE DETECTION PROTOTYPE")
    print(f"=======================================================")
    print(f"Before Image: {before_img_path}")
    print(f"After Image:  {after_img_path}")
    print(f"Grid Layout:  {grid_size}x{grid_size} ({grid_size*grid_size} total patches)")
    print(f"Quality Max Bad Fraction: {quality_thresh*100:.0f}%")
    print(f"Change z-score Threshold: >= {z_threshold}")
    print(f"=======================================================\n")

    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Data
    im_before, mask_before = load_image_and_mask(before_img_path, before_mask_path)
    im_after, mask_after = load_image_and_mask(after_img_path, after_mask_path)

    W, H = im_before.size
    patch_w = W // grid_size
    patch_h = H // grid_size
    print(f"Tile Dimensions: {W}x{H} | Patch Dimensions: {patch_w}x{patch_h} px")

    # 2. Patch Extraction and Quality Check
    patches_meta = []
    crops_before = []
    crops_after = []
    valid_indices = []

    print("\n[Stage 1] Dividing into patches and executing Quality Check...")
    for r in range(grid_size):
        for c in range(grid_size):
            idx = r * grid_size + c
            x0 = c * patch_w
            y0 = r * patch_h
            x1 = x0 + patch_w
            y1 = y0 + patch_h

            crop_b = im_before.crop((x0, y0, x1, y1))
            crop_a = im_after.crop((x0, y0, x1, y1))

            # Quality Check
            bad_frac_b = 0.0
            bad_frac_a = 0.0
            if mask_before is not None:
                patch_m_b = mask_before[y0:y1, x0:x1]
                bad_frac_b = float((patch_m_b > 0).mean())
            if mask_after is not None:
                patch_m_a = mask_after[y0:y1, x0:x1]
                bad_frac_a = float((patch_m_a > 0).mean())

            # Check nodata / pure black
            arr_b = np.array(crop_b)
            arr_a = np.array(crop_a)
            is_black_b = (arr_b.mean() < 1.0)
            is_black_a = (arr_a.mean() < 1.0)

            is_valid = True
            drop_reason = None
            if bad_frac_b > quality_thresh or bad_frac_a > quality_thresh:
                is_valid = False
                drop_reason = f"Cloud/Bad pixel (T1: {bad_frac_b:.1%}, T2: {bad_frac_a:.1%})"
            elif is_black_b or is_black_a:
                is_valid = False
                drop_reason = "NoData / Black boundary"

            patch_info = {
                "patch_id": idx,
                "row": r,
                "col": c,
                "bbox": [x0, y0, x1, y1],
                "is_valid": is_valid,
                "drop_reason": drop_reason,
                "bad_frac_t1": round(bad_frac_b, 4),
                "bad_frac_t2": round(bad_frac_a, 4),
                "distance": 0.0,
                "z_score": 0.0,
                "is_candidate": False
            }
            patches_meta.append(patch_info)

            if is_valid:
                valid_indices.append(idx)
                crops_before.append(crop_b)
                crops_after.append(crop_a)

    print(f"Total Patches: {len(patches_meta)} | Valid/Clear: {len(valid_indices)} | Dropped: {len(patches_meta) - len(valid_indices)}")

    # 3. Embedding Generation
    print("\n[Stage 2] Computing RemoteCLIP embeddings for clear patches...")
    encoder = RemoteCLIPEncoder()

    if valid_indices:
        print(f"Encoding {len(crops_before)} Before crops...")
        vecs_b = np.array(encoder.encode_image(crops_before), dtype=np.float32)
        print(f"Encoding {len(crops_after)} After crops...")
        vecs_a = np.array(encoder.encode_image(crops_after), dtype=np.float32)

        # 4. Semantic Distance Calculation (Cosine distance: 1.0 - dot_product)
        # vecs are already L2 normalized by RemoteCLIPEncoder
        sims = np.sum(vecs_b * vecs_a, axis=1)
        sims = np.clip(sims, -1.0, 1.0)
        distances = 1.0 - sims

        # Assign back to patch meta
        for i, val_idx in enumerate(valid_indices):
            patches_meta[val_idx]["distance"] = float(distances[i])

        # 5. Robust Statistics (Median and MAD) - Netsight-style Null Model
        med_dist = float(np.median(distances))
        abs_dev = np.abs(distances - med_dist)
        mad = float(np.median(abs_dev))
        mad = max(mad, 1e-4) # prevent division by zero

        print(f"\n[Stage 3] Robust Spatial Distance Distribution:")
        print(f"  • Min Distance:    {float(np.min(distances)):.4f}")
        print(f"  • Median Distance: {med_dist:.4f}")
        print(f"  • MAD:             {mad:.4f}")
        print(f"  • Max Distance:    {float(np.max(distances)):.4f}")

        candidates = []
        for i, val_idx in enumerate(valid_indices):
            d = patches_meta[val_idx]["distance"]
            z = (d - med_dist) / mad
            patches_meta[val_idx]["z_score"] = float(round(z, 2))
            if z >= z_threshold:
                patches_meta[val_idx]["is_candidate"] = True
                candidates.append(patches_meta[val_idx])

        # Sort candidates descending by distance
        candidates.sort(key=lambda x: -x["distance"])
        print(f"\n[Stage 4] Detected {len(candidates)} Change Candidate Patches (z >= {z_threshold}):")
        for rank, cand in enumerate(candidates[:10], 1):
            print(f"  Rank #{rank:02d} | Patch [{cand['row']}, {cand['col']}] (BBox: {cand['bbox']}) | Dist: {cand['distance']:.4f} | z-score: {cand['z_score']:+.2f}")
    else:
        print("Warning: No valid patches found after quality filtering!")
        candidates = []
        med_dist, mad = 0.0, 1.0

    # 6. Generate Comprehensive Visual Report using OpenCV & PIL
    print("\n[Stage 5] Generating 4-panel visual analysis panel...")
    
    # Base panels
    p1 = np.array(im_before).copy()
    p2 = np.array(im_after).copy()
    p4 = np.array(im_after).copy()

    # Draw grid lines on p1, p2, p4
    for p in [p1, p2, p4]:
        for i in range(1, grid_size):
            cv2.line(p, (i * patch_w, 0), (i * patch_w, H), (56, 189, 248), 1, cv2.LINE_AA)
            cv2.line(p, (0, i * patch_h), (W, i * patch_h), (56, 189, 248), 1, cv2.LINE_AA)

    # Panel 3: Semantic Heatmap
    # Normalize distances to 0..255 for colormap
    min_d = float(np.min(distances)) if valid_indices else 0.0
    max_d = float(np.max(distances)) if valid_indices else 1.0
    d_range = max(max_d - min_d, 1e-4)

    dist_grid_norm = np.zeros((grid_size, grid_size), dtype=np.uint8)
    for p in patches_meta:
        r, c = p["row"], p["col"]
        if p["is_valid"]:
            norm_val = int(255.0 * (p["distance"] - min_d) / d_range)
            dist_grid_norm[r, c] = np.clip(norm_val, 0, 255)
        else:
            dist_grid_norm[r, c] = 0

    # Resize colormap to full tile resolution
    heat_small = cv2.applyColorMap(dist_grid_norm, cv2.COLORMAP_MAGMA)
    # Convert BGR from cv2 to RGB
    heat_small = cv2.cvtColor(heat_small, cv2.COLOR_BGR2RGB)
    p3 = cv2.resize(heat_small, (W, H), interpolation=cv2.INTER_NEAREST)

    # Darken dropped patches in heatmap
    for p in patches_meta:
        if not p["is_valid"]:
            x0, y0, x1, y1 = p["bbox"]
            p3[y0:y1, x0:x1] = (50, 50, 50)
            cv2.line(p3, (x0, y0), (x1, y1), (100, 100, 100), 1)
            cv2.line(p3, (x0, y1), (x1, y0), (100, 100, 100), 1)

    # Draw grid lines on heatmap
    for i in range(1, grid_size):
        cv2.line(p3, (i * patch_w, 0), (i * patch_w, H), (30, 41, 59), 1)
        cv2.line(p3, (0, i * patch_h), (W, i * patch_h), (30, 41, 59), 1)

    # Highlight Candidates on Panel 4
    for rank, cand in enumerate(candidates, 1):
        x0, y0, x1, y1 = cand["bbox"]
        color = (239, 68, 68) if rank <= 3 else (245, 158, 11) # Red for top-3, Amber for others
        cv2.rectangle(p4, (x0, y0), (x1, y1), color, 2)
        # Put rank text
        label = f"#{rank} z={cand['z_score']:+.1f}"
        cv2.rectangle(p4, (x0, y0), (x0 + 75, y0 + 16), color, -1)
        cv2.putText(p4, label, (x0 + 2, y0 + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

    # Build side-by-side canvas with headers: 4 panels side-by-side
    header_h = 50
    spacing = 15
    canvas_w = (W * 4) + (spacing * 5)
    canvas_h = H + header_h + 30
    canvas = np.full((canvas_h, canvas_w, 3), 15, dtype=np.uint8) # dark slate #0f172a

    # Extract date labels dynamically
    import re
    def extract_date_label(path_str):
        # Match YYYY-MM-DD or YYYYMMDD in path/filename
        m1 = re.search(r"(\d{4}-\d{2}-\d{2})", path_str)
        if m1:
            return m1.group(1)
        m2 = re.search(r"(\d{4})(\d{2})(\d{2})", os.path.basename(path_str))
        if m2:
            return f"{m2.group(1)}-{m2.group(2)}-{m2.group(3)}"
        return os.path.basename(path_str)

    date_b = extract_date_label(before_img_path)
    date_a = extract_date_label(after_img_path)

    titles = [
        f"T1 (Before: {date_b})",
        f"T2 (After: {date_a})",
        f"Semantic Distance (Med:{med_dist:.2f}, MAD:{mad:.2f})",
        f"Change Candidates ({len(candidates)} flagged)"
    ]

    for idx, (p_img, title) in enumerate(zip([p1, p2, p3, p4], titles)):
        x_offset = spacing + idx * (W + spacing)
        y_offset = header_h
        canvas[y_offset:y_offset + H, x_offset:x_offset + W] = p_img
        # Header text
        cv2.putText(canvas, title, (x_offset, header_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (248, 250, 252), 1, cv2.LINE_AA)

    plot_out = os.path.join(output_dir, "patch_change_analysis.png")
    Image.fromarray(canvas).save(plot_out)
    print(f"Visual Analysis Plot saved to: {plot_out}")

    # 7. Save JSON Result
    result_data = {
        "before_image": before_img_path,
        "after_image": after_img_path,
        "grid_size": grid_size,
        "patch_pixels": [patch_w, patch_h],
        "total_patches": len(patches_meta),
        "valid_patches": len(valid_indices),
        "dropped_patches": len(patches_meta) - len(valid_indices),
        "robust_stats": {
            "median_distance": round(med_dist, 4),
            "mad": round(mad, 4),
            "z_threshold": z_threshold
        },
        "num_candidates": len(candidates),
        "candidates": candidates,
        "all_patches": patches_meta
    }

    json_out = os.path.join(output_dir, "patch_change_results.json")
    with open(json_out, "w") as f:
        json.dump(result_data, f, indent=2)
    print(f"Detailed JSON results saved to: {json_out}")
    print(f"=======================================================\n")
    return result_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patch-wise semantic change detection prototype")
    parser.add_argument(
        "--before",
        default="data/tiles/giga_berlin/2021-02-22/S2A_33UVU_20210222_2_L2A_tile_00002_thumb.jpg",
        help="Path to Before image"
    )
    parser.add_argument(
        "--after",
        default="data/tiles/giga_berlin/2023-09-10/S2A_33UVU_20230910_1_L2A_tile_00002_thumb.jpg",
        help="Path to After image"
    )
    parser.add_argument(
        "--before-mask",
        default="data/tiles/giga_berlin/2021-02-22/S2A_33UVU_20210222_2_L2A_tile_00002_mask.tif",
        help="Path to Before mask"
    )
    parser.add_argument(
        "--after-mask",
        default="data/tiles/giga_berlin/2023-09-10/S2A_33UVU_20230910_1_L2A_tile_00002_mask.tif",
        help="Path to After mask"
    )
    parser.add_argument("--grid", type=int, default=8, help="Grid size (e.g. 4 for 4x4, 8 for 8x8)")
    parser.add_argument("--quality-thresh", type=float, default=0.20, help="Max bad pixel fraction per patch")
    parser.add_argument("--z-thresh", type=float, default=2.0, help="z-score threshold for change candidates")
    parser.add_argument("--out-dir", default="data/runs/patch_change_demo", help="Output directory")

    args = parser.parse_args()

    analyze_patches(
        before_img_path=args.before,
        after_img_path=args.after,
        before_mask_path=args.before_mask,
        after_mask_path=args.after_mask,
        grid_size=args.grid,
        quality_thresh=args.quality_thresh,
        z_threshold=args.z_thresh,
        output_dir=args.out_dir
    )
