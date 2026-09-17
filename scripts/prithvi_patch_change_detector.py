#!/usr/bin/env python3
"""
scripts/prithvi_patch_change_detector.py
========================================
Patch-wise Multi-Spectral Semantic Change Detection using Prithvi-EO-2.0-300M.

Workflow:
1. Load Before and After multi-band GeoTIFFs (.tif) and QA masks.
2. Divide multi-spectral imagery into an NxN spatial grid (e.g. 4x4 or 8x8).
3. Perform per-patch Quality Check (cloud/shadow bad fraction from mask).
4. Extract 1024-dim Prithvi-EO-2.0-300M embeddings for all valid multi-band patches.
5. Compute cosine distance (1 - cosine_similarity), robust null statistics (Median & MAD), and z-scores.
6. Extract and rank Change Candidate patches (anomalies with z >= z_threshold).
7. Generate side-by-side 4-panel visual analysis composite (OpenCV/PIL) and JSON report.
"""

import os
import sys
import json
import re
import argparse
from typing import Tuple, Optional, List, Dict, Any
import numpy as np
from PIL import Image
import cv2
import torch

if sys.platform == "win32":
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", line_buffering=True)

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import rasterio
except ImportError:
    rasterio = None

from backend.services.prithvi_encoder import get_prithvi_encoder
from backend.services.temporal_head import load_trained_head


def compute_patch_spectral_deltas(crop_b: np.ndarray, crop_a: np.ndarray) -> np.ndarray:
    """
    Computes [delta_ndvi, delta_ndbi, delta_ndwi] from multi-band crops (C, H, W).
    Assumes standard Sentinel-2 order:
    0: Blue (B02), 1: Green (B03), 2: Red (B04), 3: NIR (B08), 4: SWIR1 (B11), 5: SWIR2 (B12)
    """
    def get_indices(c: np.ndarray):
        b_green = c[1].astype(np.float32)
        b_red = c[2].astype(np.float32)
        b_nir = c[3].astype(np.float32)
        b_swir = c[4].astype(np.float32) if c.shape[0] > 4 else c[3].astype(np.float32)
        eps = 1e-6
        ndvi = (b_nir - b_red) / (b_nir + b_red + eps)
        ndbi = (b_swir - b_nir) / (b_swir + b_nir + eps)
        ndwi = (b_green - b_nir) / (b_green + b_nir + eps)
        return float(np.nanmean(ndvi)), float(np.nanmean(ndbi)), float(np.nanmean(ndwi))

    ndvi1, ndbi1, ndwi1 = get_indices(crop_b)
    ndvi2, ndbi2, ndwi2 = get_indices(crop_a)
    return np.array([ndvi2 - ndvi1, ndbi2 - ndbi1, ndwi2 - ndwi1], dtype=np.float32)


def _to_rgb_preview(multiband: np.ndarray, band_descriptions: Optional[List[str]] = None) -> Image.Image:
    """
    Extracts an RGB visualization (H, W, 3) uint8 from a multi-band raster array (C, H, W).
    Uses percentile stretching for crisp visual rendering.
    """
    c, h, w = multiband.shape

    # Determine Red, Green, Blue channel indices
    r_idx, g_idx, b_idx = 0, 1, 2
    if band_descriptions:
        desc_lower = [str(d).lower() for d in band_descriptions]
        for idx, name in enumerate(desc_lower):
            if "red" in name or name == "b04" or name == "b4":
                r_idx = idx
            elif "green" in name or name == "b03" or name == "b3":
                g_idx = idx
            elif "blue" in name or name == "b02" or name == "b2":
                b_idx = idx
    elif c >= 4:
        # Standard Sentinel-2 L2A order in pipeline: [B02/Blue, B03/Green, B04/Red, B08/NIR]
        b_idx, g_idx, r_idx = 0, 1, 2

    # Extract RGB bands
    r = multiband[r_idx].astype(np.float32)
    g = multiband[g_idx].astype(np.float32)
    b = multiband[b_idx].astype(np.float32)

    def stretch_band(band: np.ndarray) -> np.ndarray:
        valid = band[~np.isnan(band) & ~np.isinf(band) & (band > 0)]
        if len(valid) == 0:
            return np.zeros_like(band, dtype=np.uint8)
        p2, p98 = np.percentile(valid, (2, 98))
        if p98 > p2:
            stretched = np.clip((band - p2) / (p98 - p2), 0.0, 1.0) * 255.0
        else:
            stretched = np.clip(band * 255.0, 0, 255)
        return stretched.astype(np.uint8)

    rgb = np.stack([stretch_band(r), stretch_band(g), stretch_band(b)], axis=-1)
    return Image.fromarray(rgb, mode="RGB")


def load_multiband_and_mask(
    tif_path: str,
    mask_path: Optional[str] = None
) -> Tuple[np.ndarray, Image.Image, Optional[np.ndarray], Dict[str, Any]]:
    """
    Loads multi-band GeoTIFF, generates RGB preview, and loads mask if present.
    Returns: (multiband_array, pil_rgb_preview, mask_array, metadata)
    """
    if not os.path.exists(tif_path):
        raise FileNotFoundError(f"GeoTIFF file not found: {tif_path}")

    if rasterio is None:
        raise ImportError("Rasterio is required to read GeoTIFFs. Ensure rasterio is installed.")

    with rasterio.open(tif_path) as ds:
        multiband = ds.read().astype(np.float32)
        descriptions = list(ds.descriptions) if ds.descriptions else None
        meta = {
            "crs": str(ds.crs),
            "transform": [float(x) for x in ds.transform],
            "bounds": [float(x) for x in ds.bounds],
            "count": ds.count,
            "height": ds.height,
            "width": ds.width
        }

    c, h, w = multiband.shape
    rgb_preview = _to_rgb_preview(multiband, descriptions)

    mask = None
    if mask_path and os.path.exists(mask_path):
        with rasterio.open(mask_path) as ds:
            mask = ds.read(1)
        if mask.shape != (h, w):
            mask_img = Image.fromarray(mask).resize((w, h), Image.NEAREST)
            mask = np.array(mask_img)

    return multiband, rgb_preview, mask, meta


def analyze_prithvi_patches(
    before_tif: str,
    after_tif: str,
    before_mask_path: Optional[str] = None,
    after_mask_path: Optional[str] = None,
    grid_size: int = 8,
    quality_thresh: float = 0.20,
    z_threshold: float = 2.0,
    output_dir: str = "data/runs/prithvi_patch_change_demo",
    batch_size: int = 16
) -> Dict[str, Any]:
    print(f"\n=======================================================")
    print(f"🛰️  PRITHVI-EO-2.0-300M MULTI-BAND CHANGE DETECTOR")
    print(f"=======================================================")
    print(f"Before GeoTIFF: {before_tif}")
    print(f"After GeoTIFF:  {after_tif}")
    print(f"Grid Layout:    {grid_size}x{grid_size} ({grid_size*grid_size} total patches)")
    print(f"Quality Filter: Max Bad Pixel Fraction <= {quality_thresh*100:.0f}%")
    print(f"Z-Threshold:    z >= {z_threshold}")
    print(f"=======================================================\n")

    os.makedirs(output_dir, exist_ok=True)

    # 1. Load Multi-Band Data & Mask
    mb_before, rgb_before, mask_before, meta_b = load_multiband_and_mask(before_tif, before_mask_path)
    mb_after, rgb_after, mask_after, meta_a = load_multiband_and_mask(after_tif, after_mask_path)

    W, H = rgb_before.size
    patch_w = W // grid_size
    patch_h = H // grid_size
    print(f"Raster Dimensions: {W}x{H} | Patch Dimensions: {patch_w}x{patch_h} px | Bands: {mb_before.shape[0]}")

    # 2. Patch Extraction & Quality Audit
    patches_meta = []
    crops_before = []
    crops_after = []
    valid_indices = []

    print("\n[Stage 1] Dividing multi-spectral raster into patches and executing Quality Check...")
    for r in range(grid_size):
        for c in range(grid_size):
            idx = r * grid_size + c
            x0 = c * patch_w
            y0 = r * patch_h
            x1 = x0 + patch_w
            y1 = y0 + patch_h

            crop_b = mb_before[:, y0:y1, x0:x1]
            crop_a = mb_after[:, y0:y1, x0:x1]

            # Quality Check from bad masks
            bad_frac_b = 0.0
            bad_frac_a = 0.0
            if mask_before is not None:
                patch_m_b = mask_before[y0:y1, x0:x1]
                bad_frac_b = float((patch_m_b > 0).mean())
            if mask_after is not None:
                patch_m_a = mask_after[y0:y1, x0:x1]
                bad_frac_a = float((patch_m_a > 0).mean())

            # Check nodata / empty array
            is_empty_b = bool(np.mean(crop_b) < 1e-4)
            is_empty_a = bool(np.mean(crop_a) < 1e-4)

            is_valid = True
            drop_reason = None
            if bad_frac_b > quality_thresh or bad_frac_a > quality_thresh:
                is_valid = False
                drop_reason = f"Cloud/Bad pixel (T1: {bad_frac_b:.1%}, T2: {bad_frac_a:.1%})"
            elif is_empty_b or is_empty_a:
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

    # 3. Embedding Generation via Prithvi-EO-2.0-300M
    print("\n[Stage 2] Computing 1024-dim Prithvi-EO-2.0-300M embeddings for clear multi-spectral patches...")
    encoder = get_prithvi_encoder()

    if valid_indices:
        print(f"Encoding {len(crops_before)} Before multi-spectral patches...")
        vecs_b = encoder.encode_multiband_patches(crops_before, batch_size=batch_size)
        print(f"Encoding {len(crops_after)} After multi-spectral patches...")
        vecs_a = encoder.encode_multiband_patches(crops_after, batch_size=batch_size)

        # 4. Semantic Distance (1.0 - Cosine Similarity)
        # vecs are L2-normalized
        sims = np.sum(vecs_b * vecs_a, axis=1)
        sims = np.clip(sims, -1.0, 1.0)
        distances = 1.0 - sims

        # Assign back to patch meta
        for i, val_idx in enumerate(valid_indices):
            patches_meta[val_idx]["distance"] = float(distances[i])

        # 5. Robust Statistics (Median and MAD) - Spatial Null Model
        med_dist = float(np.median(distances))
        abs_dev = np.abs(distances - med_dist)
        mad = float(np.median(abs_dev))
        mad = max(mad, 1e-4)

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

        candidates.sort(key=lambda x: -x["distance"])
        print(f"\n[Stage 4] Detected {len(candidates)} Change Candidate Patches (z >= {z_threshold}):")
        for rank, cand in enumerate(candidates[:10], 1):
            print(f"  Rank #{rank:02d} | Patch [{cand['row']}, {cand['col']}] (BBox: {cand['bbox']}) | Dist: {cand['distance']:.4f} | z-score: {cand['z_score']:+.2f}")

        # 4.5. Semantic Classification of Anomaly Patches via Trained Temporal Head
        change_head, head_classes = load_trained_head("models/change_head/temporal_change_head.pth")
        if change_head is not None and candidates:
            head_dev = "cuda" if torch.cuda.is_available() else "cpu"
            change_head = change_head.to(head_dev).eval()
            print(f"\n[Stage 4.5] Classifying {len(candidates)} candidates via SOTA Gated Residual Semantic Head ({head_dev})...")

            val_id_to_idx = {val_id: i for i, val_id in enumerate(valid_indices)}

            for rank, cand in enumerate(candidates, 1):
                i = val_id_to_idx[cand["patch_id"]]
                e1 = vecs_b[i]
                e2 = vecs_a[i]
                delta_e = e2 - e1
                c_b = crops_before[i]
                c_a = crops_after[i]
                delta_s = compute_patch_spectral_deltas(c_b, c_a)

                feat_vec = np.concatenate([delta_e, e1, delta_s], axis=0).astype(np.float32)
                with torch.no_grad():
                    feat_t = torch.from_numpy(feat_vec).unsqueeze(0).to(head_dev)
                    probs = change_head.predict_probabilities(feat_t).cpu().numpy()[0]
                    p_idx = int(np.argmax(probs))
                    p_label = head_classes[p_idx]
                    p_conf = float(probs[p_idx])

                cand["predicted_transition"] = p_label
                cand["confidence"] = round(p_conf * 100, 1)
                cand["transition_probabilities"] = {
                    head_classes[k]: round(float(probs[k]) * 100, 1) for k in range(len(head_classes))
                }
                print(f"  Candidate #{rank:02d} | Patch [{cand['row']}, {cand['col']}] => {p_label} ({cand['confidence']}%)")
    else:
        print("Warning: No clear/valid patches found after quality filtering!")
        candidates = []
        med_dist, mad = 0.0, 1.0

    # 6. Generate 4-Panel Side-by-Side Visual Plot
    print("\n[Stage 5] Generating 4-panel visual analysis panel...")
    p1 = np.array(rgb_before).copy()
    p2 = np.array(rgb_after).copy()
    p4 = np.array(rgb_after).copy()

    # Draw grid lines on p1, p2, p4
    for p in [p1, p2, p4]:
        for i in range(1, grid_size):
            cv2.line(p, (i * patch_w, 0), (i * patch_w, H), (56, 189, 248), 1, cv2.LINE_AA)
            cv2.line(p, (0, i * patch_h), (W, i * patch_h), (56, 189, 248), 1, cv2.LINE_AA)

    # Panel 3: Semantic Heatmap
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

    heat_small = cv2.applyColorMap(dist_grid_norm, cv2.COLORMAP_MAGMA)
    heat_small = cv2.cvtColor(heat_small, cv2.COLOR_BGR2RGB)
    p3 = cv2.resize(heat_small, (W, H), interpolation=cv2.INTER_NEAREST)

    # Darken dropped patches in heatmap
    for p in patches_meta:
        if not p["is_valid"]:
            x0, y0, x1, y1 = p["bbox"]
            p3[y0:y1, x0:x1] = (50, 50, 50)
            cv2.line(p3, (x0, y0), (x1, y1), (100, 100, 100), 1)
            cv2.line(p3, (x0, y1), (x1, y0), (100, 100, 100), 1)

    for i in range(1, grid_size):
        cv2.line(p3, (i * patch_w, 0), (i * patch_w, H), (30, 41, 59), 1)
        cv2.line(p3, (0, i * patch_h), (W, i * patch_h), (30, 41, 59), 1)

    # Highlight Candidates on Panel 4 with Semantic Badges
    for rank, cand in enumerate(candidates, 1):
        x0, y0, x1, y1 = cand["bbox"]
        color = (239, 68, 68) if rank <= 3 else (245, 158, 11)
        cv2.rectangle(p4, (x0, y0), (x1, y1), color, 2)

        pred_trans = cand.get("predicted_transition", "")
        short_trans = pred_trans.split(" / ")[0] if " / " in pred_trans else pred_trans
        short_trans = short_trans.replace(" (Snow & Ice Shift)", "").replace(" & Ecological Succession", "")
        conf = cand.get("confidence", 0)

        badge_line1 = f"#{rank} z={cand['z_score']:+.1f}"
        badge_line2 = f"{short_trans} ({conf:.0f}%)" if pred_trans else ""

        box_w = max(len(badge_line1) * 7 + 10, len(badge_line2) * 6 + 10)
        box_h = 28 if badge_line2 else 16

        cv2.rectangle(p4, (x0, y0), (min(x0 + box_w, W), min(y0 + box_h, H)), (15, 23, 42), -1)
        cv2.rectangle(p4, (x0, y0), (min(x0 + box_w, W), min(y0 + box_h, H)), color, 1)
        cv2.putText(p4, badge_line1, (x0 + 3, y0 + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
        if badge_line2:
            cv2.putText(p4, badge_line2, (x0 + 3, y0 + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.30, color, 1, cv2.LINE_AA)


    # Composite canvas
    header_h = 50
    spacing = 15
    canvas_w = (W * 4) + (spacing * 5)
    canvas_h = H + header_h + 30
    canvas = np.full((canvas_h, canvas_w, 3), 15, dtype=np.uint8)  # Slate dark #0f172a

    def extract_date_label(path_str: str) -> str:
        m1 = re.search(r"(\d{4}-\d{2}-\d{2})", path_str)
        if m1:
            return m1.group(1)
        m2 = re.search(r"(\d{4})(\d{2})(\d{2})", os.path.basename(path_str))
        if m2:
            return f"{m2.group(1)}-{m2.group(2)}-{m2.group(3)}"
        return os.path.basename(path_str)

    date_b = extract_date_label(before_tif)
    date_a = extract_date_label(after_tif)

    titles = [
        f"T1 Before ({date_b})",
        f"T2 After ({date_a})",
        f"Prithvi 300M Distance (Med:{med_dist:.2f}, MAD:{mad:.2f})",
        f"Change Candidates ({len(candidates)} flagged)"
    ]

    for idx, (p_img, title) in enumerate(zip([p1, p2, p3, p4], titles)):
        x_offset = spacing + idx * (W + spacing)
        y_offset = header_h
        canvas[y_offset : y_offset + H, x_offset : x_offset + W] = p_img
        cv2.putText(canvas, title, (x_offset, header_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (248, 250, 252), 1, cv2.LINE_AA)

    plot_out = os.path.join(output_dir, "patch_change_analysis.png")
    plot_out_c = os.path.join(output_dir, "c.png")
    Image.fromarray(canvas).save(plot_out)
    Image.fromarray(canvas).save(plot_out_c)
    print(f"Visual Analysis Plot saved to: {plot_out}")
    print(f"Visual Analysis Plot (c.png) saved to: {plot_out_c}")

    # 7. Save JSON Result matching downstream contract
    result_data = {
        "encoder": "Prithvi-EO-2.0-300M",
        "embedding_dim": 1024,
        "before_image": before_tif,
        "after_image": after_tif,
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
    parser = argparse.ArgumentParser(description="Multi-spectral patch change detection using Prithvi-EO-2.0-300M")
    parser.add_argument(
        "--before",
        default="data/tiles/berlin_tesla/2021-02-22/S2A_33UVU_20210222_2_L2A_tile_00002.tif",
        help="Path to Before GeoTIFF (.tif)"
    )
    parser.add_argument(
        "--after",
        default="data/tiles/berlin_tesla/2023-09-10/S2A_33UVU_20230910_1_L2A_tile_00002.tif",
        help="Path to After GeoTIFF (.tif)"
    )
    parser.add_argument(
        "--before-mask",
        default="data/tiles/berlin_tesla/2021-02-22/S2A_33UVU_20210222_2_L2A_tile_00002_mask.tif",
        help="Path to Before QA mask GeoTIFF"
    )
    parser.add_argument(
        "--after-mask",
        default="data/tiles/berlin_tesla/2023-09-10/S2A_33UVU_20230910_1_L2A_tile_00002_mask.tif",
        help="Path to After QA mask GeoTIFF"
    )
    parser.add_argument("--grid", type=int, default=8, help="Grid size (default 8 for 8x8 = 64 grids, 64x64 px each)")
    parser.add_argument("--quality-thresh", type=float, default=0.20, help="Max bad pixel fraction per patch")
    parser.add_argument("--z-thresh", type=float, default=2.0, help="Z-score threshold for change candidates")
    parser.add_argument("--out-dir", default="data/runs/prithvi_patch_change_demo", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=16, help="Inference batch size")

    args = parser.parse_args()

    analyze_prithvi_patches(
        before_tif=args.before,
        after_tif=args.after,
        before_mask_path=args.before_mask,
        after_mask_path=args.after_mask,
        grid_size=args.grid,
        quality_thresh=args.quality_thresh,
        z_threshold=args.z_thresh,
        output_dir=args.out_dir,
        batch_size=args.batch_size
    )
