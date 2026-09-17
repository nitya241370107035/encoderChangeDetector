import os
import sys
import numpy as np
import rasterio

# Ensure /app is in sys.path
sys.path.insert(0, "/app")

from backend.services.prithvi_encoder import get_prithvi_encoder

t1_path = "/app/data/change_staging/site_22.9936_72.2603_c013ddfa/T1_2022-02-26/tile.tif"
t3_path = "/app/data/change_staging/site_22.9936_72.2603_c013ddfa/T3_2026-05-21/tile.tif"

print("=" * 60)
print("  🛰️ Testing Prithvi-EO-2.0-300M Encoder on T1 vs T3")
print("=" * 60)
print(f"T1: {t1_path}")
print(f"T3: {t3_path}")
print("-" * 60)

# 1. Load GeoTIFFs
with rasterio.open(t1_path) as src1:
    data1 = src1.read().astype(np.float32)
    meta1 = src1.meta

with rasterio.open(t3_path) as src3:
    data3 = src3.read().astype(np.float32)
    meta3 = src3.meta

print(f"T1 Shape: {data1.shape} (Bands: {data1.shape[0]}, H: {data1.shape[1]}, W: {data1.shape[2]}) | Range: [{data1.min():.3f}, {data1.max():.3f}]")
print(f"T3 Shape: {data3.shape} (Bands: {data3.shape[0]}, H: {data3.shape[1]}, W: {data3.shape[2]}) | Range: [{data3.min():.3f}, {data3.max():.3f}]")

# 2. Initialize Encoder
print("\nLoading Prithvi-EO-2.0-300M Encoder...")
encoder = get_prithvi_encoder()
print(f"Encoder loaded! Model class: {encoder.model.__class__.__name__}, Device: {encoder.device}")

# 3. Whole-Tile Encoding (Global Embedding)
print("\n--- Test 1: Global Whole-Tile Embedding (512x512 -> 224x224 -> 1024-D) ---")
emb1_global = encoder.encode_multiband_patches([data1], batch_size=1)[0]
emb3_global = encoder.encode_multiband_patches([data3], batch_size=1)[0]

print(f"T1 Embedding Shape: {emb1_global.shape}, L2-Norm: {np.linalg.norm(emb1_global):.4f}")
print(f"T3 Embedding Shape: {emb3_global.shape}, L2-Norm: {np.linalg.norm(emb3_global):.4f}")
print(f"Sample T1 Vector[:5]: {np.round(emb1_global[:5], 4)}")
print(f"Sample T3 Vector[:5]: {np.round(emb3_global[:5], 4)}")

# Global Similarity & Distance
global_cos_sim = float(np.dot(emb1_global, emb3_global))
global_cos_dist = 1.0 - global_cos_sim
print(f"\n>> Global Cosine Similarity: {global_cos_sim:.4f} ({global_cos_sim*100:.2f}%)")
print(f">> Global Semantic Distance: {global_cos_dist:.4f}")

# 4. Patch-Level Encoding (8x8 Grid = 64 patches of 64x64 px)
print("\n--- Test 2: Patch-Level Encoding (8x8 = 64 patches of 64x64 px) ---")
grid_size = 8
patch_h, patch_w = 64, 64

crops1, crops3 = [], []
for r in range(grid_size):
    for c in range(grid_size):
        crops1.append(data1[:, r*patch_h:(r+1)*patch_h, c*patch_w:(c+1)*patch_w])
        crops3.append(data3[:, r*patch_h:(r+1)*patch_h, c*patch_w:(c+1)*patch_w])

patches_emb1 = encoder.encode_multiband_patches(crops1, batch_size=16)
patches_emb3 = encoder.encode_multiband_patches(crops3, batch_size=16)

print(f"T1 Encoded Patches Matrix: {patches_emb1.shape}")
print(f"T3 Encoded Patches Matrix: {patches_emb3.shape}")

# Per-patch Cosine Similarities & Distances
patch_sims = np.sum(patches_emb1 * patches_emb3, axis=1)
patch_dists = 1.0 - patch_sims

med_dist = np.median(patch_dists)
abs_dev = np.abs(patch_dists - med_dist)
mad = np.median(abs_dev)
mad = max(mad, 1e-4)
z_scores = (patch_dists - med_dist) / mad

print(f"\nPatch Distance Distribution across 64 patches:")
print(f"  • Min Distance:    {patch_dists.min():.4f}")
print(f"  • Median Distance: {med_dist:.4f}")
print(f"  • MAD:             {mad:.4f}")
print(f"  • Max Distance:    {patch_dists.max():.4f}")

# Top 5 Change Candidates
ranked_indices = np.argsort(-patch_dists)
print(f"\nTop 5 Change Candidate Patches (Highest Semantic Shift):")
candidates = []
for rank, idx in enumerate(ranked_indices[:5], 1):
    r = int(idx // grid_size)
    c = int(idx % grid_size)
    print(f"  Rank #{rank} | Patch [{r}, {c}] | Distance: {patch_dists[idx]:.4f} | Z-Score: {z_scores[idx]:+.2f}")
    candidates.append({
        "rank": rank,
        "patch_id": int(idx),
        "row": r,
        "col": c,
        "bbox": [c * patch_w, r * patch_h, (c + 1) * patch_w, (r + 1) * patch_h],
        "distance": float(round(patch_dists[idx], 4)),
        "z_score": float(round(z_scores[idx], 2)),
        "is_candidate": True
    })

# Export structured JSON results
import json
out_dir = "/app/data/runs/site_22.9936_72.2603_c013ddfa_change"
os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, "patch_change_results.json")

report = {
    "encoder": "Prithvi-EO-2.0-300M",
    "embedding_dim": 1024,
    "site_key": "site_22.9936_72.2603_c013ddfa",
    "baseline_epoch": "T1_2022-02-26",
    "subsequent_epoch": "T3_2026-05-21",
    "before_image": t1_path,
    "after_image": t3_path,
    "global_metrics": {
        "cosine_similarity": float(round(global_cos_sim, 4)),
        "semantic_distance": float(round(global_cos_dist, 4))
    },
    "patch_grid": {
        "grid_size": grid_size,
        "total_patches": grid_size * grid_size,
        "patch_dimensions": [patch_h, patch_w]
    },
    "spatial_null_distribution": {
        "min_distance": float(round(patch_dists.min(), 4)),
        "median_distance": float(round(med_dist, 4)),
        "mad": float(round(mad, 4)),
        "max_distance": float(round(patch_dists.max(), 4))
    },
    "top_change_candidates": candidates
}

with open(out_file, "w") as f:
    json.dump(report, f, indent=2)

print(f"\n[Artifact Saved] Results written to: {out_file}")
print("\n" + "=" * 60)
print("  ✅ Prithvi-EO-2.0-300M Encoder Test Successfully Completed!")
print("=" * 60)

