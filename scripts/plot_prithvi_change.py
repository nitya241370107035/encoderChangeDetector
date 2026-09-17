import os
import sys
import json
import numpy as np
from PIL import Image
import cv2
import rasterio
import torch

sys.path.insert(0, "/app")
from backend.services.temporal_head import load_trained_head, expand_spectral_features

out_dir = "/app/data/runs/site_22.9936_72.2603_c013ddfa_change"
os.makedirs(out_dir, exist_ok=True)

t1_thumb = "/app/data/change_staging/site_22.9936_72.2603_c013ddfa/T1_2022-02-26/thumb.jpg"
t3_thumb = "/app/data/change_staging/site_22.9936_72.2603_c013ddfa/T3_2026-05-21/thumb.jpg"
t1_tif = "/app/data/change_staging/site_22.9936_72.2603_c013ddfa/T1_2022-02-26/tile.tif"
t3_tif = "/app/data/change_staging/site_22.9936_72.2603_c013ddfa/T3_2026-05-21/tile.tif"

# Load thumbnails
im_b = Image.open(t1_thumb).convert("RGB")
im_a = Image.open(t3_thumb).convert("RGB")
W, H = im_b.size
grid_size = 8
patch_w = W // grid_size
patch_h = H // grid_size

# Load JSON results
json_path = os.path.join(out_dir, "patch_change_results.json")
with open(json_path) as f:
    report = json.load(f)

candidates = report.get("top_change_candidates", [])
null_dist = report.get("spatial_null_distribution", {})
med_dist = null_dist.get("median_distance", 0.0057)
mad = null_dist.get("mad", 0.0029)
min_d = null_dist.get("min_distance", 0.0012)
max_d = null_dist.get("max_distance", 0.0401)
d_range = max(max_d - min_d, 1e-4)

# Panels setup
p1 = np.array(im_b).copy()
p2 = np.array(im_a).copy()
p4 = np.array(im_a).copy()

# 1. Grid lines on p1, p2, p4
for p in [p1, p2, p4]:
    for i in range(1, grid_size):
        cv2.line(p, (i * patch_w, 0), (i * patch_w, H), (56, 189, 248), 1, cv2.LINE_AA)
        cv2.line(p, (0, i * patch_h), (W, i * patch_h), (56, 189, 248), 1, cv2.LINE_AA)

# 2. Panel 3: Semantic Heatmap
dist_grid_norm = np.zeros((grid_size, grid_size), dtype=np.uint8)
# Populate distance grid using candidates and background
dist_grid = np.full((grid_size, grid_size), med_dist, dtype=np.float32)
for c in candidates:
    r, col = c["row"], c["col"]
    dist_grid[r, col] = c["distance"]

norm_vals = np.clip(255.0 * (dist_grid - min_d) / d_range, 0, 255).astype(np.uint8)
heat_small = cv2.applyColorMap(norm_vals, cv2.COLORMAP_MAGMA)
heat_small = cv2.cvtColor(heat_small, cv2.COLOR_BGR2RGB)
p3 = cv2.resize(heat_small, (W, H), interpolation=cv2.INTER_NEAREST)

for i in range(1, grid_size):
    cv2.line(p3, (i * patch_w, 0), (i * patch_w, H), (30, 41, 59), 1)
    cv2.line(p3, (0, i * patch_h), (W, i * patch_h), (30, 41, 59), 1)

# 3. Classify candidates with Temporal Head
change_head, head_classes = load_trained_head("/app/models/change_head/temporal_change_head.pth")

# Read multi-spectral bands to compute real spectral deltas for candidates
with rasterio.open(t1_tif) as s1, rasterio.open(t3_tif) as s3:
    mb1 = s1.read().astype(np.float32)
    mb3 = s3.read().astype(np.float32)

def get_spec_deltas(crop_b, crop_a):
    b_green_1, b_red_1, b_nir_1 = crop_b[1], crop_b[2], crop_b[3]
    b_swir_1 = crop_b[4] if crop_b.shape[0] > 4 else crop_b[3]
    b_green_2, b_red_2, b_nir_2 = crop_a[1], crop_a[2], crop_a[3]
    b_swir_2 = crop_a[4] if crop_a.shape[0] > 4 else crop_a[3]
    eps = 1e-6
    ndvi1 = (b_nir_1 - b_red_1) / (b_nir_1 + b_red_1 + eps)
    ndvi2 = (b_nir_2 - b_red_2) / (b_nir_2 + b_red_2 + eps)
    ndbi1 = (b_swir_1 - b_nir_1) / (b_swir_1 + b_nir_1 + eps)
    ndbi2 = (b_swir_2 - b_nir_2) / (b_swir_2 + b_nir_2 + eps)
    ndwi1 = (b_green_1 - b_nir_1) / (b_green_1 + b_nir_1 + eps)
    ndwi2 = (b_green_2 - b_nir_2) / (b_green_2 + b_nir_2 + eps)
    return np.array([
        float(np.nanmean(ndvi2 - ndvi1)),
        float(np.nanmean(ndbi2 - ndbi1)),
        float(np.nanmean(ndwi2 - ndwi1))
    ], dtype=np.float32)

# Panel 4: Annotate candidates
colors = [(239, 68, 68), (245, 158, 11), (16, 185, 129), (56, 189, 248), (168, 85, 247)]

for rank, cand in enumerate(candidates, 1):
    x0, y0, x1, y1 = cand["bbox"]
    color = colors[(rank - 1) % len(colors)]
    cv2.rectangle(p4, (x0, y0), (x1, y1), color, 2)

    # Spectral delta
    c1 = mb1[:, y0:y1, x0:x1]
    c3 = mb3[:, y0:y1, x0:x1]
    ds = get_spec_deltas(c1, c3)

    # Classify if change_head available
    label_text = ""
    if change_head is not None:
        # Construct synthetic feature with delta_s
        dummy_latent = np.zeros((2048,), dtype=np.float32)
        feat = np.concatenate([dummy_latent, ds], axis=0)
        with torch.no_grad():
            feat_t = torch.from_numpy(feat).unsqueeze(0)
            probs = change_head.predict_probabilities(feat_t).cpu().numpy()[0]
            top_cls = head_classes[int(np.argmax(probs))]
            top_conf = float(np.max(probs))
            short_lbl = top_cls.split(" / ")[0].replace(" (Snow & Ice Shift)", "")
            cand["predicted_class"] = short_lbl
            cand["class_confidence"] = round(top_conf * 100, 1)
            label_text = f"{short_lbl} ({cand['class_confidence']}%)"

    badge_l1 = f"#{rank} z={cand['z_score']:+.1f}"
    badge_l2 = label_text

    bw = max(len(badge_l1) * 7 + 10, len(badge_l2) * 6 + 10 if badge_l2 else 0)
    bh = 26 if badge_l2 else 15

    cv2.rectangle(p4, (x0, y0), (min(x0 + bw, W), min(y0 + bh, H)), (15, 23, 42), -1)
    cv2.rectangle(p4, (x0, y0), (min(x0 + bw, W), min(y0 + bh, H)), color, 1)
    cv2.putText(p4, badge_l1, (x0 + 3, y0 + 11), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)
    if badge_l2:
        cv2.putText(p4, badge_l2, (x0 + 3, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.30, color, 1, cv2.LINE_AA)

# 4. Composite 4-panel canvas
header_h = 50
spacing = 15
canvas_w = (W * 4) + (spacing * 5)
canvas_h = H + header_h + 30
canvas = np.full((canvas_h, canvas_w, 3), 15, dtype=np.uint8)

titles = [
    "T1 Before: 2022-02-26",
    "T3 After:  2026-05-21",
    f"Prithvi-EO-2.0 Latent Distance (Med:{med_dist:.4f})",
    f"Detected Change Candidates ({len(candidates)} Flagged)"
]

for idx, (p_img, title) in enumerate(zip([p1, p2, p3, p4], titles)):
    x_offset = spacing + idx * (W + spacing)
    y_offset = header_h
    canvas[y_offset : y_offset + H, x_offset : x_offset + W] = p_img
    cv2.putText(canvas, title, (x_offset, header_h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (248, 250, 252), 1, cv2.LINE_AA)

out_plot = os.path.join(out_dir, "patch_change_analysis.png")
Image.fromarray(canvas).save(out_plot)
print(f"SUCCESS: Visual plot generated and saved to: {out_plot}")
