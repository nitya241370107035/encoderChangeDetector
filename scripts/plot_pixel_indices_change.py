import os
import sys
import json
import numpy as np
from PIL import Image
import cv2
import rasterio

out_dir = "/app/data/runs/site_22.9936_72.2603_c013ddfa_change"
os.makedirs(out_dir, exist_ok=True)

t1_tif = "/app/data/change_staging/site_22.9936_72.2603_c013ddfa/T1_2022-02-26/tile.tif"
t3_tif = "/app/data/change_staging/site_22.9936_72.2603_c013ddfa/T3_2026-05-21/tile.tif"
t1_thumb = "/app/data/change_staging/site_22.9936_72.2603_c013ddfa/T1_2022-02-26/thumb.jpg"
t3_thumb = "/app/data/change_staging/site_22.9936_72.2603_c013ddfa/T3_2026-05-21/thumb.jpg"

print("Reading Sentinel-2 multi-spectral bands...")
with rasterio.open(t1_tif) as s1, rasterio.open(t3_tif) as s3:
    b1 = s1.read().astype(np.float32)
    b3 = s3.read().astype(np.float32)

# Band indices (0-based):
# B03: Green (idx 2), B04: Red (idx 3), B08: NIR (idx 5), B11: SWIR (idx 7)
g1, r1, n1, s1_ = b1[2], b1[3], b1[5], b1[7]
g3, r3, n3, s3_ = b3[2], b3[3], b3[5], b3[7]

eps = 1e-6
ndvi1 = (n1 - r1) / (n1 + r1 + eps)
ndvi3 = (n3 - r3) / (n3 + r3 + eps)

ndbi1 = (s1_ - n1) / (s1_ + n1 + eps)
ndbi3 = (s3_ - n3) / (s3_ + n3 + eps)

ndwi1 = (g1 - n1) / (g1 + n1 + eps)
ndwi3 = (g3 - n3) / (g3 + n3 + eps)

# Delta and Change Vector Magnitude
d_ndvi = ndvi3 - ndvi1
d_ndbi = ndbi3 - ndbi1
d_ndwi = ndwi3 - ndwi1
change_magnitude = np.sqrt(d_ndvi**2 + d_ndbi**2 + d_ndwi**2)

# Land Cover Classification Rules (NDVI, NDBI, NDWI)
# Classes:
# 0: Water Body
# 1: Dense Vegetation
# 2: Moderate / Cropland Vegetation
# 3: Built-up / Impervious Surface
# 4: Bare Soil / Barren
def classify_landcover(ndvi, ndbi, ndwi):
    cls = np.full(ndvi.shape, 4, dtype=np.int32) # Default Bare Soil
    
    # 0. Water Body: Positive NDWI, low NDVI
    water_mask = (ndwi > 0.0) & (ndvi < 0.15)
    cls[water_mask] = 0
    
    # 1. Dense Vegetation: NDVI >= 0.35, NDBI < 0.05
    dense_veg = (ndvi >= 0.35) & (ndbi < 0.05) & ~water_mask
    cls[dense_veg] = 1
    
    # 2. Moderate / Agricultural Veg: 0.20 <= NDVI < 0.35, NDVI > NDBI
    mod_veg = (ndvi >= 0.20) & (ndvi < 0.35) & (ndvi > ndbi) & ~water_mask
    cls[mod_veg] = 2
    
    # 3. Built-up / Impervious: NDBI >= 0.05, NDBI > NDVI
    built = (ndbi >= 0.05) & (ndbi > ndvi) & ~water_mask
    cls[built] = 3
    
    return cls

c1 = classify_landcover(ndvi1, ndbi1, ndwi1)
c3 = classify_landcover(ndvi3, ndbi3, ndwi3)

# Filter: True change requires class shift AND significant change magnitude (>= 0.20)
is_change = (c1 != c3) & (change_magnitude >= 0.20)
total_pixels = c1.size
changed_pixel_count = int(np.sum(is_change))

print(f"Total Pixels: {total_pixels} | Changed Pixels: {changed_pixel_count} ({changed_pixel_count / total_pixels * 100:.2f}%)")

# Categorize transitions for CHANGED pixels
# Transition map:
# 0: Unchanged (background)
# 1: New Built-up / Urbanization (Red)
# 2: Vegetation Loss / Clearing (Orange)
# 3: Vegetation Growth / Regrowth (Green)
# 4: Water Inundation / Ponding (Blue)
# 5: Water Depletion / Reclamation (Yellow)
# 6: Other Surface Transformation (Purple)
trans_map = np.zeros(c1.shape, dtype=np.uint8)

# 1: New Built-up (Bare or Veg -> Built-up)
trans_map[is_change & (c3 == 3) & (c1 != 3)] = 1

# 2: Vegetation Loss (Dense or Mod Veg -> Bare Soil)
trans_map[is_change & ((c1 == 1) | (c1 == 2)) & (c3 == 4)] = 2

# 3: Vegetation Growth (Bare or Built -> Dense or Mod Veg)
trans_map[is_change & ((c1 == 4) | (c1 == 3)) & ((c3 == 1) | (c3 == 2))] = 3

# 4: Water Inundation (Non-water -> Water)
trans_map[is_change & (c3 == 0) & (c1 != 0)] = 4

# 5: Water Depletion / Reclamation (Water -> Non-water)
trans_map[is_change & (c1 == 0) & (c3 != 0)] = 5

# 6: Other Structural Transformation
trans_map[is_change & (trans_map == 0)] = 6

# Color definitions (RGB)
color_palette = {
    1: ("New Built-up / Urbanization", (239, 68, 68), "Red"),          # #ef4444
    2: ("Vegetation Loss / Clearing", (245, 158, 11), "Amber"),        # #f59e0b
    3: ("Vegetation Growth / Regrowth", (16, 185, 129), "Emerald"),    # #10b981
    4: ("Water Inundation / Ponding", (14, 165, 233), "Sky Blue"),     # #0ea5e9
    5: ("Water Depletion / Reclamation", (234, 179, 8), "Yellow"),     # #eab308
    6: ("Other Surface Transformation", (168, 85, 247), "Purple"),     # #a855f7
}

# Compile stats
stats_breakdown = {}
for code, (label, rgb, color_name) in color_palette.items():
    cnt = int(np.sum(trans_map == code))
    pct = round(cnt / total_pixels * 100, 2)
    stats_breakdown[label] = {
        "code": code,
        "pixel_count": cnt,
        "area_percentage": pct,
        "color_rgb": list(rgb),
        "color_name": color_name
    }

stats_output = {
    "site_id": "site_22.9936_72.2603_c013ddfa",
    "baseline_epoch": "T1_2022-02-26",
    "followup_epoch": "T3_2026-05-21",
    "total_pixels": total_pixels,
    "total_changed_pixels": changed_pixel_count,
    "total_changed_percentage": round(changed_pixel_count / total_pixels * 100, 2),
    "significance_threshold_magnitude": 0.20,
    "transition_breakdown": stats_breakdown
}

stats_json_path = os.path.join(out_dir, "pixel_change_stats.json")
with open(stats_json_path, "w") as f:
    json.dump(stats_output, f, indent=2)
print(f"Saved pixel change stats to: {stats_json_path}")

# Load RGB Thumbnails or construct from bands
im_b = Image.open(t1_thumb).convert("RGB")
im_a = Image.open(t3_thumb).convert("RGB")
W, H = im_b.size

# Panel 1 & 2: True Color images
p1 = np.array(im_b)
p2 = np.array(im_a)

# Panel 3: Change Vector Magnitude Heatmap
mag_norm = np.clip(change_magnitude / 0.60 * 255.0, 0, 255).astype(np.uint8)
mag_colored = cv2.applyColorMap(mag_norm, cv2.COLORMAP_VIRIDIS)
p3 = cv2.cvtColor(mag_colored, cv2.COLOR_BGR2RGB)

# Panel 4: ONLY CHANGED PIXELS HIGHLIGHTED
# Background: Dimmed/desaturated T3 image (35% brightness for context)
gray_bg = cv2.cvtColor(p2, cv2.COLOR_RGB2GRAY)
p4 = np.zeros_like(p2)
for ch in range(3):
    p4[:, :, ch] = (gray_bg * 0.35).astype(np.uint8)

# Paint ONLY changed pixels in their distinct category color
for code, (label, rgb, _) in color_palette.items():
    mask = (trans_map == code)
    p4[mask] = rgb

# Render composite 4-panel visual canvas
header_h = 55
legend_h = 110
spacing = 16
canvas_w = (W * 4) + (spacing * 5)
canvas_h = H + header_h + legend_h + spacing

# Dark slate sovereign theme background (#0b0f19)
canvas = np.full((canvas_h, canvas_w, 3), (11, 15, 25), dtype=np.uint8)

titles = [
    "1. T1 Baseline (2022-02-26)",
    "2. T3 Follow-up (2026-05-21)",
    f"3. Spectral Delta Magnitude (||dS||2 >= 0.20)",
    f"4. Classified Changed Pixels Only ({stats_output['total_changed_percentage']}% Scene Shift)"
]

for idx, (p_img, title) in enumerate(zip([p1, p2, p3, p4], titles)):
    x_offset = spacing + idx * (W + spacing)
    y_offset = header_h
    canvas[y_offset : y_offset + H, x_offset : x_offset + W] = p_img
    
    # Outer border around each panel
    border_color = (56, 189, 248) if idx == 3 else (71, 85, 105)
    border_thick = 2 if idx == 3 else 1
    cv2.rectangle(canvas, (x_offset - 1, y_offset - 1), (x_offset + W, y_offset + H), border_color, border_thick)
    
    # Title badge
    cv2.putText(canvas, title, (x_offset, header_h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (241, 245, 249), 1, cv2.LINE_AA)

# Draw tactical Legend for Panel 4 at the bottom
legend_y = header_h + H + 25
cv2.putText(canvas, "TACTICAL TRANSITION CLASSIFICATION (STANDARD NDVI / NDBI / NDWI RULES):", 
            (spacing, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (148, 163, 184), 1, cv2.LINE_AA)

# Draw legend swatches in two rows
items = list(color_palette.items())
col_w = (canvas_w - spacing * 2) // 3
for i, (code, (label, rgb, color_name)) in enumerate(items):
    col_idx = i % 3
    row_idx = i // 3
    lx = spacing + col_idx * col_w
    ly = legend_y + 15 + row_idx * 32
    
    # Color swatch
    cv2.rectangle(canvas, (lx, ly), (lx + 20, ly + 20), (int(rgb[0]), int(rgb[1]), int(rgb[2])), -1)
    cv2.rectangle(canvas, (lx, ly), (lx + 20, ly + 20), (255, 255, 255), 1)
    
    # Label and stats
    cnt = stats_breakdown[label]["pixel_count"]
    pct = stats_breakdown[label]["area_percentage"]
    text = f"{label}: {cnt:,} px ({pct}%)"
    cv2.putText(canvas, text, (lx + 28, ly + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (241, 245, 249), 1, cv2.LINE_AA)

out_plot = os.path.join(out_dir, "pixel_indices_change_analysis.png")
Image.fromarray(canvas).save(out_plot)
print(f"SUCCESS: Generated pixel indices change plot at: {out_plot}")
