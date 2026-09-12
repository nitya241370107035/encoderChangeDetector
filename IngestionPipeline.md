# Implementation Plan — Phase 1: Ingestion & Preprocessing Pipeline
### (Covers PS sections 2.2.3 quality handling, 2.2.6 ingestion/sovereignty, 2.2.7 constraints, and the data foundation for 2.2.1/2.2.2)

**Scope:** everything from "get imagery in" to "clean, quality-scored tiles embedded and sitting in both databases." Semantic search/retrieval logic itself is explicitly out of scope here — that's a separate tool, built next, consuming what this phase produces.

---

## Phase 1.0 — Input handling: TWO separate entry points, not one

This is the correction from before, and it needs to be architectural, not an afterthought. The PS requires the pipeline to work two different ways, and your ingestion code should have one shared core with two different "front doors":

**Entry Point A — AOI + date range (your own archive-building, e.g. Ahmedabad)**
- Input: GeoJSON polygon (from map-drawing, already implemented, or pasted coordinates) + a date range (`date_from`, `date_to`).
- This path uses the network (STAC API), which is fine — this happens *before* evaluation, while network access is still allowed.

**Entry Point B — direct file ingestion (the organiser's evaluation imagery)**
- Input: one or more GeoTIFF/COG file paths, handed to you directly — no AOI polygon, no STAC search, no network call.
- **This must work with network fully disabled** — it's the literal compliance path for 2.2.6 ("ingest organiser-defined common geospatial formats") and 2.2.7 ("network access disabled ... during evaluation").
- Read whatever metadata the file itself carries (CRS, bounds, band count) — don't assume it matches Sentinel-2's specific band order/count. Fail with a clear error if a required band is genuinely missing, rather than silently misreading one band as another.

**Both entry points converge into the same shared pipeline from Phase 1.2 onward.** Build Entry Point B now, even though you won't use it until demo day — an untested path is a real risk on evaluation day specifically because it's the one path you can't casually test with your normal workflow (it requires deliberately simulating "no network, here's a random file").

---

## Phase 1.1 — Entry Point A: multi-temporal STAC search (your correction: not just "recent")

**Goal:** given an AOI + date range, get **multiple** scenes spread across time — not one.

**Tasks:**
1. Validate the input polygon (`shapely .is_valid` check — as established, do this before anything else).
2. Split the requested date range into meaningful sub-windows (e.g. "old" and "recent" buckets, or more if you want a denser time series) — run a **separate STAC search per bucket**, not one search returning only the single most-recent low-cloud scene.
3. For each bucket: query AWS Element84 STAC (`sentinel-2-l2a`), filter by bbox + that bucket's date range + `eo:cloud_cover < 15%` (your number is fine), sort by cloud cover, pick the best match.
4. Record each selected scene's real metadata straight from the API response (`scene_id`, `acquisition_date`, `crs`) — same rule as before, this is the one true source for these fields.

**Output:** a list of selected scenes, one (or more) per time bucket — genuinely multi-temporal, ready for change detection later.

---

## Phase 1.2 — Canvas assembly (shared by both entry points)

**Tasks:**
1. For Entry Point A: download the needed bands (RGB + NIR at minimum) for each selected scene.
   For Entry Point B: read bands directly from the provided file(s), using `band_order`/`descriptions` from the file itself rather than assuming Sentinel-2's layout.
2. Reproject to EPSG:4326 once, at this stage — not per-tile.
3. Crop to the AOI's bounding box plus a small buffer (Entry Point A), or use the file's own extent (Entry Point B).

**Output:** one reprojected, cropped multi-band "big canvas" per scene, ready for quality handling.

---

## Phase 1.3 — Quality handling / false-alarm suppression (2.2.3), applied before storage

This is exactly the stage you described — done here, on the big canvas, before tiling. This part of your plan was already correct; formalizing it:

**Tasks:**
1. **Cloud masking** — `s2cloudless` on the canvas → per-pixel cloud probability → threshold to binary mask.
2. **Shadow masking** — dark-pixel-near-cloud heuristic, combined with the cloud mask into one "bad pixel" mask.
3. **Normalization** — percentile stretch (2nd-98th), computed excluding masked bad pixels so they don't skew the good pixels' normalization.
4. **Per-tile quality signals get computed later, in Phase 1.4**, once the canvas is actually sliced — `cloud_pct` is a per-tile number, not a per-canvas one.

**Important, carried over from earlier in this conversation:** this stage produces each tile's *individual* quality signal. The *pairwise* quality checks (co-registration between two specific dates, seasonal-gap guard) can't happen here — they belong to the change-detection comparison step, which runs later, on two already-ingested tiles. Don't try to fold registration_residual into this phase; it stays placeholder at ingestion time, exactly as previously decided.

---

## Phase 1.4 — Tiling + per-band index computation

**Tasks:**
1. Slice the cleaned canvas into 512×512 tiles (or your chosen `GROUND_CROP_SIZE`, per the zoom decision from earlier — pick one and be consistent across the whole archive).
2. Compute each tile's real geometry/centroid from the transform.
3. Compute each tile's real `cloud_pct` from the Phase 1.3 mask, cropped to that tile's window.
4. Compute derived indices **per tile**, from the bands you actually have:
   - **NDVI** = `(NIR - Red) / (NIR + Red)` — ✅ computable with your RGB+NIR band set.
   - **NDWI** = `(Green - NIR) / (Green + NIR)` — ✅ computable.
   - **NDBI** — ❌ **not computable with your current 4-band decision.** NDBI needs a SWIR band. **Decision point, don't skip this:** either (a) add a 5th band (Sentinel-2 B11, SWIR) to your download/storage now, before locking the ingestion code — this is a real, small addition if done now, and a painful re-ingestion later if decided after the fact — or (b) explicitly drop NDBI from this build and say so plainly in your architecture note, rather than quietly shipping a schema column that's always null. **Make this call today with the team, not silently.**
5. Store the **mean** NDVI/NDWI as a scalar per tile (for fast filtering), while the full-resolution index arrays are recomputed on demand from the raw bands whenever the change-detection step needs pixel-level detail — full raster arrays don't belong in Postgres rows.
6. Filter tiles against the exact AOI polygon (`intersects`, not just bbox) — Entry Point A only; Entry Point B has no drawn polygon to filter against, so all valid tiles from the provided file are kept.

**Output:** tiles with real geometry, real `cloud_pct`, real mean NDVI/NDWI, ready to save.

---

## Phase 1.5 — Save artifacts to disk

**Tasks:**
1. Full data `.tif` per tile — all bands, full bit depth — this is the only copy anything computational (embedding, NDVI recompute, change detection) ever reads from.
2. Thumbnail `.jpg` per tile — RGB bands only, rescaled to 8-bit, purely for frontend display. NIR is dropped here; JPG has no channel for it, as established.
3. Folder structure: `data/tiles/{region_id_or_scene_id}/{date}/`.
4. `manifest.json` per ingestion run, listing every tile's full record.

---

## Phase 1.6 — Schema updates (the actual changes to `schema.sql`)

Here's exactly what changes, and why each one is needed:

```sql
-- Real quality data now, not placeholders (cloud_pct/quality_confidence were
-- already columns; this just confirms they get REAL computed values from
-- Phase 1.3/1.4, not the old hardcoded stub values)

ALTER TABLE tiles ADD COLUMN band_order JSONB;
-- e.g. '["blue","green","red","nir"]' — records which array index is which
-- band for THIS tile, since not every source will order bands identically.
-- Prevents silently misreading a band if a future source differs.

ALTER TABLE tiles ADD COLUMN band_stats JSONB;
-- per-band summary only — e.g. {"blue": {"min":120,"max":4800,"mean":1350}, ...}
-- NOT full pixel arrays. This is a cheap, queryable summary; the real
-- per-pixel data lives only in the .tif at file_path.

ALTER TABLE tiles ADD COLUMN mean_ndvi FLOAT;
ALTER TABLE tiles ADD COLUMN mean_ndwi FLOAT;
-- scalar summaries, computed once at ingestion, for fast filtering/sorting
-- without opening any raster file. Full-resolution NDVI/NDWI are recomputed
-- on demand from file_path's raw bands when pixel-level detail is needed.

-- NDBI intentionally omitted unless the team decides to add a SWIR band
-- (Phase 1.4, step 4). If added later: ALTER TABLE tiles ADD COLUMN mean_ndbi FLOAT;

ALTER TABLE tiles ADD COLUMN thumbnail_path TEXT;
-- if not already present from earlier work — the RGB-only JPG's location,
-- separate from file_path (the full-data .tif).

ALTER TABLE tiles ADD COLUMN source_type TEXT DEFAULT 'aoi_search';
-- 'aoi_search' (Entry Point A) or 'organiser_provided' (Entry Point B) —
-- lets you trace, and later filter/report on, which ingestion path a tile
-- came through. Useful for the submission's evaluation report too.
```

**Everything else in the existing `schema.sql` stays as-is** — `scenes`, `change_events`, `clusters`, `review_items`, `exports`, `search_log`, `ingestion_coverage` are all still correct and unaffected by this phase's work.

---

## Phase 1.7 — Populate Postgres

**Tasks:**
1. Insert/upsert the scene row (`scenes` table) — once per selected scene, not per tile.
2. Insert/upsert each tile row (`tiles` table) — with all the real fields from Phases 1.3-1.6 now filled in properly, not stubbed.
3. Update `ingestion_coverage` for the region — increment `tile_count`, set `status` appropriately.

---

## Phase 1.8 — Embed and populate Qdrant

**Your existing `vector_schema.py` needs zero changes** — it's already correct: 512-dim (matching RemoteCLIP ViT-B/32), cosine distance, the right payload fields and indexes. Confirmed good as-is.

**Tasks:**
1. Run each tile's `.tif` (full bands, but the encoder will only actually consume whatever channels RemoteCLIP expects — typically RGB, confirm this against the checkpoint) through the RemoteCLIP encoder.
2. Build the `TileVectorPayload` per tile, matching the schema already defined.
3. Upsert into the `tile_embeddings` collection.

---

## Phase 1.9 — Validation pass (do this before calling Phase 1 done)

- Run Entry Point A end-to-end on a small AOI batch, confirm Postgres + Qdrant both have correct, matching rows.
- **Actually test Entry Point B** — take one already-downloaded `.tif`, feed it through that path in isolation, with network disabled if possible, confirm it produces the same quality tile output as Entry Point A would. This is the single most important test in this whole phase, since it's the one path that directly maps to how you'll be evaluated.
- Confirm `mean_ndvi`/`mean_ndwi` values look sane (vegetation-heavy tiles should show higher NDVI than bare ground/urban tiles) — a quick sanity check that the index math is actually wired correctly, not just present as columns.
- Confirm the NDBI decision (add SWIR or explicitly drop it) is actually reflected consistently — schema, ingestion code, and architecture note should all agree, not have the column exist while ingestion never fills it with an explanation why.

---

## What's explicitly NOT in this phase

- The actual `SemanticSearchTool` / RemoteCLIP query-time retrieval logic — separate tool, next phase, as you said.
- Pairwise change-detection quality checks (co-registration between two specific tiles, seasonal-gap guard) — belongs to the change-detection comparison step, not ingestion.
- Clustering (2.2.4) — runs as a periodic batch job over already-embedded tiles, not part of ingestion itself.
