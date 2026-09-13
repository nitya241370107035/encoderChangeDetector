# ProjectContext.md
### Semantic Retrieval & Multi-Temporal Change Analysis of Satellite Imagery

**PS Number:** SIH-26227  
**Title:** Semantic Retrieval and Multi-Temporal Change Analysis of Satellite Imagery  
**Organization:** Ministry of Defence (MoD)  
**Department:** Indian Army (DGIS)  
**Category:** Software · **Theme:** Space Technology  

---

## 1. What the Actual Problem Is (Problem Statement Breakdown)

### 1.1 The Operational Bottleneck & Background (PS 2.1)
Modern Earth Observation (EO) archives are growing exponentially in volume, complexity, and heterogeneity. Satellite imagery streams contain multi-temporal observations, multi-spectral band stacks (visible, NIR, SWIR), and multi-sensor products across disparate spatial and radiometric resolutions.

In conventional military, defense, and intelligence workflows, catalog search relies strictly on **structured metadata queries** — such as spatial bounding boxes / geographic coordinates, platform names, product types, and acquisition timestamps. 

This creates two critical operational bottlenecks for defense analysts:
1. **The "Know Before You Look" Blindspot:** Analysts cannot discover targets unless they already know the exact geographic coordinates and date window where an event occurred. They cannot search archives using plain natural language describing visual features or operational activities (e.g., *"newly built structures near a river"*, *"large vehicle concentrations on open ground"*, or *"aircraft parked on dispersed tarmac"*).
2. **The Multi-Temporal Change & False-Alarm Problem:** Manually inspecting large geographic expanses across multiple temporal epochs ($T_1$ vs. $T_2$) is slow and labor-intensive. Automated change detection systems fail operationally because they trigger massive volumes of **false alarms** caused by confounding environmental and physical factors:
   - Ephemeral atmospheric artifacts (cloud cover, cloud shadow projection, haze, aerosol scattering, snow).
   - Seasonal and cyclical phenology (seasonal vegetation swings, agricultural crop cycles, dry vs. wet season water levels).
   - Sensor differences, viewing geometry variations, off-nadir angles, and sub-pixel spatial co-registration misalignments.
3. **Operational Sovereignty & Air-Gapped Constraints:** Defense analysis systems cannot rely on external third-party cloud APIs, external LLM endpoints, or ongoing internet access. The entire solution must execute 100% on-premises in an isolated, air-gapped environment while supporting fast incremental ingestion of newly acquired imagery without rebuilding the entire database or vector index from scratch.

---

### 1.2 Core Capabilities & Constraints Mandated by the Problem Statement (PS 2.2 & 2.3)

The table below outlines the exact core capabilities, functional requirements, constraints, and evaluation criteria specified by the Indian Army / MoD in Problem Statement SIH-26227:

| PS Section | Capability / Requirement | What the PS Mandates |
|---|---|---|
| **2.2.1** | **Semantic and Multimodal Retrieval** | • Support free-text natural-language search over imagery tiles without requiring prior coordinates.<br>• Support multimodal image-to-image visual similarity retrieval (querying by an exemplar satellite image patch).<br>• Rank-order retrieved candidates by semantic relevance score.<br>• Provide compound filtering to refine queries by Area of Interest (AOI), date-range window, sensor modality, and metadata. |
| **2.2.2** | **Multi-Temporal Change Analysis** | • For a user-specified AOI and time window, identify physical, structural, and environmental changes (appearance, disappearance, expansion, contraction).<br>• Classify detected changes into supported military and terrain categories: construction, land clearance, water-extent variation, road development.<br>• Estimate the earliest available observation date at which the physical change is supported by usable imagery via temporal traversal. |
| **2.2.3** | **False-Alarm Suppression & Quality Handling** | • Suppress false changes caused by seasonal variation, sun angle, illumination differences, clouds, haze, snow, shadows, radiometric shifts, and imperfect co-registration.<br>• Utilize pixel-level validity masks, dynamic radiometric normalization, and confidence estimation gates.<br>• Prioritize analytically useful precision over indiscriminate change recall (zero tolerance for noise floods). |
| **2.2.4** | **Discovery and Clustering** | • Support unsupervised or embedding-based clustering of similar geographic and infrastructure sites across the entire archive.<br>• Allow an analyst who identifies a single location of interest to branch out and discover other locations with comparable visual or semantic characteristics without constructing manual queries for each site. |
| **2.2.5** | **Analyst Workflow and Provenance** | • Provide a ranked review queue featuring synchronized before-and-after evidence, coordinates, capture timestamps, sensor/source metadata, confidence metrics, and processing history.<br>• Enable analysts to confirm or reject candidate alerts with all decisions logged in an immutable audit trail.<br>• Support feedback integration for downstream result refinement/reranking.<br>• Guarantee complete source-scene and processing lineage preservation in exported intelligence reports. |
| **2.2.6** | **Scale, Incremental Ingestion & Sovereignty** | • Implement efficient high-dimensional vector and spatial indexing.<br>• Enable incremental addition of new satellite acquisitions without triggering a full index rebuild.<br>• Operate 100% on-premises in an air-gapped environment with zero cloud services or external API dependencies.<br>• Preserve native georeferencing and ingest standard formats (GeoTIFF, Cloud Optimized GeoTIFF / COG). |
| **2.2.7** | **Operational Constraints** | • The complete demonstration must run with network access strictly disabled after approved models, libraries, and datasets are staged locally.<br>• Pretrained public models are permitted only if their origin, license, and checkpoints are explicitly declared and packaged for offline use.<br>• Evaluation uses public or organizer-supplied imagery only (no classified operational data). |
| **2.3** | **Evaluation Criteria & Deliverables** | • **Semantic Retrieval Evaluation:** Tested against held-out natural language queries and relevance judgments.<br>• **Change Analysis Evaluation:** Tested against a held-out benchmark of labeled change and no-change test cases.<br>• **Required Submission Deliverables:** Source code, technical architecture documentation, index-build and incremental-ingestion guide, model and dataset provenance records, and a reproducible evaluation report detailing indexed area, scene/tile counts, build duration, storage footprint, query latency, and hardware specs. |

---

## 2. High-Level Architectural Approach & Pipeline Phases

To systematically solve the operational requirements of SIH-26227 without creating a monolithic bottleneck, our architecture is partitioned into modular, specialized pipeline phases:

1. **Ingestion & Preprocessing Pipeline:**  
   Standardizes heterogeneous incoming Earth Observation streams (multi-spectral Sentinel-2 and high-resolution optical Maxar), reprojects CRS, segments large rasters into uniform 512×512 tiles, extracts multi-spectral indices (NDVI, NDWI, NDBI), and computes radiometric baselines.

2. **Semantic & Multimodal Retrieval Pipeline:**  
   Maps satellite tile imagery and natural language descriptions into a shared high-dimensional latent space using a vision-language foundation model (RemoteCLIP ViT-B-32). Sub-second vector similarity indexing (Qdrant) powers text-to-image and image-to-image search with compound spatial, temporal, and sensor filtering.

3. **Multi-Temporal Change Analysis Pipeline:**  
   Evaluates co-registered bi-temporal observations over user-specified Area of Interest (AOI) time windows. It isolates real physical changes (appearance, disappearance, expansion, contraction), classifies change types, and estimates the earliest emergence date via temporal traversal.

4. **Unsupervised Clustering & Cross-Site Discovery:**  
   - **Background Job:** Periodically executes density-based clustering (**HDBSCAN**) across vector embeddings without arbitrary cluster count assumptions, isolating terrain clusters and representative medoids.  
   - **Query-Time Discovery ("Find Similar"):** Provides a lightweight similarity lookup allowing an analyst who flags a location of interest to immediately discover its visual and semantic "twins" across the archive without formulating a new query.

5. **Analyst Workflow, Verification & Provenance:**  
   Delivers interactive map workspaces, persistent multi-turn chat sessions, evidence inspection modals, and a confirm/reject analyst review queue tied to an audit trail and exportable provenance.

6. **100% Offline & Sovereign Deployment:**  
   The entire stack (FastAPI, PostGIS, Qdrant, MinIO) operates fully containerized on-premises with pre-packaged model weights, strictly functioning in air-gapped environments without any external cloud or API dependencies.

---

### Two-Tiered False-Alarm Suppression Strategy

False alarms are addressed at **two distinct, dedicated architectural checkpoints**:

* **Tier 1 — At the Ingestion Stage (Atmospheric & Radiometric Normalization):**  
  Incoming raw tiles undergo pixel-level cloud detection (`s2cloudless`), directional cloud-shadow ray-tracing, and validity masking (`bad_mask`). Dynamic percentile contrast stretching (0.5%–99.5%) is calculated strictly over clean ground pixels, neutralizing lighting inconsistencies and ephemeral haze before vectors or indices are calculated.

* **Tier 2 — At the Change Detection Stage (Mask-Aware Delta Verification):**  
  Rather than performing naive pixel subtraction (which flags seasonal vegetation blooming or slight shadows as construction), the change detection engine applies per-pixel bad-mask exclusion and multi-spectral index delta verification (Delta NDVI, Delta NDBI, Delta NDWI). Confounding atmospheric pixels are masked out, and only persistent physical/structural transformations are retained.

---

## 3. Ingestion Pipeline

### 3.1 Pipeline Overview & Workflow
The Ingestion Pipeline (`backend/ingestion/`) converts raw satellite rasters into clean, uniformly tiled, atmospherically calibrated, and atomically indexed 512×512 patches ready for semantic retrieval and change detection:

1. **Dual Entry Point Ingestion:** Ingests via interactive AOI polygon queries (streaming multi-spectral bands via STAC) or direct local GeoTIFF files/folders for 100% offline evaluation.
2. **Multi-Sensor Band Harmonization:** Automatically maps spectral channels (`blue`, `green`, `red`, `nir`, `swir`) across Sentinel-2, Landsat, and Maxar, dynamically reprojecting rasters to `EPSG:4326` using `WarpedVRT`.
3. **Tier-1 False-Alarm Suppression:** Executes machine-learning cloud detection (`s2cloudless`) and directional shadow ray-tracing, producing a combined per-pixel boolean validity mask (`bad_mask`) saved as `{tile_id}_mask.tif` and computing per-tile `cloud_pct` and `quality_confidence`.
4. **Dual Normalization Architecture:**
   - **Adaptive Percentile Stretching (0.5%–99.5%):** Applied strictly over clean ground pixels to generate clear visual RGB thumbnails (`{tile_id}_thumb.jpg`) for analyst inspection and RemoteCLIP embedding.
   - **Fixed-Scale Surface Reflectance (DN / 10,000):** Retains raw physical reflectance [0.0, 1.5] in multi-band GeoTIFFs (`{tile_id}.tif`) for mathematically sound inter-temporal change detection.
5. **Clean Spectral Indices:** Calculates NDVI, NDWI, and NDBI strictly on non-cloud, non-shadow pixels (`~bad_mask`), recording `mean_ndvi`, `mean_ndwi`, and `mean_ndbi`.
6. **RemoteCLIP Embedding & Qdrant Upsert:** Encodes RGB visual tiles into 512-dimensional vectors with rich metadata payloads (`tile_id`, `site_key`, coordinates, date, spectral means, sensor) and upserts them to Qdrant without full index rebuilds.
7. **Spatial DB Registration:** Persists footprint geometries, file paths, and scene metadata to PostgreSQL + PostGIS, linked to Qdrant vectors via deterministic `tile_id`.
8. **Downstream Pair Preparation:** The saved `{tile_id}.tif` and `{tile_id}_mask.tif` directly feed downstream change analysis (`resolve_change_pair()`), where bad pixels from both dates are combined (`mask_before | mask_after`) to eliminate atmospheric false alarms.

### 3.2 Problem Statement Requirements Resolved
This pipeline directly closes:
* **Ingestion Process:** Standardized handling of heterogeneous multi-spectral satellite imagery.
* **Tier-1 False-Alarm Suppression (PS 2.2.3):** Eliminating cloud, shadow, and illumination noise at the pixel level.
* **Scale, Incremental Ingestion & Sovereignty (PS 2.2.6):**
  - High-dimensional vector (Qdrant) and spatial (PostGIS) indexing.
  - **Incremental Ingestion:** Adding new satellite acquisitions on-the-fly without rebuilding the index. *(See [Index-Build & Incremental Ingestion Procedure (`docs/incremental_ingestion_procedure.md`)](docs/incremental_ingestion_procedure.md)).*
  - **100% On-Premises Air-Gapped Operation:** Zero external cloud or API dependencies.
  - **Standard Formats & Georeferencing:** Preserves native CRS transforms in LZW-compressed GeoTIFFs.

👉 **Complete Technical Architecture & Code Reference:**  
For in-depth mathematical formulations, code flows, and schemas, refer to the dedicated specification:  
**[Detailed Ingestion Pipeline Specification (`docs/ingestionPipeline.md`)](docs/ingestionPipeline.md)**

---

## 4. Semantic Retrieval Pipeline

### 4.1 Pipeline Overview & Workflow
The Semantic Retrieval Pipeline (`backend/services/vector_search.py`, `backend/services/encoder.py`, `backend/api/routers/search.py`) allows defense analysts to search satellite archives by meaning rather than coordinates:

1. **Dual Query Modalities:** Supports natural language descriptive text queries (e.g. *"runways near river"*) and multimodal image-to-image visual similarity search (uploading an exemplar satellite image patch).
2. **Unified Vision-Language Embedding:** Both query modalities are processed through the fine-tuned `RemoteCLIP ViT-B-32` foundation model, producing a normalized 512-dimensional vector in the same latent metric space as the archive tiles.
3. **Compound High-Dimensional Vector Search ($k$-NN):** Matches the query vector against Qdrant collections (`tile_embeddings` and `maxar_tile_embeddings`) using Cosine similarity. Filters are applied dynamically during graph traversal:
   - Spatial Area of Interest (AOI polygon or bounding box pre-resolved via PostGIS `ST_Intersects`).
   - Temporal acquisition date ranges.
   - Sensor selection (`Sentinel-2`, `Maxar WorldView`, or `All`).
   - Quality confidence gates (`quality_confidence >= min_quality`) and cloud thresholds (`cloud_pct <= max_cloud_pct`).
4. **PostgreSQL Hydration & Spatial Deduplication:** Hydrates Top-$K$ candidate `tile_id`s in a single SQL query from PostgreSQL `tiles`. Applies spatial deduplication (~200m radius and `site_key`) so returned candidates represent distinct physical ground locations rather than overlapping crops.
5. **Multi-Spectral Ground Truth & Deep Inspection:** Every retrieved candidate provides coordinates, capture timestamp, sensor provenance, cloud percentage, quality confidence score, true footprint geometry, and multi-spectral indices (NDVI, NDWI, NDBI).
6. **Automated Tactical Explanations:** Evaluates compound spectral distributions to synthesize human-interpretable terrain assessments (e.g., classifying airfields, urban cores, riparian wetlands, or natural forests).

### 4.2 Problem Statement Requirements Resolved
This pipeline directly closes **PS Section 2.2.1 (Semantic and Multimodal Retrieval)**:
* **Free-Text Natural Language Search:** Zero-shot text search over imagery tiles without requiring prior coordinates.
* **Multimodal Image-to-Image Retrieval:** Querying by an exemplar satellite image patch.
* **Rank-Ordered Candidates:** Ranked strictly by mathematical cosine similarity relevance scores.
* **Compound Multi-Parameter Filtering:** Combining spatial AOI, date windows, sensor platforms, and quality thresholds.

👉 **Complete Technical Architecture & Code Reference:**  
For mathematical formulations, encoder mechanics, compound filtering rules, and API contracts, see:  
**[Detailed Semantic Retrieval Pipeline Specification (`docs/semantic_retrieval.md`)](docs/semantic_retrieval.md)**

---

## 5. Unsupervised Clustering & Cross-Site Discovery Subsystem

### 5.1 Subsystem Overview & Workflow
The Clustering and Discovery subsystem (`backend/jobs/run_clustering.py`, `backend/services/discovery.py`, `backend/api/routers/discovery.py`) organizes the entire image archive into unsupervised semantic groupings and enables analysts to branch out from any tile of interest:

1. **Periodic Background Clustering Job (`run_clustering_job`):**
   - Scrolls all 512-dimensional tile vectors across both Sentinel-2 and Maxar Qdrant collections.
   - Normalizes embeddings to unit length (L2) so Euclidean distance corresponds directly to Cosine similarity.
   - Executes **HDBSCAN** density clustering to partition tiles into cohesive geographic/infrastructure clusters without human supervision. Outlier/noise points receive label `-1` and a `NULL` cluster ID.
   - Elects the **medoid** (the actual tile vector closest to the cluster's geometric centroid) as the visual representative.
   - Analyzes PostgreSQL multi-spectral index averages (average NDVI, NDWI, NDBI) across member tiles to synthesize human-readable tactical landscape labels (e.g. *"Dense Woodland & Forest Canopy"*, *"Industrial / Tarmac Logistics"*).
   - Atomically updates `tiles.cluster_id` and the `clusters` table in PostgreSQL and updates Qdrant point payloads (`cluster_id`).

2. **Single-Click Analyst Discovery ("Find Similar"):**
   - When an analyst discovers a tile of interest on the interactive map or from a search result, the imagery patch is already indexed in the archive.
   - Clicking **"Find Similar"** invokes `GET /api/v1/discover/{tile_id}`.
   - The service pulls the seed tile's precomputed 512-D vector directly from Qdrant by UUID. **No prompt engineering and no neural network re-encoding** are required.
   - Executes global $k$-NN search in Qdrant across all regions, filters out the seed tile itself, and hydrates top-$k$ visual and semantic "twin" tiles from PostgreSQL with true footprints and spectral metrics.

3. **Instant Cluster Member Exploration:**
   - To inspect all tiles belonging to an identified cluster, the system executes an index-accelerated SQL query:
     `SELECT tile_id FROM tiles WHERE cluster_id = %s;`
   - Returns all matching imagery tiles across diverse regions and dates without constructing complex manual search filters.

### 5.2 Problem Statement Requirements Resolved
This subsystem directly closes **PS Section 2.2.4 (Discovery and Clustering)**:
* **Unsupervised Site Clustering:** Embedding-based clustering of similar terrain, geographic, and infrastructure installations across the entire archive using HDBSCAN.
* **Rapid Cross-Site Discovery:** Allows an analyst who identifies a single location of interest to branch out and discover other locations with comparable visual or semantic characteristics across the archive with zero manual query engineering.
* **Instant Group Browsing:** Direct SQL hydration of cluster members without search latency.
* **Tactical Explainability:** Grounded multi-spectral labeling of cluster categories from member band indices.

👉 **Complete Technical Architecture & Code Reference:**  
For algorithmic formulations, medoid selection mechanics, API schemas, and SQL workflows, refer to the dedicated specification:  
**[Detailed Clustering & Cross-Site Discovery Specification (`docs/Clustering_Discovery.md`)](docs/Clustering_Discovery.md)**

---

## 6. Multi-Temporal Change Detection Pipeline(Working)

### 6.1 Subsystem Status & Active Development
> [!NOTE]
> **Active Implementation in Progress:**  
> Development, mathematical calibration, and operational benchmarking for the **Multi-Temporal Change Detection Pipeline** are currently underway.  
> Whoever is concerned with this subsystem must document the end-to-end implementation, algorithms, false-alarm suppression logic, and API contracts in the dedicated specification document:  
> 👉 **[Multi-Temporal Change Detection Specification (`docs/Multi_temporal_change_detection.md`)](docs/Multi_temporal_change_detection.md)**

### 6.2 Scope & Requirements to be Closed
Once completed and benchmarked, this pipeline will directly resolve:
* **PS Section 2.2.2 (Multi-Temporal Change Analysis):**
  - Identification of physical, structural, and environmental changes (appearance, disappearance, expansion, contraction) across user-specified AOIs and temporal windows.
  - Classification of detected changes into tactical defense categories: construction, land clearance, water-extent variation, and road development.
  - Estimation of earliest available observation date via backward temporal trajectory traversal.
* **PS Section 2.2.3 (False-Alarm Suppression & Quality Handling — Tier 2):**
  - Elimination of false positives caused by seasonal vegetation cycles, sun angle, illumination variations, clouds, and moving shadows.
  - Enforcement of the pixel-level bad-mask union (`bad_mask_before | bad_mask_after`) and minimum clean ground area thresholds (>= 30%).

---

## 7. Analyst Workflow and Provenance Subsystem

### 7.1 Subsystem Role & Critical Prerequisite
> [!IMPORTANT]
> **Implementation Prerequisite:**  
> This subsystem operates directly on the output of the **Multi-Temporal Change Detection Pipeline**.  
> The core prerequisite is that change detection algorithms must actively write detected changes into the PostgreSQL `change_events` table (`event_id`, `site_key`, `tile_before`, `tile_after`, `confidence`, `change_type`, `earliest_visible_date`).  
> Once `change_events` is actively populated, this subsystem provides the human-in-the-loop triage, audit trail, and intelligence export engine.

### 7.2 Subsystem Overview & Planned Workflow (Working)
The Analyst Workflow and Provenance Subsystem provides the operational decision-support interface for military commanders and intelligence analysts:

1. **Ranked Review Queue (`review_items`):**
   - Automatically ingests candidate alerts from `change_events`.
   - Ranks alerts strictly by confidence and quality score (`rank_score = confidence * quality_confidence`), ensuring critical high-threat developments (e.g., runway or fortified position construction) appear at the top of the analyst's queue.
2. **Synchronized Before-and-After Evidence Viewer:**
   - Provides a coordinate-locked dual-pane or swipe viewer displaying pre-event ($T_1$) and post-event ($T_2$) visual thumbnails and physical reflectance GeoTIFFs side-by-side.
   - Panning and zooming are spatially synchronized across both temporal panes.
   - Displays real-time delta spectral indices (Delta NDVI, Delta NDBI, Delta NDWI), capture timestamps, sensor platforms, and bad-pixel validity masks.
3. **Immutable Decision Audit Trail (`review_audit_log`):**
   - Enables analysts to take verifiable operational action: **[ Confirm Change ]** or **[ Reject False Alarm ]**.
   - Permanently logs every decision into an append-only audit ledger with the reviewing officer's ID, decision timestamp, and tactical rationale.
   - Decisions cannot be altered or purged, establishing strict military chain of custody.
4. **Downstream Feedback Integration:**
   - Rejections of seasonal false positives (e.g. agricultural harvesting) feed into the reranking engine to suppress similar false alarms in subsequent detection passes.
   - Confirmed military developments elevate priority scoring for alerts detected within the same geographic cluster (`tiles.cluster_id`).
5. **Provenance-Preserving Intelligence Dossier Export (`exports`):**
   - Generates briefing-ready intelligence reports in **OGC GeoJSON**, **PDF Dossier**, and **CSV** formats.
   - Guarantees complete chain of custody by embedding raw satellite scene IDs (`scene_id`, `mosaicked_scenes`), acquisition parameters, algorithm version, and officer sign-off history in every exported alert.

### 7.3 Problem Statement Requirements Resolved
Upon completion, this subsystem directly closes **PS Section 2.2.5 (Analyst Workflow and Provenance)**:
* **Ranked Review Queue:** Prioritized queue featuring synchronized before-and-after evidence, coordinates, timestamps, sensor metadata, and confidence metrics.
* **Immutable Audit Trail:** Confirmation/rejection of candidate alerts with permanent logging.
* **Feedback Loop:** Integration of analyst judgments for downstream result refinement and suppression.
* **Complete Lineage Preservation:** Complete source-scene and processing history preserved in exported intelligence reports.

👉 **Complete Engineering Implementation Plan & Schemas:**  
For step-by-step developer instructions, database migration DDL, Pydantic models, and REST API routes, see:  
**[Analyst Workflow & Provenance Implementation Blueprint (`docs/analyst_workflow_provenance.md`)](docs/analyst_workflow_provenance.md)**

---

## 8. Problem Statement Coverage Summary & Provenance Records

### 8.1 End-to-End Problem Statement Closure
This project documentation comprehensively maps and addresses all mandates stipulated across **Ministry of Defence (MoD) / Indian Army Problem Statement SIH-26227**:
- **PS 2.1:** Background & Operational Challenge (Metadata Blindspot & Multi-Temporal False Alarms).
- **PS 2.2.1:** Semantic and Multimodal Retrieval *(Closed in Section 4 & [`docs/semantic_retrieval.md`](docs/semantic_retrieval.md))*.
- **PS 2.2.2 & 2.2.3:** Multi-Temporal Change Analysis & Tier-2 False-Alarm Suppression *(In Active Development in Section 6 & [`docs/Multi_temporal_change_detection.md`](docs/Multi_temporal_change_detection.md))*.
- **PS 2.2.4:** Discovery & Unsupervised Clustering *(Closed in Section 5 & [`docs/Clustering_Discovery.md`](docs/Clustering_Discovery.md))*.
- **PS 2.2.5:** Analyst Workflow & Provenance *(Planned Implementation Blueprint in Section 7 & [`docs/analyst_workflow_provenance.md`](docs/analyst_workflow_provenance.md))*.
- **PS 2.2.6:** Scale, Incremental Ingestion & Sovereignty *(Closed in Section 3, [`docs/ingestionPipeline.md`](docs/ingestionPipeline.md), and [`docs/incremental_ingestion_procedure.md`](docs/incremental_ingestion_procedure.md))*.
- **PS 2.2.7 & 2.3:** Air-Gapped Sovereign Operation, Checkpoint Declarations & Submission Deliverables *(Addressed below)*.

### 8.2 Model & Dataset Provenance Registry
As explicitly mandated in **PS Section 2.2.7** (*"Pretrained public models are permitted only if their origin, license, and checkpoints are explicitly declared and packaged for offline use"*) and **PS Section 2.3** (*"Model and dataset provenance records"*):

👉 **Official Provenance Records & Checkpoints Registry:**  
**[Model & Dataset Provenance Records (`PROVENANCE.md`)](PROVENANCE.md)**

> [!TIP]
> **Notice to Contributors & Reviewers:**  
> Please continuously update and append any newly introduced weights checkpoints or dataset sources directly to [`PROVENANCE.md`](PROVENANCE.md), citing the specific Problem Statement requirement line justifying its addition.


