# Data & Model Provenance Registry (`PROVENANCE.md`)

## 1. Overview & Operational Purpose

This document serves as the official **Data and Model Lineage Audit Log** for the *Semantic Retrieval & Multi-Temporal Change Analysis of Satellite Imagery* project (Ministry of Defence / Indian Army Problem Statement SIH-26227).

To ensure complete transparency, security compliance, and licensing auditability in a **fully on-premises and air-gapped environment**, every external pre-trained model checkpoint and satellite data source ingested into the platform is declared in this registry.

---

## 2. Pre-Trained Model Checkpoint Registry (PS 2.2.7)

As mandated by **PS Section 2.2.7**, public pre-trained models are declared with their origin, license, checkpoint path, and operational function:

| Model Name | Architecture / Version | Source Provider / Paper Citation | License | Storage Location | Operational Function |
|---|---|---|---|---|---|
| **RemoteCLIP** | `ViT-B/32` (512-dim $L_2$-normalized vector) | Liu et al., *"RemoteCLIP: A Vision Language Foundation Model for Remote Sensing"*, IEEE TGRS. Checkpoint: [HuggingFace / RemoteCLIP](https://huggingface.co/chendelong/RemoteCLIP) | MIT / Apache 2.0 | `models/retrieval/RemoteCLIP-ViT-B-32.pt` | Unified text-to-image and image-to-image semantic embedding generation. |
| **s2cloudless** | `S2PixelCloudDetector` (LightGBM 10-band model) | Sentinel Hub / Sinergise: [GitHub / sentinel2-cloud-detector](https://github.com/sentinel-hub/sentinel2-cloud-detector) | MIT License | Bundled package (`s2cloudless`) | Per-pixel cloud probability classification for Tier-1 false-alarm suppression. |

> [!NOTE]
> **Internal Algorithms (Not External Pretrained Checkpoints):**  
> Custom algorithmic components such as the **Directional Shadow Ray-Tracer** (`cloud_removal/shadow_mask.py`) and the **HDBSCAN Unsupervised Clustering Job** (`backend/jobs/run_clustering.py`) are proprietary algorithms and workflow scripts authored within this repository, not downloaded third-party deep learning model weights.

---

## 3. Satellite Imagery Data Sources

This satellite imagery is used strictly for collecting the data and serving as a baseline source of data for our ingestion, semantic retrieval, clustering, and change detection pipeline:

| Dataset / Sensor | Spatial Resolution | Bands Utilized | Provider / Access Point | Terms / License | Role in Pipeline |
|---|---|---|---|---|---|
| **Sentinel-2 MSI Level-2A** | 10m & 20m | Blue (B2), Green (B3), Red (B4), NIR (B8), SWIR1 (B11) | European Space Agency (ESA) / AWS Earth Search STAC | Copernicus Open Data Policy | Baseline multi-spectral source for tile tiling, spectral indices (NDVI, NDWI, NDBI), and change detection. |
| **Maxar WorldView** | 0.5m Pan-sharpened / 2.0m Multi-spectral | Red, Green, Blue, NIR | Maxar Open Data Program / Organiser Supplied | Maxar Open Data Attribution Non-Commercial | High-resolution optical validation and cross-sensor semantic vector search. |

---

## 4. Air-Gap Verification & Sovereign Compliance (PS 2.2.6 & 2.2.7)

To guarantee that the platform functions 100% offline without external cloud or internet connectivity:
1. **Model Weights Staging:** `RemoteCLIP-ViT-B-32.pt` weights and `s2cloudless` LightGBM models are pre-staged locally on disk before evaluation.
2. **Zero Outbound Telemetry:** No runtime network calls, external API queries, or telemetry endpoints are triggered by either model during initialization or inference.
3. **Local Container Network:** Microservices (`eo_backend`, `eo_postgres`, `eo_qdrant`, `eo_minio`) communicate exclusively across the internal Docker bridge network (`0.0.0.0:8000`).
