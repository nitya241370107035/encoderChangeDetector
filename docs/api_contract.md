# 🌐 AeroLens REST API Contract Specification

This document provides the complete API specification for the AeroLens backend service (`/api/v1`), detailing available endpoints, HTTP methods, request bodies, query parameters, response structures, and status codes.

---

## 1. System Health & Readiness

### `GET /health`
Returns the operational health status of the FastAPI backend service.

- **Request**: None
- **Response**: `200 OK`
```json
{
  "status": "ok"
}
```

---

## 2. Ingestion & Preprocessing Pipeline

### `POST /api/v1/ingest/aoi`
Triggers the multi-temporal ingestion pipeline for a user-drawn AOI polygon across requested temporal intervals.

- **Request Body**: `application/json`
```json
{
  "aoi_geometry": {
    "type": "Polygon",
    "coordinates": [
      [
        [77.10, 28.50],
        [77.30, 28.50],
        [77.30, 28.70],
        [77.10, 28.70],
        [77.10, 28.50]
      ]
    ]
  },
  "start_date": "2023-01-01",
  "end_date": "2024-01-01",
  "num_buckets": 2,
  "max_cloud_cover": 15.0,
  "region_name": "Delhi_NCR_Sector"
}
```
- **Response**: `200 OK`
```json
{
  "status": "success",
  "region_id": "region_28.60_77.20",
  "scenes_processed": 2,
  "tiles_generated": 16,
  "elapsed_seconds": 12.4
}
```

---

### `POST /api/v1/ingest/file`
Ingests an offline local GeoTIFF raster file with custom band mappings and georeferenced coordinates.

- **Request Body**: `application/json`
```json
{
  "file_path": "/app/data/sample_scene.tif",
  "source_name": "Local_Aerial_Survey",
  "band_order": ["B02", "B03", "B04", "B08", "B11"]
}
```
- **Response**: `200 OK`
```json
{
  "status": "success",
  "tiles_generated": 8,
  "tile_ids": ["tile_00001", "tile_00002"]
}
```

---

## 3. Geographic Coverage & Slices

### `GET /api/v1/coverage`
Fetches a GeoJSON `FeatureCollection` of all ingested geographic regions currently cataloged in the spatial database.

- **Request**: None
- **Response**: `200 OK`
```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [...]
      },
      "properties": {
        "region_id": "region_28.60_77.20",
        "region_name": "Delhi_NCR_Sector",
        "status": "done",
        "tile_count": 16,
        "last_updated": "2026-09-12T18:30:00Z"
      }
    }
  ]
}
```

---

## 4. Multimodal Semantic Retrieval & Vector Search

### `POST /api/v1/search/semantic`
Performs zero-shot natural language vector search against tile embeddings using RemoteCLIP ViT-B-32 with compound filtering.

- **Request Body**: `application/json`
```json
{
  "query": "airstrip runway with visible hangars near water",
  "top_k": 5,
  "confidence_threshold": 0.20,
  "sensor": "all",
  "start_date": "2020-01-01",
  "end_date": "2026-12-31",
  "max_cloud_cover": 20.0,
  "min_ndvi": null,
  "max_ndvi": null,
  "bbox": [72.50, 23.00, 73.00, 23.50]
}
```
- **Response**: `200 OK`
```json
{
  "query": "airstrip runway with visible hangars near water",
  "total_matches": 5,
  "execution_time_ms": 42.1,
  "results": [
    {
      "tile_id": "maxar_region_36.92_-76.25_2026_9_3",
      "score": 0.884,
      "sensor": "Maxar WorldView / Wayback",
      "acquisition_date": "2026-08-05T00:00:00",
      "centroid_lat": 36.9349,
      "centroid_lon": -76.2542,
      "thumbnail_url": "/data/tiles/region_36.92_-76.25/2026-08-05/thumb.jpg",
      "mean_ndvi": 0.002,
      "mean_ndwi": null,
      "mean_ndbi": null,
      "quality_confidence": 0.95,
      "spot_description": "High-resolution optical reconnaissance imagery from Maxar archive."
    }
  ]
}
```

---

### `POST /api/v1/search/image`
Uploads a reference satellite image to perform visual similarity search (image-to-image retrieval) across the vector archive.

- **Request**: `multipart/form-data`
  - `file`: Image file (`.jpg`, `.png`, `.tif`)
  - `top_k`: Integer (default: `5`)
  - `confidence_threshold`: Float (default: `0.20`)
  - `sensor`: String (default: `"all"`)
- **Response**: `200 OK` (Same schema as `/search/semantic`)

---

### `GET /api/v1/search/health`
Checks whether the RemoteCLIP encoder and Qdrant vector collection are loaded and responsive.

- **Response**: `200 OK`
```json
{
  "status": "healthy",
  "encoder_device": "cpu",
  "model_name": "RemoteCLIP-ViT-B-32",
  "qdrant_status": "connected",
  "collection": "tile_embeddings",
  "total_vectors": 348
}
```

---

## 5. Tile Catalog & Archive Statistics

### `GET /api/v1/archive/stats`
Returns system-wide operational summary statistics.

- **Response**: `200 OK`
```json
{
  "total_scenes": 12,
  "total_tiles": 348,
  "total_regions": 4,
  "total_clusters": 10
}
```

---

### `GET /api/v1/archive/tiles`
Queries paginated catalog of tiles with multi-spectral properties.

- **Query Parameters**:
  - `region_id` (optional): Filter by region string
  - `date` (optional): Filter by ISO date
  - `limit` (default: `50`): Max results to return
- **Response**: `200 OK` (Array of tile objects with centroid coordinates, spectral means, and thumbnail paths).

---

## 6. Unsupervised HDBSCAN Clustering & Discovery

### `GET /api/v1/clusters`
Returns all discovered terrain and infrastructure clusters generated by unsupervised density partitioning.

- **Response**: `200 OK`
```json
[
  {
    "cluster_id": "cluster_20260912223054_10",
    "label": "Cluster 10 — Open Arid & Transition Terrain (NDVI 0.02)",
    "representative_tile_id": "maxar_region_36.92_-76.25_2026_9_3",
    "tile_count": 54,
    "computed_at": "2026-09-12T22:30:54Z",
    "model_version": "RemoteCLIP-ViT-B-32",
    "thumbnail_url": "/data/tiles/region_36.92_-76.25/2026-08-05/thumb.jpg",
    "centroid_lat": 36.9349,
    "centroid_lon": -76.2542,
    "mean_ndvi": 0.0019,
    "sensor": "Maxar WorldView / Wayback"
  }
]
```

---

### `GET /api/v1/clusters/{cluster_id}/tiles`
Retrieves all member tiles belonging to a specific cluster ID.

- **Query Parameters**:
  - `limit` (default: `60`): Maximum member tiles to return
- **Response**: `200 OK` (Array of tile metadata objects belonging to the cluster).

---

### `POST /api/v1/clusters/recompute`
Triggers an asynchronous re-computation of HDBSCAN clustering over all vector embeddings currently present in Qdrant.

- **Response**: `200 OK`
```json
{
  "status": "success",
  "clusters_count": 10,
  "total_points": 348,
  "elapsed_seconds": 2.14
}
```

---

## 7. Persistent Chat System (ChatGPT/Gemini Session Workflows)

### `GET /api/v1/chat/conversations`
Lists all conversations associated with a specific user ID, sorted by most recently active.

- **Query Parameters**:
  - `user_id` (required): Unique user session identifier (e.g., `default_analyst`)
- **Response**: `200 OK`
```json
[
  {
    "conversation_id": "conv_9f3b12a8-382a-4a7b-891d-b5e7d21c4310",
    "user_id": "default_analyst",
    "title": "Airstrip search near coastal waters",
    "message_count": 4,
    "created_at": "2026-09-13T03:15:00Z",
    "updated_at": "2026-09-13T03:22:10Z"
  }
]
```

---

### `POST /api/v1/chat/conversations`
Creates a brand-new persistent conversation session.

- **Request Body**: `application/json`
```json
{
  "user_id": "default_analyst",
  "title": "New Semantic Retrieval Session"
}
```
- **Response**: `200 OK` (Returns the newly initialized conversation object).

---

### `GET /api/v1/chat/conversations/{conversation_id}`
Fetches the full message trajectory, query parameters, attached images, and retrieved tile results for an existing conversation.

- **Response**: `200 OK`
```json
{
  "conversation_id": "conv_9f3b12a8...",
  "user_id": "default_analyst",
  "title": "Airstrip search near coastal waters",
  "messages": [
    {
      "message_id": 1,
      "role": "user",
      "content": "find runways and airstrips",
      "attached_image_name": null,
      "attached_image_preview": null,
      "created_at": "2026-09-13T03:15:10Z"
    },
    {
      "message_id": 2,
      "role": "assistant",
      "content": "Retrieved 5 candidate satellite tiles matching your query.",
      "query_context": { "top_k": 5, "confidence": 0.20 },
      "results": [ ... ],
      "created_at": "2026-09-13T03:15:12Z"
    }
  ]
}
```

---

### `PATCH /api/v1/chat/conversations/{conversation_id}`
Renames the title of an existing conversation.

- **Request Body**: `application/json`
```json
{
  "title": "Naval Base & Anchorage Analysis"
}
```
- **Response**: `200 OK`

---

### `DELETE /api/v1/chat/conversations/{conversation_id}`
Permanently deletes an existing conversation and cascades deletion to all associated messages.

- **Response**: `200 OK`
```json
{
  "status": "success",
  "deleted_conversation_id": "conv_9f3b12a8..."
}
```

---

### `POST /api/v1/chat/conversations/{conversation_id}/messages`
Appends a new message (user query or assistant search response) to the conversation history.

- **Request Body**: `application/json`
```json
{
  "role": "user",
  "content": "show vegetation changes along river banks",
  "attached_image_name": null,
  "attached_image_preview": null,
  "query_context": { "top_k": 5 },
  "results": null
}
```
- **Response**: `200 OK` (Returns the appended message with assigned `message_id` and timestamp).

---

## 8. Multi-Temporal Change Detection

### `POST /api/v1/change/analyze`
Computes bi-temporal change metrics ($\Delta\text{NDVI}$, $\Delta\text{NDWI}$, $\Delta\text{NDBI}$, structural shift) between two acquisitions of the same tile coordinate.

- **Request Body**: `application/json`
```json
{
  "tile_id_t1": "S2A_42QZL_20200322_0_L2A_tile_00001",
  "tile_id_t2": "S2B_42QZL_20240322_0_L2A_tile_00001",
  "threshold": 0.15
}
```
- **Response**: `200 OK`
```json
{
  "event_id": 1,
  "change_magnitude": 0.342,
  "delta_ndvi": -0.28,
  "delta_ndwi": 0.05,
  "delta_ndbi": 0.22,
  "change_type": "urbanization_or_construction",
  "mask_url": "/data/tiles/change_masks/event_1_mask.png"
}
```
