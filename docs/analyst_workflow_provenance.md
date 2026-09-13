# 📋 Analyst Workflow & Provenance Subsystem — Operational Blueprint

> **Target Problem Statement Mandate:** Ministry of Defence (MoD) / Indian Army Problem Statement **SIH-26227 Section 2.2.5 (Analyst Workflow and Provenance)**  
> **Document Purpose:** Detailed operational guide and implementation roadmap for the engineering team. Explains the exact conceptual mechanics, UI workflow, and data flow required to deliver this subsystem.  
> **Status:** Planned Implementation Blueprint

---

## 1. Architectural Purpose & Critical Prerequisite

In defense and intelligence operations, **no autonomous AI algorithm makes final tactical decisions**. Automated algorithms (such as Change Detection and Semantic Search) can identify statistical variations across terabytes of satellite rasters, but a human officer—the **Intelligence Analyst**—must inspect the physical evidence and make the final call.

```
                           SYSTEM DATAFLOW & PREREQUISITE
                           ──────────────────────────────

    [Satellite Imagery Ingestion]
                 │
                 ▼
    [Multi-Temporal Change Detection Pipeline]
                 │
                 ▼  CRITICAL PREREQUISITE (Must be completed first!)
    ┌──────────────────────────────────────────────────────────────────┐
    │  PostgreSQL Table: 'change_events'                               │
    │  • Contains detected physical changes:                           │
    │    (event_id, site_key, tile_before, tile_after, confidence,     │
    │     change_type, earliest_visible_date)                          │
    └─────────────────────────────────┬────────────────────────────────┘
                                      │
                                      ▼
    ════════════════════════════════════════════════════════════════════
    THIS SUBSYSTEM: ANALYST WORKFLOW & PROVENANCE (PS 2.2.5)
    ════════════════════════════════════════════════════════════════════
    1. Ranked Review Queue: Triage incoming change alerts by priority.
    2. Synchronized Evidence Viewer: Dual-pane locked inspection.
    3. Immutable Audit Trail: Confirm or Reject with officer identity.
    4. Feedback Integration: Learning from analyst decisions.
    5. Intelligence Export: Dossiers carrying 100% verifiable lineage.
    ════════════════════════════════════════════════════════════════════
```

> [!IMPORTANT]
> **CRITICAL PREREQUISITE FOR THE DEVELOPER:**  
> This subsystem is **anchored directly on Change Detection**. It cannot function in isolation.  
> The prerequisite is that the **Multi-Temporal Change Detection Pipeline** must be operational and actively storing detected change events into the PostgreSQL `change_events` table. Once that table has data, this workflow takes over to manage analyst review, verification, and reporting.

---

## 2. The 4 Core Operational Implementations (What To Build)

Here is the exact breakdown of the 4 key functional components that must be built:

---

### Implementation 1: The Ranked Review Queue (API + UI Dashboard)

#### The Problem It Solves:
When change detection runs over large border sectors, it produces dozens or hundreds of candidate alerts. If an analyst is given an unsorted list or raw GeoTIFF files, they suffer from **Alert Fatigue** and miss critical threats.

#### How It Works:
1. **Confidence-Based Prioritization:**  
   The backend retrieves all alerts marked as `status = 'pending'` from the database and orders them in descending priority based on their `confidence` and `quality_confidence` scores. High-threat, clear structural developments appear at the very top.
2. **Category & Sector Filtering:**  
   The analyst can filter the queue by:
   - Status (`pending`, `confirmed`, `rejected`)
   - Change Category (`construction`, `land_clearance`, `road_development`, `water_extent_variation`)
   - Geographic Sector / AOI or Date Window.
3. **Analyst Dashboard Interface:**  
   The UI displays a prioritized triage table:
   - **Alert #1:** `[Construction]` | Confidence: **94%** | Location: `31.4285°N, 74.3120°E` | Date: `14-Sep-2024` | Status: `Pending`
   - **Alert #2:** `[Land Clearance]` | Confidence: **86%** | Location: `32.1140°N, 75.1205°E` | Date: `12-Sep-2024` | Status: `Pending`
   - **Alert #3:** `[Water Extent]` | Confidence: **52%** | Location: `30.8520°N, 74.0510°E` | Date: `09-Sep-2024` | Status: `Pending`

* **Backend Endpoint:** `GET /api/v1/review/queue?status=pending&sort=confidence_desc`

---

### Implementation 2: Synchronized Before-and-After Evidence Viewer

#### The Problem It Solves:
An alert cannot be evaluated from a single number or summary. An analyst must inspect the pre-change baseline image and the post-change image side-by-side to visually confirm what physically occurred on the ground.

#### How It Works:
When the analyst clicks any alert row in the review queue:
1. **Backend Evidence Package Assembly:**  
   The backend fetches the complete historical evidence packet:
   - **$T_{\text{before}}$ (Baseline):** Pre-event acquisition date, RGB visual thumbnail path, multi-band GeoTIFF path, and sensor metadata.
   - **$T_{\text{after}}$ (Post-Event):** Current acquisition date, RGB visual thumbnail path, multi-band GeoTIFF path, and sensor metadata.
   - **Change Mask:** Colored pixel overlay demarcating the exact spatial footprint where physical surface changes were detected.
   - **Multi-Spectral Delta Metrics:** Physical index shifts calculated strictly over clean, non-cloud ground pixels:
     - $\Delta\text{NDVI} = -0.35$ (Significant loss of vegetative canopy)
     - $\Delta\text{NDBI} = +0.40$ (Significant increase in built-up concrete/tarmac)
     - $\Delta\text{NDWI} = -0.05$ (No moisture change)
   - **Quality & Atmospheric Provenance:** Cloud percentage, bad-pixel validity mask, and sun azimuth angle.

2. **Frontend Synchronized Inspection Screen:**  
   The frontend displays a split-screen or interactive swipe slider:
   - **Left Pane:** $T_{\text{before}}$ satellite imagery.
   - **Right Pane:** $T_{\text{after}}$ satellite imagery.
   - **Spatial Synchronization (Coordinate Lock):** Both panes are strictly locked to the identical geographical coordinates. When the analyst pans, tilts, or zooms into the left pane, the right pane moves in exact synchrony.
   - **Mask Overlay Toggle:** A button allowing the analyst to toggle the red/yellow change outline on or off over the visual imagery.

* **Backend Endpoint:** `GET /api/v1/review/{item_id}/evidence`

---

### Implementation 3: Confirm / Reject Action with Immutable Audit Logging

#### The Problem It Solves:
In military intelligence, **accountability and non-repudiation are mandatory**. Every operational confirmation or dismissal must be permanently recorded with who made the decision, when it was made, and why.

#### How It Works:
1. **Analyst Operational Actions:**  
   On the evidence screen, the analyst has two primary action buttons:
   - **[ ✅ Confirm Change ]**: Validates that the alert is a genuine tactical development (e.g., enemy infrastructure expansion, new border outpost, airstrip).
   - **[ ❌ Reject False Alarm ]**: Dismisses the alert as a non-tactical false alarm (e.g., seasonal crop harvest, legal agricultural plowing, shadow artifact).

2. **Decision Input & Operational Rationale:**  
   Clicking either button opens a concise prompt:
   - Analyst ID / Callsign: e.g., `"Captain_Rao"`
   - Decision Rationale: e.g., `"Confirmed new paved tarmac and hangar construction along northern perimeter."`

3. **Immutable Database Logging:**  
   The backend executes an atomic update:
   - Updates `review_items`: `status = 'confirmed'`, `analyst_id = 'Captain_Rao'`, `decided_at = NOW()`.
   - Appends a permanent record to `review_audit_log`:
     ```json
     {
       "item_id": 42,
       "event_id": 101,
       "analyst_id": "Captain_Rao",
       "action": "CONFIRM",
       "rationale": "Confirmed new paved tarmac and hangar construction.",
       "timestamp": "2026-09-14T03:30:00Z"
     }
     ```
   - The alert immediately moves out of the `pending` queue into the `reviewed` historical archive.
   - **Zero Deletion / Zero Tampering:** Audit records can never be overwritten, edited, or deleted.

* **Backend Endpoint:** `POST /api/v1/review/{item_id}/decide`

---

### Implementation 4: Export Intelligence Report with Full Provenance (Chain of Custody)

#### The Problem It Solves:
When intelligence reports are submitted to Military Headquarters or higher command, commanders will ask: *"What is the proof that this change is authentic? Where did this image come from, and who verified it?"* An isolated screenshot or JPEG has no legal or military standing without a verifiable chain of custody.

#### How It Works:
When an analyst clicks **"Export Intelligence Dossier"** for one or more confirmed alerts:
1. **Assembly of Complete Lineage (Janam-Kundali):**  
   The backend compiles full source-scene provenance for each confirmed alert:
   - **Raw Satellite Scene IDs:** Native satellite granule IDs (e.g., `S2A_MSIL2A_20240315...` and `S2B_MSIL2A_20240914...`).
   - **Sensor & Platform Details:** Platform name (`Sentinel-2 MSI Level-2A` or `Maxar WorldView-3`), spatial resolution (10m / 0.5m), and band mapping.
   - **Exact Acquisition Timestamps:** Date, hour, and minute of satellite passes.
   - **Processing & Algorithmic History:** Preprocessing radiometric scale (`reflectance_fixed_10000`), bad-pixel cloud mask parameters (`s2cloudless`), and change algorithm version commit (`change_pair.py v1.2`).
   - **Officer Sign-off Audit:** Reviewing officer ID, review timestamp, and verified rationale.
2. **Export Formats:**
   - **Tactical Briefing PDF:** Printable, formatted military intelligence dossier with side-by-side thumbnails, change metrics table, coordinates, and cryptographic verification hash.
   - **OGC GeoJSON:** GIS vector layer of changed polygons with all metadata and lineage fields embedded in the GeoJSON feature properties for ingestion into military GIS suites (QGIS, ArcGIS, command consoles).
   - **Tabular CSV:** Structured summary table for statistical analysis and archiving.

* **Backend Endpoint:** `POST /api/v1/export/report`

---

## 3. Feedback Loop for Downstream Refinement

PS 2.2.5 requires: *"Support feedback integration for downstream result refinement/reranking."*

How this functions in practice:
1. **False-Alarm Suppression Loop:**  
   When an analyst rejects an alert and tags it as *"Seasonal Agricultural Harvesting"*, the system logs the spectral delta profile ($\Delta\text{NDVI} < 0$ with $\Delta\text{NDBI} \approx 0$) into the database. Subsequent automated runs over similar agricultural areas in that seasonal window automatically downweight those changes, preventing repetitive false-alarm noise floods.
2. **High-Value Cluster Elevation:**  
   When an analyst confirms a strategic military asset (e.g., an ammunition depot or radar installation), the system queries Qdrant for similar tiles belonging to the same cluster (`tiles.cluster_id`). Alerts detected in those twin clusters are elevated in priority across the entire national archive.

---

## 4. Database Foundations Summary

Your teammate can leverage the existing tables in `backend/db/schema.sql`:

| Table Name | Role in this Subsystem |
|---|---|
| `change_events` | *(Prerequisite feed)* Contains pre-change and post-change tile IDs, change category, confidence, and earliest visible date. |
| `review_items` | The analyst queue holding pending, confirmed, and rejected alerts with ranking scores. |
| `review_audit_log` | The append-only, immutable record of every analyst confirmation, rejection, and rationale. |
| `exports` | The historical log of generated PDF and GeoJSON intelligence dossiers with lineage hashes. |
| `search_log` | Logs analyst queries and feedback to support reranking and false-alarm suppression. |

---

## 5. Developer Checklist for Teammate

Follow this sequence to build and verify this subsystem:

- [ ] **Step 1: Check Prerequisite Data**  
  Ensure `change_events` table has sample change records.
- [ ] **Step 2: Build the Review Service (`backend/services/review_service.py`)**  
  Implement functions to:
  - Query pending `review_items` sorted by `rank_score DESC`.
  - Assemble the synchronized before-and-after evidence packet for a given `item_id`.
  - Record Confirm/Reject decisions and append to `review_audit_log`.
- [ ] **Step 3: Build the Export Service (`backend/services/export_service.py`)**  
  Implement dossier assembly extracting raw scene IDs, sensor metadata, timestamps, and officer sign-offs into PDF and GeoJSON.
- [ ] **Step 4: Expose FastAPI Endpoints (`backend/api/routers/review.py`)**  
  Connect the service functions to REST endpoints:
  - `GET /api/v1/review/queue`
  - `GET /api/v1/review/{item_id}/evidence`
  - `POST /api/v1/review/{item_id}/decide`
  - `POST /api/v1/export/report`
- [ ] **Step 5: Frontend UI Wireup**  
  Build the review queue table and the dual-pane/swipe synchronized image viewer with coordinate lock and [Confirm] / [Reject] buttons.
