"""
backend/ingestion/db_writer.py
==============================
Phase 1.6 & 1.7: Database Schema Migration & Postgres Ingestion Engine
======================================================================
PS Sections: 2.2.3, 2.2.6 (Sovereign Storage & Quality Persistence)

Inserts and upserts:
  1. Scene rows (`scenes` table) — once per scene
  2. Tile rows (`tiles` table) — with real geometry, indices (NDVI, NDWI, NDBI), and quality scores
  3. Coverage status (`ingestion_coverage` table) — tracking region completion
"""

import json
import logging
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
import psycopg2
import psycopg2.extras
from shapely.geometry import mapping, Polygon, MultiPolygon

from backend.ingestion.tiler import TileCandidate

logger = logging.getLogger(__name__)

PG_HOST = os.getenv("POSTGRES_HOST", "localhost")
PG_PORT = os.getenv("POSTGRES_PORT", "5434")
PG_DB = os.getenv("POSTGRES_DB", "eo_archive")
PG_USER = os.getenv("POSTGRES_USER", "eo_admin")
PG_PASSWORD = os.getenv("POSTGRES_PASSWORD", "eo_password")
PG_DSN = f"host={PG_HOST} port={PG_PORT} dbname={PG_DB} user={PG_USER} password={PG_PASSWORD}"


def get_pg_connection():
    return psycopg2.connect(PG_DSN)


def ensure_schema_migrated(conn=None):
    """
    Applies non-destructive schema migrations for Phase 1.6 columns if missing.
    """
    should_close = False
    if conn is None:
        conn = get_pg_connection()
        should_close = True

    try:
        with conn.cursor() as cur:
            migration_sql = """
            ALTER TABLE tiles ADD COLUMN IF NOT EXISTS thumbnail_path TEXT;
            ALTER TABLE tiles ADD COLUMN IF NOT EXISTS bad_mask_path TEXT;
            ALTER TABLE tiles ADD COLUMN IF NOT EXISTS pixel_scale TEXT DEFAULT 'reflectance_fixed_10000';
            ALTER TABLE tiles ADD COLUMN IF NOT EXISTS band_order JSONB;
            ALTER TABLE tiles ADD COLUMN IF NOT EXISTS band_stats JSONB;
            ALTER TABLE tiles ADD COLUMN IF NOT EXISTS mean_ndvi FLOAT;
            ALTER TABLE tiles ADD COLUMN IF NOT EXISTS mean_ndwi FLOAT;
            ALTER TABLE tiles ADD COLUMN IF NOT EXISTS mean_ndbi FLOAT;
            ALTER TABLE tiles ADD COLUMN IF NOT EXISTS source_type TEXT DEFAULT 'aoi_search';
            ALTER TABLE tiles ADD COLUMN IF NOT EXISTS mosaicked_scenes JSONB;

            CREATE INDEX IF NOT EXISTS tiles_ndvi_idx ON tiles (mean_ndvi);
            CREATE INDEX IF NOT EXISTS tiles_ndbi_idx ON tiles (mean_ndbi);

            CREATE TABLE IF NOT EXISTS clusters (
              cluster_id               TEXT PRIMARY KEY,
              label                    TEXT,
              representative_tile_id   TEXT,
              tile_count               INT DEFAULT 0,
              computed_at              TIMESTAMP DEFAULT now(),
              model_version            TEXT
            );
            """
            cur.execute(migration_sql)
            conn.commit()
            logger.info("[Phase 1.6] Postgres schema verified & migrated.")
    finally:
        if should_close:
            conn.close()


def upsert_scene(
    scene_id: str,
    source: str,
    acquisition_date: str,
    crs: str,
    footprint_geom: Optional[Any] = None,
    raw_file_path: Optional[str] = None,
    provenance: Optional[Dict[str, Any]] = None,
    conn=None
) -> None:
    """Inserts or updates a satellite scene record."""
    should_close = False
    if conn is None:
        conn = get_pg_connection()
        should_close = True

    try:
        with conn.cursor() as cur:
            # Parse acquisition date
            dt = datetime.fromisoformat(str(acquisition_date).replace("Z", "+00:00"))
            
            # WKT footprint if shapely geometry provided
            footprint_wkt = footprint_geom.wkt if hasattr(footprint_geom, "wkt") else "POLYGON((-180 -90, 180 -90, 180 90, -180 90, -180 -90))"

            sql = """
            INSERT INTO scenes (scene_id, source, acquisition_date, crs, footprint, raw_file_path, provenance)
            VALUES (%s, %s, %s, %s, ST_GeomFromText(%s, 4326), %s, %s)
            ON CONFLICT (scene_id) DO UPDATE SET
                source = EXCLUDED.source,
                acquisition_date = EXCLUDED.acquisition_date,
                crs = EXCLUDED.crs,
                footprint = EXCLUDED.footprint,
                raw_file_path = EXCLUDED.raw_file_path,
                provenance = EXCLUDED.provenance;
            """
            cur.execute(
                sql,
                (
                    scene_id,
                    source,
                    dt,
                    crs,
                    footprint_wkt,
                    raw_file_path or "",
                    json.dumps(provenance or {})
                )
            )
            conn.commit()
            logger.info(f"[Phase 1.7] Scene '{scene_id}' upserted into Postgres.")
    finally:
        if should_close:
            conn.close()


def upsert_tiles(
    tiles: List[TileCandidate],
    scene_id: str,
    acquisition_date: str,
    sensor: str = "Sentinel-2",
    base_data_dir: str = "data",
    region_id: str = "custom_region",
    mosaicked_scenes: Optional[List[str]] = None,
    conn=None
) -> int:
    """
    Batch inserts or updates tile records with all real computed spectral indices, quality flags, and mosaicked scene provenance.
    """
    if not tiles:
        return 0

    should_close = False
    if conn is None:
        conn = get_pg_connection()
        should_close = True

    ensure_schema_migrated(conn)

    try:
        dt = datetime.fromisoformat(str(acquisition_date).replace("Z", "+00:00"))
        date_folder = dt.strftime("%Y-%m-%d")
        mosaicked_scenes_json = json.dumps(mosaicked_scenes or [scene_id])

        with conn.cursor() as cur:
            sql = """
            INSERT INTO tiles (
                tile_id, scene_id, site_key, geometry,
                centroid_lat, centroid_lon, acquisition_date, sensor,
                cloud_pct, quality_confidence, file_path, thumbnail_path,
                bad_mask_path, pixel_scale,
                band_order, band_stats, mean_ndvi, mean_ndwi, mean_ndbi, source_type,
                mosaicked_scenes
            ) VALUES (
                %s, %s, %s, ST_GeomFromText(%s, 4326),
                %s, %s, %s, %s,
                %s, %s, %s, %s,
                %s, %s,
                %s, %s, %s, %s, %s, %s, %s
            )
            ON CONFLICT (tile_id) DO UPDATE SET
                site_key = EXCLUDED.site_key,
                geometry = EXCLUDED.geometry,
                centroid_lat = EXCLUDED.centroid_lat,
                centroid_lon = EXCLUDED.centroid_lon,
                acquisition_date = EXCLUDED.acquisition_date,
                sensor = EXCLUDED.sensor,
                cloud_pct = EXCLUDED.cloud_pct,
                quality_confidence = EXCLUDED.quality_confidence,
                file_path = EXCLUDED.file_path,
                thumbnail_path = EXCLUDED.thumbnail_path,
                bad_mask_path = EXCLUDED.bad_mask_path,
                pixel_scale = EXCLUDED.pixel_scale,
                band_order = EXCLUDED.band_order,
                band_stats = EXCLUDED.band_stats,
                mean_ndvi = EXCLUDED.mean_ndvi,
                mean_ndwi = EXCLUDED.mean_ndwi,
                mean_ndbi = EXCLUDED.mean_ndbi,
                source_type = EXCLUDED.source_type,
                mosaicked_scenes = EXCLUDED.mosaicked_scenes;
            """

            records = []
            for t in tiles:
                tif_path = os.path.abspath(os.path.join(base_data_dir, "tiles", region_id, date_folder, f"{t.tile_id}.tif"))
                jpg_path = os.path.abspath(os.path.join(base_data_dir, "tiles", region_id, date_folder, f"{t.tile_id}_thumb.jpg"))
                mask_path = getattr(t, "bad_mask_path", None) or os.path.abspath(os.path.join(base_data_dir, "tiles", region_id, date_folder, f"{t.tile_id}_mask.tif"))
                pixel_scale = getattr(t, "pixel_scale", "reflectance_fixed_10000")

                records.append((
                    t.tile_id,
                    scene_id,
                    t.site_key,
                    t.footprint_geom.wkt,
                    t.centroid_lat,
                    t.centroid_lon,
                    dt,
                    sensor,
                    t.cloud_pct,
                    t.quality_confidence,
                    tif_path,
                    jpg_path,
                    mask_path,
                    pixel_scale,
                    json.dumps(t.band_order),
                    json.dumps(t.band_stats),
                    t.mean_ndvi,
                    t.mean_ndwi,
                    t.mean_ndbi,
                    t.source_type,
                    mosaicked_scenes_json
                ))

            psycopg2.extras.execute_batch(cur, sql, records, page_size=100)
            conn.commit()
            logger.info(f"[Phase 1.7] Successfully upserted {len(records)} tile(s) into Postgres `tiles`.")
            return len(records)
    finally:
        if should_close:
            conn.close()


def update_coverage_status(
    region_id: str,
    region_name: str,
    aoi_geom: Optional[Any],
    tile_count: int,
    status: str = "done",
    conn=None
) -> None:
    """Updates the ingestion_coverage table."""
    should_close = False
    if conn is None:
        conn = get_pg_connection()
        should_close = True

    try:
        with conn.cursor() as cur:
            geom_wkt = aoi_geom.wkt if hasattr(aoi_geom, "wkt") else "POLYGON((-180 -90, 180 -90, 180 90, -180 90, -180 -90))"
            sql = """
            INSERT INTO ingestion_coverage (region_id, region_name, geometry, status, tile_count, last_updated)
            VALUES (%s, %s, ST_GeomFromText(%s, 4326), %s, %s, now())
            ON CONFLICT (region_id) DO UPDATE SET
                region_name = EXCLUDED.region_name,
                geometry = EXCLUDED.geometry,
                status = EXCLUDED.status,
                tile_count = ingestion_coverage.tile_count + EXCLUDED.tile_count,
                last_updated = now();
            """
            cur.execute(sql, (region_id, region_name, geom_wkt, status, tile_count))
            conn.commit()
            logger.info(f"[Phase 1.7] Updated ingestion_coverage for region '{region_id}' (Total Tiles Added: {tile_count}).")
    finally:
        if should_close:
            conn.close()


update_coverage = update_coverage_status
