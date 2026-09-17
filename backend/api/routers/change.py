"""
backend/api/routers/change.py
=============================
Change Detection API Router:
1. /grid-map: Sovereign multi-region offline mosaic grid
2. /stage-tile: Quality Check Gate & Staging engine (drops cloudy tiles, stages clean tiles, generates manifest.json)
3. /pair: Pixel-precise change detection using per-pixel bad-mask exclusion
"""

import logging
from typing import Dict, Any, Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException
import psycopg2.extras

from backend.ingestion.db_writer import get_pg_connection
from backend.services.change_pair import resolve_change_pair
from backend.services.change_stager import get_all_regions_grid_map, stage_tile_temporal_series
from backend.services.quality_auditor import quality_engine

logger = logging.getLogger("change_router")
router = APIRouter(prefix="/api/v1/change", tags=["Change Detection"])


class TilePairChangeRequest(BaseModel):
    tile_id_before: Optional[str] = Field(None, description="Database tile_id for baseline (t1)")
    tile_id_after: Optional[str] = Field(None, description="Database tile_id for subsequent (t2)")
    tile_before_path: Optional[str] = Field(None, description="Direct file path to baseline GeoTIFF")
    tile_after_path: Optional[str] = Field(None, description="Direct file path to subsequent GeoTIFF")
    mask_before_path: Optional[str] = Field(None, description="Direct file path to baseline bad-mask GeoTIFF")
    mask_after_path: Optional[str] = Field(None, description="Direct file path to subsequent bad-mask GeoTIFF")
    min_usable_fraction: float = Field(0.30, ge=0.05, le=0.95, description="Minimum clear ground fraction required")
    diff_threshold: float = Field(0.15, ge=0.01, le=1.0, description="Fixed-scale reflectance change threshold")


class StageTileRequest(BaseModel):
    tile_id: str = Field(..., description="Target tile ID to stage multi-temporal series for")
    staging_dir: str = Field("change_staging", description="Target folder name under data/")


@router.get("/grid-map", response_model=Dict[str, Any])
def get_grid_map():
    """
    Returns all ingested regions and tiles formatted for the sovereign canvas.
    """
    try:
        return get_all_regions_grid_map()
    except Exception as e:
        logger.error(f"Error assembling grid map: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Grid map assembly failed: {str(e)}")


@router.post("/stage-tile", response_model=Dict[str, Any])
def stage_tile(request: StageTileRequest):
    """
    Passes all temporal observations through the Quality Check Engine (inspecting mask.tif).
    DROPS tiles exceeding the bad pixel threshold (>15%), and STAYS/copies only validated
    clean observations into data/change_staging/<site_key>/.
    Writes manifest.json detailing selected_tiles and dropped_tiles.
    """
    try:
        return stage_tile_temporal_series(
            tile_id=request.tile_id,
            staging_dir_name=request.staging_dir
        )
    except ValueError as ve:
        raise HTTPException(status_code=404, detail=str(ve))
    except Exception as e:
        logger.error(f"Error staging tile {request.tile_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Staging failed: {str(e)}")


@router.post("/pair", response_model=Dict[str, Any])
def compute_pair_change(request: TilePairChangeRequest):
    """
    Computes pixel-precise multi-temporal change between two tiles using per-pixel bad-mask exclusion.
    """
    try:
        tif_before = request.tile_before_path
        tif_after = request.tile_after_path
        mask_before = request.mask_before_path
        mask_after = request.mask_after_path

        # If tile IDs are supplied, look up paths from PostgreSQL
        if request.tile_id_before and request.tile_id_after:
            conn = get_pg_connection()
            with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
                cur.execute(
                    "SELECT tile_id, file_path, bad_mask_path FROM tiles WHERE tile_id IN (%s, %s);",
                    (request.tile_id_before, request.tile_id_after)
                )
                rows = {r["tile_id"]: r for r in cur.fetchall()}
            conn.close()

            if request.tile_id_before not in rows or request.tile_id_after not in rows:
                raise HTTPException(status_code=404, detail="One or both tile IDs not found in database.")

            tif_before = rows[request.tile_id_before]["file_path"]
            mask_before = rows[request.tile_id_before].get("bad_mask_path")
            tif_after = rows[request.tile_id_after]["file_path"]
            mask_after = rows[request.tile_id_after].get("bad_mask_path")

        if not tif_before or not tif_after:
            raise HTTPException(
                status_code=400,
                detail="Must provide either (tile_id_before and tile_id_after) or (tile_before_path and tile_after_path)."
            )

        report = resolve_change_pair(
            tile_before_path=tif_before,
            tile_after_path=tif_after,
            mask_before_path=mask_before,
            mask_after_path=mask_after,
            min_usable_fraction=request.min_usable_fraction,
            diff_threshold=request.diff_threshold
        )

        return {
            "status": "success",
            "tile_before": request.tile_id_before or str(tif_before),
            "tile_after": request.tile_id_after or str(tif_after),
            "result": report
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Change pair evaluation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Change detection failed: {str(e)}")
