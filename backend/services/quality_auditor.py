"""
backend/services/quality_auditor.py
====================================
Sovereign Quality Check Engine & Mask Auditor
PS Sections: 2.2.1 / 2.2.2 / 2.2.3

Functions:
  1. audit_mask_file:
     Opens `_mask.tif` with rasterio, counts total pixels, bad pixels, good pixels,
     and computes bad_pixel_pct and usable_surface_pct.
  2. audit_temporal_series:
     Evaluates all candidate temporal acquisitions for a ground footprint (`site_key`).
     Categorizes each tile into STAY (Passed) or DROPPED (Failed quality threshold).
"""

import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Union

import numpy as np
import rasterio

logger = logging.getLogger("quality_auditor")


@dataclass
class TileQualityReport:
    tile_id: str
    site_key: str
    acquisition_date: str
    total_pixels: int
    bad_pixels_count: int
    good_pixels_count: int
    bad_pixel_pct: float
    usable_surface_pct: float
    quality_confidence: float
    status: str              # "STAY" or "DROPPED"
    is_dropped: bool
    rejection_reasons: List[str]
    mask_source: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class QualityCheckEngine:
    """
    Quality Check Engine evaluating mask.tif and cloud/bad pixel percentages.
    """
    DEFAULT_MAX_BAD_PIXEL_PCT: float = 15.0     # Max 15% bad pixels allowed
    DEFAULT_MIN_USABLE_PCT: float = 85.0        # Min 85% clear surface required

    def __init__(
        self,
        max_bad_pixel_pct: float = DEFAULT_MAX_BAD_PIXEL_PCT,
        min_usable_pct: float = DEFAULT_MIN_USABLE_PCT
    ):
        self.max_bad_pixel_pct = max_bad_pixel_pct
        self.min_usable_pct = min_usable_pct

    def audit_mask_file(
        self,
        mask_path: Optional[Union[str, Path]],
        tile_id: str = "unknown",
        site_key: str = "unknown",
        acquisition_date: str = "unknown"
    ) -> TileQualityReport:
        """
        Reads `_mask.tif` using rasterio, counts bad pixels (value > 0),
        and applies threshold check.
        """
        total_pixels = 262144  # 512x512 default
        rejection_reasons: List[str] = []

        if not mask_path or not Path(mask_path).is_file():
            # If no mask file on disk, assume 0 bad pixels (clean)
            return TileQualityReport(
                tile_id=tile_id,
                site_key=site_key,
                acquisition_date=acquisition_date,
                total_pixels=total_pixels,
                bad_pixels_count=0,
                good_pixels_count=total_pixels,
                bad_pixel_pct=0.0,
                usable_surface_pct=100.0,
                quality_confidence=1.0,
                status="STAY",
                is_dropped=False,
                rejection_reasons=[],
                mask_source="none_assumed_clean"
            )

        p = Path(mask_path)
        try:
            with rasterio.open(str(p)) as src:
                data = src.read(1)
                total_pixels = int(data.size)
                bad_pixels_count = int(np.count_nonzero(data > 0))
                good_pixels_count = int(total_pixels - bad_pixels_count)
        except Exception as e:
            logger.warning(f"Failed to read mask {mask_path}: {e}")
            return TileQualityReport(
                tile_id=tile_id,
                site_key=site_key,
                acquisition_date=acquisition_date,
                total_pixels=total_pixels,
                bad_pixels_count=total_pixels,
                good_pixels_count=0,
                bad_pixel_pct=100.0,
                usable_surface_pct=0.0,
                quality_confidence=0.0,
                status="DROPPED",
                is_dropped=True,
                rejection_reasons=[f"UNREADABLE_MASK_FILE: {e}"],
                mask_source=str(p)
            )

        bad_pixel_pct = round((bad_pixels_count / total_pixels) * 100.0, 2)
        usable_surface_pct = round((good_pixels_count / total_pixels) * 100.0, 2)
        quality_confidence = round(good_pixels_count / total_pixels, 4)

        if bad_pixel_pct > self.max_bad_pixel_pct:
            rejection_reasons.append(
                f"EXCESSIVE_BAD_PIXELS: {bad_pixel_pct}% bad pixels exceeds limit of {self.max_bad_pixel_pct}%"
            )

        if usable_surface_pct < self.min_usable_pct:
            rejection_reasons.append(
                f"INSUFFICIENT_USABLE_SURFACE: {usable_surface_pct}% usable area below minimum {self.min_usable_pct}%"
            )

        is_dropped = len(rejection_reasons) > 0
        status = "DROPPED" if is_dropped else "STAY"

        return TileQualityReport(
            tile_id=tile_id,
            site_key=site_key,
            acquisition_date=acquisition_date,
            total_pixels=total_pixels,
            bad_pixels_count=bad_pixels_count,
            good_pixels_count=good_pixels_count,
            bad_pixel_pct=bad_pixel_pct,
            usable_surface_pct=usable_surface_pct,
            quality_confidence=quality_confidence,
            status=status,
            is_dropped=is_dropped,
            rejection_reasons=rejection_reasons,
            mask_source=str(p)
        )

    def audit_temporal_series(
        self,
        site_key: str,
        tile_records: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Audits all candidate acquisitions for a ground footprint.
        Splits them into `selected_tiles` (STAY) and `dropped_tiles` (DROPPED).
        """
        selected_tiles: List[Dict[str, Any]] = []
        dropped_tiles: List[Dict[str, Any]] = []

        # Sort chronologically by date
        sorted_records = sorted(
            tile_records,
            key=lambda r: str(r.get("acquisition_date") or "")
        )

        for row in sorted_records:
            tid = row["tile_id"]
            acq_date = str(row.get("acquisition_date") or "")[:10]
            mask_p = row.get("bad_mask_path")

            report = self.audit_mask_file(
                mask_path=mask_p,
                tile_id=tid,
                site_key=site_key,
                acquisition_date=acq_date
            )

            record_summary = {
                "tile_id": tid,
                "date": acq_date,
                "bad_pixels": report.bad_pixels_count,
                "total_pixels": report.total_pixels,
                "bad_pixel_pct": report.bad_pixel_pct,
                "usable_pct": report.usable_surface_pct,
                "quality_confidence": report.quality_confidence,
                "status": report.status,
                "raw_record": row
            }

            if report.is_dropped:
                record_summary["reasons"] = report.rejection_reasons
                dropped_tiles.append(record_summary)
            else:
                selected_tiles.append(record_summary)

        return {
            "thresholds_applied": {
                "max_bad_pixel_pct": self.max_bad_pixel_pct,
                "min_usable_pct": self.min_usable_pct
            },
            "total_evaluated": len(tile_records),
            "passed_count": len(selected_tiles),
            "dropped_count": len(dropped_tiles),
            "selected_tiles": selected_tiles,
            "dropped_tiles": dropped_tiles
        }


quality_engine = QualityCheckEngine()
