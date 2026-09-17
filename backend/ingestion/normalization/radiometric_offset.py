"""
backend/ingestion/normalization/radiometric_offset.py
======================================================
Dual-Verification Radiometric Offset Harvester for Sentinel-2
PS Sections: 2.2.1 / 2.2.2 (Multi-Temporal Harmonization & Radiometric Stability)

Problem:
  On January 25, 2022, ESA deployed Sentinel-2 Processing Baseline (PB) 04.00.
  To prevent clipping negative surface reflectance over dark targets, ESA introduced
  an additive radiometric offset of +1000 DN (BOA_ADD_OFFSET = -1000).
  
  True physical surface reflectance is:
      Reflectance = (DN - 1000) / 10000.0   [for PB >= 04.00 raw products]
      Reflectance = DN / 10000.0            [for legacy PB < 04.00 products]

The Trap:
  Blindly trusting metadata tags fails because:
  1. Many STAC providers (e.g. AWS Earth Search, Microsoft Planetary Computer)
     or reprocessing tools silently harmonize post-2022 scenes by subtracting 1000,
     yet the metadata tag `s2:processing_baseline` remains "04.00" or "05.00+".
     Blind subtraction would corrupt dark pixels into negative/zeros.
  2. Conversely, raw ESA granules or unharmonized mirrors omit the harmonization flag
     while retaining the +1000 DN shift. Failing to correct them creates a massive
     artificial +0.10 (+10%) reflectance jump, causing 100% false-positive change detections.

Dual-Verification Solution:
  1. Metadata check: Inspects processing baseline, acquisition date, and provider flags.
  2. Physical Impossibility Test: Under the +1000 offset hypothesis, NO valid ground
     pixel can physically have DN < 900. If valid pixels exist below this threshold
     (e.g., deep water, shadows in DN 50-700), the offset hypothesis is DISPROVEN.
     Only when the offset is unharmonized AND all valid pixels satisfy DN >= 950
     is the 1000 DN offset subtracted.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Tuple, Optional, Union
import numpy as np

logger = logging.getLogger(__name__)

# Baseline threshold where ESA introduced the +1000 DN shift
ESA_PB_OFFSET_INTRO_BASELINE = "04.00"
# Date when PB 04.00 was deployed operationally by ESA
ESA_PB_OFFSET_INTRO_DATE = "2022-01-25"
# Standard ESA offset value in DN
ESA_DN_OFFSET = 1000.0


def parse_baseline_version(baseline_str: Optional[str]) -> Tuple[int, int]:
    """Parses a baseline string like '04.00', '05.09', '4.0' into (major, minor)."""
    if not baseline_str:
        return (0, 0)
    try:
        parts = str(baseline_str).strip().split(".")
        major = int(parts[0])
        minor = int(parts[1]) if len(parts) > 1 else 0
        return (major, minor)
    except Exception:
        return (0, 0)


def detect_and_correct_sentinel2_offset(
    band_array: np.ndarray,
    metadata_properties: Optional[Dict[str, Any]] = None,
    band_name: str = "",
    min_valid_dn_threshold: float = 900.0,
    min_valid_pixel_count: int = 25
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Applies the Dual-Verification Radiometric Offset detection & correction.

    Args:
        band_array: 2D (H, W) or 3D (Bands, H, W) numpy array of raw Digital Numbers.
        metadata_properties: Dict of scene metadata (STAC properties, tags).
        band_name: Name of the band (e.g. 'B02', 'blue', 'B04', 'red').
        min_valid_dn_threshold: DN threshold below which the +1000 offset is disproven.
        min_valid_pixel_count: Minimum valid non-zero pixels required to perform test.

    Returns:
        Tuple of:
          - corrected_band_array: float32 array with offset removed if verified.
          - audit_info: Dict documenting the decision, baseline, min pixel, and offset applied.
    """
    props = metadata_properties or {}
    baseline_str = str(
        props.get("s2:processing_baseline")
        or props.get("processing_baseline")
        or props.get("processing:baseline")
        or ""
    ).strip()
    
    date_str = str(props.get("datetime") or props.get("acquisition_date") or "")[:10]
    provider_already_applied = bool(
        props.get("earthsearch:boa_offset_applied")
        or props.get("boa_offset_applied")
        or props.get("harmonized", False)
    )

    # 1. Evaluate Metadata Hypothesis
    major, minor = parse_baseline_version(baseline_str)
    baseline_is_post_pb04 = (major > 4) or (major == 4 and minor >= 0)
    date_is_post_pb04 = False
    if date_str:
        try:
            date_is_post_pb04 = date_str >= ESA_PB_OFFSET_INTRO_DATE
        except Exception:
            pass

    metadata_indicates_offset = (baseline_is_post_pb04 or date_is_post_pb04) and not provider_already_applied

    # 2. Physical Impossibility Test over valid data
    # Non-nodata pixels (excluding 0, NaN, Inf, and negative sentinels)
    valid_mask = (band_array > 0) & np.isfinite(band_array)
    valid_pixels = band_array[valid_mask]

    audit_info: Dict[str, Any] = {
        "band": band_name,
        "processing_baseline": baseline_str or "unknown",
        "acquisition_date": date_str,
        "provider_already_applied": provider_already_applied,
        "metadata_indicates_offset": metadata_indicates_offset,
        "offset_subtracted": 0.0,
        "hypothesis_outcome": "no_offset_needed",
        "min_valid_dn": None,
        "p1_valid_dn": None
    }

    if valid_pixels.size < min_valid_pixel_count:
        # Not enough valid data to run physical test; trust metadata/fallback safely
        logger.debug(
            f"[RadiometricOffset] Band {band_name}: insufficient valid pixels ({valid_pixels.size}). "
            "Skipping physical offset check."
        )
        return band_array.astype(np.float32), audit_info

    min_val = float(np.min(valid_pixels))
    p1_val = float(np.percentile(valid_pixels, 1.0))
    audit_info["min_valid_dn"] = round(min_val, 2)
    audit_info["p1_valid_dn"] = round(p1_val, 2)

    # Physical Disproof: If any significant population of valid pixels is < 900 DN,
    # the offset cannot be present (or has already been subtracted).
    # Real water / shadow / forest reflectance in visible/NIR is typically 50 - 700 DN.
    offset_physically_disproven = (p1_val < min_valid_dn_threshold) or (min_val < (min_valid_dn_threshold - 150.0))

    if provider_already_applied:
        audit_info["hypothesis_outcome"] = "provider_already_harmonized"
        logger.debug(
            f"[RadiometricOffset] Band {band_name}: provider flag 'boa_offset_applied' is True. "
            f"(min={min_val:.1f}, p1={p1_val:.1f}). No correction needed."
        )
        return band_array.astype(np.float32), audit_info

    if offset_physically_disproven:
        audit_info["hypothesis_outcome"] = "disproven_by_dark_pixels"
        if metadata_indicates_offset:
            logger.info(
                f"[RadiometricOffset] Band {band_name}: Metadata indicated PB >= 04.00, BUT physical test "
                f"detected valid dark pixels (min={min_val:.1f}, p1={p1_val:.1f} < {min_valid_dn_threshold}). "
                "Disproven! Preventing erroneous -1000 DN subtraction."
            )
        return band_array.astype(np.float32), audit_info

    # Both conditions met:
    # 1. Metadata or date indicates PB >= 04.00 and not already applied
    # 2. All valid pixels are >= 900 DN (physical confirmation)
    if metadata_indicates_offset and not offset_physically_disproven:
        audit_info["hypothesis_outcome"] = "confirmed_and_corrected"
        audit_info["offset_subtracted"] = ESA_DN_OFFSET
        logger.info(
            f"[RadiometricOffset] Band {band_name}: PB 04.00+ radiometric offset CONFIRMED "
            f"(baseline={baseline_str}, min={min_val:.1f}, p1={p1_val:.1f} >= {min_valid_dn_threshold}). "
            f"Subtracting {ESA_DN_OFFSET} DN."
        )
        # Subtract offset strictly on non-nodata pixels, keeping 0 as nodata
        corrected = band_array.astype(np.float32).copy()
        corrected[valid_mask] = np.maximum(0.0, corrected[valid_mask] - ESA_DN_OFFSET)
        return corrected, audit_info

    # Default: legacy scene (PB < 04.00, e.g. 2017-2021)
    audit_info["hypothesis_outcome"] = "legacy_scene_no_offset"
    return band_array.astype(np.float32), audit_info
