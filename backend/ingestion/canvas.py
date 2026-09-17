"""
backend/ingestion/canvas.py
===========================
Phase 1.2: Multi-Band Working Canvas Assembly & Multi-Scene Mosaicking
======================================================================
PS Sections: 2.2.1, 2.2.3, 2.2.6 (Ingestion), 2.2.7 (Evaluation Constraints)

Assembles a unified, reprojected EPSG:4326 multi-band canvas from either:
  1. Entry Point A (STAC Scene Metadata / Multi-Scene Bucket Mosaics):
     - Streams 10 spectral bands (B01, B02, B03, B04, B05, B08, B8A, B09, B10, B11, B12) or 5 core bands.
     - Merges overlapping Sentinel-2 granules in the same time bucket into one continuous mosaic.
     - Reprojects native UTM to EPSG:4326 in a single stage.
     - Crops to buffered AOI bounding box (+5% margin).

  2. Entry Point B (Direct Local GeoTIFF — 100% Offline):
     - Reads bands directly from local evaluation files without network.
     - Reprojects to EPSG:4326 if native CRS differs.
     - Uses file's own extent and detected band layout.

Both entry points produce the identical CanvasData representation, feeding directly
into Phase 1.3 (Quality Masking) and Phase 1.4 (Tiling & NDVI/NDWI/NDBI indices).
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Union

import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.vrt import WarpedVRT

from backend.ingestion.input_validator import ValidatedFileInput
from backend.ingestion.stac_search import STACSceneMetadata
from backend.ingestion.normalization.radiometric_offset import detect_and_correct_sentinel2_offset

logger = logging.getLogger(__name__)

# ~10 meters in EPSG:4326 degrees (at equator)
DEFAULT_PIXEL_RES_DEG = 0.00008983
DEFAULT_BUFFER_PCT = 0.05

# 5 Core Bands for Visuals, Vegetation (NDVI), Water (NDWI), and Built-Up (NDBI)
DEFAULT_5_BANDS = ["blue", "green", "red", "nir", "swir"]
# 10 Full Bands for s2cloudless ML Model & Spectral Analysis
DEFAULT_10_BANDS = ["B01", "B02", "B03", "B04", "B05", "B08", "B8A", "B09", "B10", "B11", "B12"]
STANDARD_CANVAS_BANDS = DEFAULT_5_BANDS

# Alias lookup for flexible band resolution
BAND_ALIASES: Dict[str, List[str]] = {
    "blue": ["blue", "b02", "b2"], "b02": ["b02", "blue", "b2"], "b2": ["b2", "b02", "blue"],
    "green": ["green", "b03", "b3"], "b03": ["b03", "green", "b3"], "b3": ["b3", "b03", "green"],
    "red": ["red", "b04", "b4"], "b04": ["b04", "red", "b4"], "b4": ["b4", "b04", "red"],
    "nir": ["nir", "b08", "b8", "b5"], "b08": ["b08", "nir", "b8"], "b8": ["b8", "b08", "nir"],
    "swir": ["swir", "b11", "b6"], "b11": ["b11", "swir", "b6"], "b6": ["b6", "b11", "swir"],
    "b01": ["b01", "b1", "coastal"], "b1": ["b1", "b01", "coastal"],
    "b05": ["b05", "b5"], "b8a": ["b8a"],
    "b09": ["b09", "b9"], "b10": ["b10"], "b12": ["b12", "b7"],
}


@dataclass
class CanvasData:
    """Represents a standardized multi-band satellite raster canvas in EPSG:4326."""
    data: np.ndarray             # Shape: (Bands, Height, Width), float32 or uint16
    band_names: List[str]        # e.g., ['B01', 'B02', 'B03', 'B04', ...] or ['blue', 'green', ...]
    transform: rasterio.Affine   # Affine transform for pixel-to-geographic mapping
    crs: str                     # Standardized to 'EPSG:4326'
    bbox: Tuple[float, float, float, float]  # (min_lon, min_lat, max_lon, max_lat)
    height: int
    width: int
    source_type: str = "aoi_search"
    visual_rgb: Optional[np.ndarray] = None # Shape (3, H, W) pristine 10m True Color Image
    nodata_val: Optional[float] = None

    def _match_band_index(self, name: str) -> Optional[int]:
        target = name.lower()
        lower_names = [b.lower() for b in self.band_names]
        if target in lower_names:
            return lower_names.index(target)
        
        aliases = BAND_ALIASES.get(target, [])
        for alias in aliases:
            if alias.lower() in lower_names:
                return lower_names.index(alias.lower())
        return None

    def has_band(self, name: str) -> bool:
        return self._match_band_index(name) is not None

    def get_band(self, name: str) -> np.ndarray:
        """Returns 2D array of the requested band (float32)."""
        idx = self._match_band_index(name)
        if idx is None:
            raise KeyError(f"Band '{name}' not found in canvas. Available: {self.band_names}")
        return self.data[idx].astype(np.float32)

    def get_rgb(self) -> np.ndarray:
        """
        Returns 3-band RGB array of shape (3, H, W).
        Gracefully handles 1-band grayscale/panchromatic and non-standard bands without crashing.
        """
        if self.visual_rgb is not None and self.visual_rgb.shape[0] >= 3:
            return self.visual_rgb.astype(np.float32)

        # 1-band / Grayscale / Panchromatic handling (PS requirement: gracefully degrade, no crash)
        if len(self.band_names) == 1 or self.data.shape[0] == 1:
            gray = self.data[0].astype(np.float32)
            return np.stack([gray, gray, gray], axis=0)

        # Standard red/green/blue lookup
        has_r = self.has_band("red") or self.has_band("B04") or self.has_band("b4")
        has_g = self.has_band("green") or self.has_band("B03") or self.has_band("b3")
        has_b = self.has_band("blue") or self.has_band("B02") or self.has_band("b2")

        if has_r and has_g and has_b:
            red = self.get_band("red") if self.has_band("red") else (self.get_band("B04") if self.has_band("B04") else self.get_band("b4"))
            green = self.get_band("green") if self.has_band("green") else (self.get_band("B03") if self.has_band("B03") else self.get_band("b3"))
            blue = self.get_band("blue") if self.has_band("blue") else (self.get_band("B02") if self.has_band("B02") else self.get_band("b2"))
            return np.stack([red, green, blue], axis=0)

        # Graceful fallback when standard RGB channels are not all detected
        if self.data.shape[0] >= 3:
            return self.data[:3].astype(np.float32)
        elif self.data.shape[0] == 2:
            b0 = self.data[0].astype(np.float32)
            b1 = self.data[1].astype(np.float32)
            return np.stack([b0, b1, b0], axis=0)
        else:
            gray = self.data[0].astype(np.float32)
            return np.stack([gray, gray, gray], axis=0)


def calculate_buffered_bbox(
    bbox: Tuple[float, float, float, float],
    buffer_pct: float = 0.05
) -> Tuple[float, float, float, float]:
    """Expands bounding box by a safety margin percentage."""
    min_lon, min_lat, max_lon, max_lat = bbox
    width = max_lon - min_lon
    height = max_lat - min_lat
    buf_x = width * buffer_pct
    buf_y = height * buffer_pct
    return (
        max(-180.0, min_lon - buf_x),
        max(-90.0, min_lat - buf_y),
        min(180.0, max_lon + buf_x),
        min(90.0, max_lat + buf_y)
    )


# Helper function for synthetic test canvas generation
def generate_synthetic_canvas(
    bbox: Tuple[float, float, float, float] = (72.5, 23.0, 72.6, 23.1),
    cloud_pct_target: float = 0.1,
    height: int = 512,
    width: int = 512,
    band_names: Optional[List[str]] = None
) -> CanvasData:
    """Generates a synthetic 5-band or 10-band CanvasData for unit testing / mock runs."""
    if band_names is None:
        band_names = ["blue", "green", "red", "nir", "swir"]

    min_lon, min_lat, max_lon, max_lat = bbox
    transform = from_bounds(min_lon, min_lat, max_lon, max_lat, width, height)

    np.random.seed(42)
    bands_count = len(band_names)
    data = np.random.uniform(500, 3000, size=(bands_count, height, width)).astype(np.float32)

    if cloud_pct_target > 0:
        c_pixels = int(height * width * cloud_pct_target)
        dim = int(np.sqrt(c_pixels))
        data[:, :dim, :dim] = 9500.0

    red = data[band_names.index("red")] if "red" in band_names else data[2]
    green = data[band_names.index("green")] if "green" in band_names else data[1]
    blue = data[band_names.index("blue")] if "blue" in band_names else data[0]
    visual_rgb = np.stack([red, green, blue], axis=0)

    return CanvasData(
        data=data,
        band_names=band_names,
        transform=transform,
        crs="EPSG:4326",
        bbox=bbox,
        height=height,
        width=width,
        source_type="synthetic",
        visual_rgb=visual_rgb
    )


# ============================================================
# 1. Entry Point A: Assemble Canvas from STAC COG URLs
# ============================================================

def assemble_canvas_from_stac(
    scene_meta: STACSceneMetadata,
    aoi_bbox: Tuple[float, float, float, float],
    buffer_pct: float = DEFAULT_BUFFER_PCT,
    pixel_res_deg: float = DEFAULT_PIXEL_RES_DEG,
    fetch_10_bands: bool = True
) -> CanvasData:
    """
    Phase 1.2 Single Scene Canvas Assembler:
    Streams spectral bands from remote Cloud-Optimized GeoTIFFs (COGs)
    using rasterio WarpedVRT reprojection to EPSG:4326.
    """
    return assemble_mosaicked_canvas_from_stac(
        scenes=[scene_meta],
        aoi_bbox=aoi_bbox,
        buffer_pct=buffer_pct,
        pixel_res_deg=pixel_res_deg,
        fetch_10_bands=fetch_10_bands
    )


def assemble_mosaicked_canvas_from_stac(
    scenes: List[STACSceneMetadata],
    aoi_bbox: Tuple[float, float, float, float],
    buffer_pct: float = DEFAULT_BUFFER_PCT,
    pixel_res_deg: float = DEFAULT_PIXEL_RES_DEG,
    fetch_10_bands: bool = True
) -> CanvasData:
    """
    Phase 1.2 Multi-Scene Canvas Assembler (Mosaicking per Time Bucket):
    Merges overlapping scenes/granules in the same time bucket into a single
    unified EPSG:4326 working canvas, eliminating duplicate tiles & coverage gaps.
    """
    if not scenes:
        raise ValueError("Cannot assemble canvas: empty scene list provided.")

    buffered_bbox = calculate_buffered_bbox(aoi_bbox, buffer_pct=buffer_pct)
    min_lon, min_lat, max_lon, max_lat = buffered_bbox

    width = max(512, int(round((max_lon - min_lon) / pixel_res_deg)))
    height = max(512, int(round((max_lat - min_lat) / pixel_res_deg)))

    # If all scenes are mock/offline fallback, return synthetic canvas instantly
    if any(getattr(s, "is_mock", False) for s in scenes):
        logger.info("[Phase 1.2] Mock scene detected — generating synthetic multi-band canvas.")
        target_bands = DEFAULT_10_BANDS if fetch_10_bands else DEFAULT_5_BANDS
        return generate_synthetic_canvas(
            bbox=buffered_bbox,
            cloud_pct_target=scenes[0].cloud_cover / 100.0,
            height=height,
            width=width,
            band_names=target_bands
        )

    target_transform = from_bounds(min_lon, min_lat, max_lon, max_lat, width, height)

    # Determine band targets to stream
    if fetch_10_bands:
        band_targets = [
            ("B01", ["b01"]),
            ("B02", ["blue", "b02"]),
            ("B03", ["green", "b03"]),
            ("B04", ["red", "b04"]),
            ("B05", ["b05"]),
            ("B08", ["nir", "b08"]),
            ("B8A", ["b8a"]),
            ("B09", ["b09"]),
            ("B10", ["b10"]),
            ("B11", ["swir", "b11"]),
            ("B12", ["b12"]),
        ]
    else:
        band_targets = [
            ("blue", ["blue", "b02"]),
            ("green", ["green", "b03"]),
            ("red", ["red", "b04"]),
            ("nir", ["nir", "b08"]),
            ("swir", ["swir", "b11"]),
        ]

    env_params = {
        "AWS_NO_SIGN_REQUEST": "YES",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF",
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "GDAL_HTTP_MAX_RETRY": "3",
        "GDAL_HTTP_RETRY_DELAY": "2",
        "GDAL_HTTP_TIMEOUT": "15",
        "GDAL_HTTP_CONNECTTIMEOUT": "10",
    }

    canvas_layers = []
    loaded_band_names = []
    visual_rgb = np.zeros((3, height, width), dtype=np.float32)
    tci_loaded = False

    with rasterio.Env(**env_params):
        # 1. Stream & Mosaic True Color Image (TCI)
        for scene in scenes:
            tci_url = scene.assets.visual
            if tci_url:
                try:
                    with rasterio.open(tci_url) as src:
                        with WarpedVRT(src, crs="EPSG:4326", transform=target_transform,
                                       width=width, height=height, resampling=Resampling.bilinear) as vrt:
                            tci_raw = vrt.read(out_shape=(3, height, width), resampling=Resampling.bilinear).astype(np.float32)
                            valid_mask = (tci_raw > 0)
                            visual_rgb[valid_mask] = tci_raw[valid_mask]
                            tci_loaded = True
                except Exception as e:
                    logger.warning(f"[Phase 1.2] Failed to stream TCI for scene {scene.scene_id}: {e}")

        # 2. Stream & Mosaic scientific bands across scene list
        for b_name, alias_keys in band_targets:
            combined_band = np.zeros((height, width), dtype=np.float32)
            band_has_data = False

            for scene in scenes:
                # Find matching asset URL for this band
                b_url = None
                for key in alias_keys:
                    b_url = scene.assets.get_band_url(key)
                    if b_url:
                        break

                if not b_url:
                    continue

                try:
                    with rasterio.open(b_url) as src:
                        with WarpedVRT(src, crs="EPSG:4326", transform=target_transform,
                                       width=width, height=height, resampling=Resampling.bilinear) as vrt:
                            data = vrt.read(1, out_shape=(height, width), resampling=Resampling.bilinear).astype(np.float32)
                            
                            # Dual-verification radiometric offset check for Sentinel-2 PB 04.00+
                            data, _ = detect_and_correct_sentinel2_offset(
                                data,
                                metadata_properties=scene.properties,
                                band_name=b_name
                            )

                            # Overwrite zeros with non-nodata pixels across granule boundaries
                            valid_mask = (data > 0)
                            combined_band[valid_mask] = data[valid_mask]
                            band_has_data = True
                except Exception as e:
                    logger.warning(f"[Phase 1.2] Failed to stream band '{b_name}' for scene {scene.scene_id}: {e}")

            if band_has_data:
                canvas_layers.append(combined_band)
                loaded_band_names.append(b_name)

    if not canvas_layers or len(canvas_layers) < 3:
        logger.warning(f"[Phase 1.2] Only {len(canvas_layers)} bands loaded for mosaicked canvas. Retrying with synthetic fallback.")
        return generate_synthetic_canvas(
            bbox=buffered_bbox,
            cloud_pct_target=scenes[0].cloud_cover / 100.0,
            height=height,
            width=width
        )

    canvas_array = np.stack(canvas_layers, axis=0)

    return CanvasData(
        data=canvas_array,
        band_names=loaded_band_names,
        transform=target_transform,
        crs="EPSG:4326",
        bbox=buffered_bbox,
        height=height,
        width=width,
        source_type="aoi_search",
        visual_rgb=visual_rgb if tci_loaded else None
    )


# ============================================================
# 2. Entry Point B: Assemble Canvas from Local GeoTIFF (Offline)
# ============================================================

def assemble_canvas_from_file(
    file_input: ValidatedFileInput,
    target_crs: str = "EPSG:4326",
    pixel_res_deg: float = DEFAULT_PIXEL_RES_DEG
) -> CanvasData:
    """
    Entry Point B Canvas Assembler:
    Reads local evaluation GeoTIFF directly, reprojecting to EPSG:4326.
    Handles both single-file multi-band GeoTIFF and multi-file separate band rasters (e.g. Landsat folder).
    """
    min_lon, min_lat, max_lon, max_lat = file_input.bounds_wgs84
    width = max(512, int(round((max_lon - min_lon) / pixel_res_deg)))
    height = max(512, int(round((max_lat - min_lat) / pixel_res_deg)))
    target_transform = from_bounds(min_lon, min_lat, max_lon, max_lat, width, height)

    # Check if multi-file separate bands (e.g. Landsat Collection 2 or Sentinel-2 separate bands)
    if file_input.sub_file_paths and len(file_input.sub_file_paths) > 1:
        band_layers: List[np.ndarray] = []
        loaded_band_names: List[str] = []

        for p, b_name in zip(file_input.sub_file_paths, file_input.band_order):
            try:
                with rasterio.open(str(p)) as src:
                    with WarpedVRT(src, crs=target_crs, transform=target_transform,
                                   width=width, height=height, resampling=Resampling.bilinear) as vrt:
                        band_data = vrt.read(1, out_shape=(height, width), resampling=Resampling.bilinear).astype(np.float32)
                        band_layers.append(band_data)
                        loaded_band_names.append(b_name)
            except Exception as e:
                logger.warning(f"[Canvas] Error loading multi-file band '{p.name}': {e}")

        if not band_layers:
            raise ValueError(f"Failed to read any bands from multi-file input: {file_input.sub_file_paths}")

        raw_data = np.stack(band_layers, axis=0)
        final_band_names = loaded_band_names
    else:
        with rasterio.open(str(file_input.file_path)) as src:
            if src.crs is None:
                raise ValueError(f"GeoTIFF '{file_input.file_path.name}' lacks embedded CRS metadata.")
            with WarpedVRT(src, crs=target_crs, transform=target_transform,
                           width=width, height=height, resampling=Resampling.bilinear) as vrt:
                raw_data = vrt.read(out_shape=(src.count, height, width), resampling=Resampling.bilinear)
        final_band_names = file_input.band_order

    raw_data_float = raw_data.astype(np.float32)
    sensor_name = getattr(file_input, "sensor", "") or ""
    # If Sentinel-2 or unspecified sensor, run radiometric offset check across bands
    if "sentinel" in sensor_name.lower() or not sensor_name:
        corrected_bands = []
        for b_idx in range(raw_data_float.shape[0]):
            b_label = final_band_names[b_idx] if b_idx < len(final_band_names) else f"band_{b_idx+1}"
            c_band, _ = detect_and_correct_sentinel2_offset(
                raw_data_float[b_idx],
                metadata_properties={"sensor": sensor_name},
                band_name=b_label
            )
            corrected_bands.append(c_band)
        raw_data_float = np.stack(corrected_bands, axis=0)

    return CanvasData(
        data=raw_data_float,
        band_names=final_band_names,
        transform=target_transform,
        crs="EPSG:4326",
        bbox=file_input.bounds_wgs84,
        height=height,
        width=width,
        source_type="organiser_provided",
        nodata_val=file_input.nodata_value
    )


# Unified entry point dispatcher
def assemble_working_canvas(
    scene_meta_or_file: Union[STACSceneMetadata, ValidatedFileInput],
    aoi_bbox: Optional[Tuple[float, float, float, float]] = None,
    buffer_pct: float = 0.05,
    pixel_res_deg: float = DEFAULT_PIXEL_RES_DEG
) -> CanvasData:
    """Unified entry point dispatcher."""
    if isinstance(scene_meta_or_file, ValidatedFileInput):
        return assemble_canvas_from_file(scene_meta_or_file, pixel_res_deg=pixel_res_deg)
    elif isinstance(scene_meta_or_file, STACSceneMetadata):
        if aoi_bbox is None:
            aoi_bbox = scene_meta_or_file.bbox
        return assemble_canvas_from_stac(scene_meta_or_file, aoi_bbox, buffer_pct, pixel_res_deg)
    else:
        raise TypeError(f"Unsupported source for canvas assembly: {type(scene_meta_or_file)}")
