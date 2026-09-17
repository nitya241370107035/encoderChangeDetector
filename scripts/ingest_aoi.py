#!/usr/bin/env python3
"""
scripts/ingest_aoi.py
=====================
CLI tool to trigger end-to-end multi-temporal satellite imagery ingestion
for any Area of Interest (AOI) polygon into:
  1. data/tiles/<region_id>/<YYYY-MM-DD>/ (tile.tif, thumb.jpg, mask.tif, manifest.json)
  2. PostgreSQL / PostGIS (scenes, tiles, ingestion_coverage)
  3. Qdrant vector database (RemoteCLIP embeddings)
"""

import os
import sys
import json
import argparse
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.ingestion.pipeline import run_aoi_ingestion_pipeline


def main():
    parser = argparse.ArgumentParser(description="AeroLens Satellite AOI Ingestion CLI")
    parser.add_argument("--aoi", default="data/berlin_tesla_aoi.geojson", help="Path to GeoJSON AOI file")
    parser.add_argument("--region", default="giga_berlin", help="Target region ID (e.g. giga_berlin)")
    parser.add_argument("--name", default="Tesla Gigafactory Berlin", help="Human readable region name")
    parser.add_argument("--from-date", default="2021-02-01", help="Start date (YYYY-MM-DD)")
    parser.add_argument("--to-date", default="2023-09-15", help="End date (YYYY-MM-DD)")
    parser.add_argument("--buckets", type=int, default=2, help="Number of multi-temporal time buckets")
    parser.add_argument("--max-cloud", type=float, default=15.0, help="Maximum allowable cloud cover percentage")
    parser.add_argument("--no-db", action="store_true", help="Skip database and vector DB upserts")

    args = parser.parse_args()

    if not os.path.exists(args.aoi):
        print(f"Error: AOI file not found at {args.aoi}")
        sys.exit(1)

    with open(args.aoi, "r") as f:
        aoi_data = json.load(f)

    print("\n=======================================================")
    print("🚀 STARTING SATELLITE AOI INGESTION PIPELINE")
    print("=======================================================")
    print(f"AOI File:     {args.aoi}")
    print(f"Region ID:    {args.region} ({args.name})")
    print(f"Date Window:  {args.from_date} to {args.to_date}")
    print(f"Time Buckets: {args.buckets} | Max Cloud: {args.max_cloud}%")
    print(f"Database:     {'Disabled' if args.no_db else 'Enabled (Postgres + Qdrant)'}")
    print("=======================================================\n")

    result = run_aoi_ingestion_pipeline(
        geojson_input=aoi_data,
        date_from=args.from_date,
        date_to=args.to_date,
        region_id=args.region,
        region_name=args.name,
        num_time_buckets=args.buckets,
        max_cloud_cover=args.max_cloud,
        base_data_dir="data",
        populate_db=not args.no_db,
        fetch_10_bands=True
    )

    print("\n=======================================================")
    print("✅ INGESTION PIPELINE FINISHED SUCCESSFULLY")
    print("=======================================================")
    print(f"Region ID:               {result.region_id}")
    print(f"Total Tiles Generated:   {result.total_tiles_generated}")
    print(f"Scenes Ingested:         {len(result.scenes_processed)}")
    for s in result.scenes_processed:
        print(f"  • {s.scene_id} ({s.acquisition_date}): {s.tiles_count} tiles, cloud {s.cloud_pct:.1f}%")
        print(f"    Manifest: {s.manifest_path}")
    print(f"Elapsed Time:            {result.elapsed_seconds}s")
    print("=======================================================\n")


if __name__ == "__main__":
    main()
