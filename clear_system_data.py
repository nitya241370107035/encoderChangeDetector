"""
clear_system_data.py
====================
Purges all ingested data across:
  1. PostgreSQL database (tiles, scenes, change_events, review_items, ingestion_coverage)
  2. Qdrant vector database collections ('tile_embeddings' and 'maxar_tile_embeddings')
  3. Local data directories (data/tiles, data/uploads)

  It is used for clearing the data in postgis and all the tiles present in the data/ folder 
"""

import os
import shutil
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))

import psycopg2
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

# Database config
PG_HOST = os.getenv("POSTGRES_HOST", "localhost")
PG_PORT = os.getenv("POSTGRES_PORT", "5434")
PG_DB = os.getenv("POSTGRES_DB", "eo_archive")
PG_USER = os.getenv("POSTGRES_USER", "eo_admin")
PG_PASSWORD = os.getenv("POSTGRES_PASSWORD", "eo_password")
PG_DSN = f"host={PG_HOST} port={PG_PORT} dbname={PG_DB} user={PG_USER} password={PG_PASSWORD}"

# Qdrant config
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
COLLECTIONS = ["tile_embeddings", "maxar_tile_embeddings"]


def clear_postgres():
    print("\n--- 1. Clearing PostgreSQL Database ---")
    conn = psycopg2.connect(PG_DSN)
    with conn.cursor() as cur:
        # Check table counts before
        cur.execute("SELECT COUNT(*) FROM tiles;")
        tiles_before = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM scenes;")
        scenes_before = cur.fetchone()[0]
        print(f"Pre-purge: {tiles_before} tiles, {scenes_before} scenes.")

        # Truncate tables with cascade
        cur.execute("""
            TRUNCATE TABLE 
                change_events, 
                review_items, 
                tiles, 
                scenes, 
                ingestion_coverage 
            CASCADE;
        """)
        conn.commit()

        # Check table counts after
        cur.execute("SELECT COUNT(*) FROM tiles;")
        tiles_after = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM scenes;")
        scenes_after = cur.fetchone()[0]
        print(f"Post-purge: {tiles_after} tiles, {scenes_after} scenes.")
    conn.close()
    print("PostgreSQL tables successfully truncated.")


def clear_qdrant():
    print("\n--- 2. Clearing Qdrant Vector Stores ---")
    client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, check_compatibility=False)
    
    existing = [c.name for c in client.get_collections().collections]
    print(f"Existing collections: {existing}")

    for coll in COLLECTIONS:
        if coll in existing:
            client.delete_collection(collection_name=coll)
            print(f"Deleted collection '{coll}'.")
        
        # Recreate fresh empty collection
        client.create_collection(
            collection_name=coll,
            vectors_config=VectorParams(size=512, distance=Distance.COSINE)
        )
        print(f"Recreated fresh empty collection '{coll}' (512-dim, Cosine).")

    # Verify counts
    for coll in COLLECTIONS:
        info = client.get_collection(collection_name=coll)
        print(f"Collection '{coll}' points count: {info.points_count}")


def clear_data_folder():
    print("\n--- 3. Clearing data/ Image Artifacts ---")
    data_dir = ROOT_DIR / "data"
    
    # 1. Clear data/tiles
    tiles_dir = data_dir / "tiles"
    if tiles_dir.exists():
        for item in tiles_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
                print(f"Removed directory: {item.name}")
            elif item.is_file() and item.name != ".gitkeep":
                try:
                    item.unlink()
                    print(f"Removed file: {item.name}")
                except Exception:
                    pass
        print("data/tiles/ cleared.")

    # 2. Clear data/uploads
    uploads_dir = data_dir / "uploads"
    if uploads_dir.exists():
        for item in uploads_dir.iterdir():
            if item.is_dir():
                shutil.rmtree(item, ignore_errors=True)
                print(f"Removed upload dir: {item.name}")
            elif item.is_file() and item.name != ".gitkeep":
                try:
                    item.unlink()
                except Exception:
                    pass
        print("data/uploads/ cleared.")


def main():
    print("=" * 60)
    print("STARTING COMPLETE PURGE OF DB, VECTOR DB & DATA IMAGES")
    print("=" * 60)

    clear_postgres()
    clear_qdrant()
    clear_data_folder()

    print("\n" + "=" * 60)
    print("ALL DATABASES, VECTOR STORES & STORED IMAGES PURGED CLEANLY!")
    print("=" * 60)


if __name__ == "__main__":
    main()
