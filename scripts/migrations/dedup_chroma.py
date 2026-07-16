"""
dedup_chroma.py
---------------
Removes duplicate-source documents from the Chroma collection.

The crawl sometimes saves the same web page under two different filenames
(e.g., different URL hash suffixes), producing multiple sets of chunks
with the same 'url' metadata but different 'source' (local file path).
At query time, all copies compete for the same top-10 Chroma slots,
crowding out genuinely different pages.

Fix: For each unique normalized URL, pick ONE source file to keep.
     Delete ALL chunks whose source file is not the chosen keeper.
     This preserves all semantic chunks from the chosen file (so we
     still have full multi-chunk coverage of each page), while freeing
     up retrieval slots wasted on exact duplicates.

This is a $0 operation — no re-embedding, metadata-only deletions.

Usage:
    python dedup_chroma.py            # dry run (prints stats, no delete)
    python dedup_chroma.py --delete   # actually deletes duplicates
"""

import argparse
import sys
import os
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config import AppConfig
from chromadb import PersistentClient


def normalize_url(url: str) -> str:
    return url.strip().rstrip("/").lower()


def run_dedup(chroma_path: str, dry_run: bool = True) -> None:
    client = PersistentClient(path=chroma_path)
    collections = client.list_collections()
    if not collections:
        print("ERROR: No collections found.")
        return

    col = client.get_collection(collections[0].name)
    total = col.count()
    print(f"Collection '{col.name}'  |  Total chunks: {total}")
    print("Fetching all metadata (no embeddings loaded)...")

    # Fetch all IDs + metadata in batches
    batch_size = 1000
    all_ids = []
    all_metas = []
    offset = 0
    while True:
        result = col.get(limit=batch_size, offset=offset, include=["metadatas"])
        if not result["ids"]:
            break
        all_ids.extend(result["ids"])
        all_metas.extend(result["metadatas"])
        offset += batch_size
        if len(result["ids"]) < batch_size:
            break

    print(f"Loaded {len(all_ids)} chunk records.")

    # Group chunk IDs by (normalized_url, source_file)
    # Structure: url_norm -> {source_file -> [chunk_ids]}
    url_to_sources: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))

    for chunk_id, meta in zip(all_ids, all_metas):
        url_raw = meta.get("url", "") or meta.get("source", "")
        source  = meta.get("source", "")
        url_norm = normalize_url(url_raw)
        if not url_norm:
            continue
        url_to_sources[url_norm][source].append(chunk_id)

    # For each URL, pick the source file with the MOST chunks as keeper
    # (heuristic: the copy with more chunks was split more completely)
    duplicate_ids: list[str] = []
    duplicate_url_count = 0
    total_kept_chunks = 0
    total_deleted_chunks = 0

    for url_norm, sources in url_to_sources.items():
        if len(sources) <= 1:
            # Only one source file for this URL — nothing to dedup
            total_kept_chunks += sum(len(ids) for ids in sources.values())
            continue

        # Multiple source files for the same URL
        duplicate_url_count += 1
        # Keep the source with the most chunks; tie-break by source path (alphabetical)
        keeper = max(sources.keys(), key=lambda s: (len(sources[s]), s))

        for source, ids in sources.items():
            if source == keeper:
                total_kept_chunks += len(ids)
            else:
                duplicate_ids.extend(ids)
                total_deleted_chunks += len(ids)

    print(f"\n{'DRY RUN' if dry_run else 'LIVE RUN'} Results:")
    print(f"  Unique URLs in corpus:           {len(url_to_sources)}")
    print(f"  URLs with duplicate source files: {duplicate_url_count}")
    print(f"  Chunks to keep:                  {total_kept_chunks}")
    print(f"  Chunks to delete (duplicates):   {total_deleted_chunks}")
    print(f"  Post-dedup collection size:      {total - total_deleted_chunks}")

    if not duplicate_ids:
        print("\nNo duplicates found — nothing to delete.")
        return

    if dry_run:
        print("\nDry run complete. Run with --delete to apply.")
        return

    # Delete in batches of 500
    print(f"\nDeleting {len(duplicate_ids)} duplicate chunks...")
    batch_size = 500
    for i in range(0, len(duplicate_ids), batch_size):
        batch = duplicate_ids[i : i + batch_size]
        col.delete(ids=batch)
        print(f"  Deleted {min(i + batch_size, len(duplicate_ids))}/{len(duplicate_ids)}")

    print(f"\nDone. Collection now has {col.count()} chunks.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dedup Chroma collection by URL.")
    parser.add_argument("--delete", action="store_true",
                        help="Actually delete duplicates (default is dry run)")
    args = parser.parse_args()

    config = AppConfig()
    run_dedup(config.DATABASE_PATH, dry_run=not args.delete)
