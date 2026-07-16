"""
patch_chroma_urls.py
--------------------
One-time $0 metadata patch: reads the <!-- SOURCE: url TITLE: ... --> comment
that crawl_tamu.py writes at the top of every HTML file, and back-fills a
real 'url' field into the existing Chroma collection metadata.

This does NOT re-embed anything. It uses collection.update() which only
writes metadata, not vectors.

Usage:
    python patch_chroma_urls.py
"""

import os
import re
import sys
import glob

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import AppConfig
from chromadb import PersistentClient

SOURCE_RE = re.compile(r"<!--\s*SOURCE:\s*(\S+)\s+TITLE:")


def extract_url_from_html(filepath: str) -> str | None:
    """Read only the first line of an HTML file and parse the SOURCE comment."""
    try:
        with open(filepath, encoding="utf-8", errors="ignore") as fh:
            first_line = fh.readline()
        m = SOURCE_RE.search(first_line)
        return m.group(1) if m else None
    except Exception:
        return None


def build_filename_to_url_map(docs_dir: str) -> dict[str, str]:
    """Build a map from html filename (basename) -> original web URL."""
    mapping: dict[str, str] = {}
    for fpath in glob.glob(os.path.join(docs_dir, "*.html")):
        url = extract_url_from_html(fpath)
        if url:
            mapping[os.path.basename(fpath)] = url
    print(f"  Built URL map: {len(mapping)} html files with SOURCE comments")
    return mapping


def patch_collection(chroma_path: str, docs_dir: str, batch_size: int = 500) -> None:
    client = PersistentClient(path=chroma_path)
    collections = client.list_collections()
    if not collections:
        print("ERROR: No collections found in Chroma DB.")
        return

    # Chroma typically uses a single unnamed collection
    col = client.get_collection(collections[0].name)
    print(f"  Collection: '{col.name}'  |  Total chunks: {col.count()}")

    url_map = build_filename_to_url_map(docs_dir)

    offset = 0
    patched = 0
    skipped_no_source = 0
    skipped_already_set = 0

    while True:
        # Fetch a batch of items with their current metadata
        result = col.get(
            limit=batch_size,
            offset=offset,
            include=["metadatas"],
        )
        ids = result["ids"]
        metadatas = result["metadatas"]

        if not ids:
            break

        update_ids = []
        update_metadatas = []

        for chunk_id, meta in zip(ids, metadatas):
            # Skip if already has a real 'url' field
            existing_url = meta.get("url", "")
            if existing_url and existing_url.startswith("http"):
                skipped_already_set += 1
                continue

            # 'source' is the local file path set by DirectoryLoader
            source_path = meta.get("source", "")
            filename = os.path.basename(source_path)
            web_url = url_map.get(filename)

            if not web_url:
                skipped_no_source += 1
                continue

            new_meta = {**meta, "url": web_url}
            update_ids.append(chunk_id)
            update_metadatas.append(new_meta)

        if update_ids:
            col.update(ids=update_ids, metadatas=update_metadatas)
            patched += len(update_ids)

        print(
            f"  Processed {offset + len(ids):>7} / {col.count()}  |  "
            f"Patched: {patched}  |  Already set: {skipped_already_set}  |  "
            f"No source match: {skipped_no_source}"
        )

        offset += batch_size
        if len(ids) < batch_size:
            break

    print(f"\nDone. Patched {patched} chunks with real web URLs.")
    print(f"  Already had URL: {skipped_already_set}")
    print(f"  No matching HTML file found: {skipped_no_source}")


if __name__ == "__main__":
    config = AppConfig()
    docs_dir = config.DOCUMENTS_PATH
    chroma_path = config.DATABASE_PATH

    print("=== Chroma URL Metadata Patch ===")
    print(f"  Chroma DB:    {chroma_path}")
    print(f"  HTML source:  {docs_dir}")
    print()

    if not os.path.exists(chroma_path):
        print("ERROR: Chroma DB path does not exist. Run create_database.py first.")
        sys.exit(1)

    patch_collection(chroma_path, docs_dir)
