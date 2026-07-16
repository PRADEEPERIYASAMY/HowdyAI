"""
create_database.py  –  Build the Chroma vector store from crawled TAMU HTML.

Enhancement over base code:
  * Replaces RecursiveCharacterTextSplitter(chunk_size=300) with semantic
    chunking: consecutive paragraphs are merged while their cosine similarity
    exceeds SEMANTIC_THRESHOLD; when it drops below, a new chunk begins.
  * Everything else (DirectoryLoader, Chroma, OpenAIEmbeddings) is the same
    as the base codebase.

Usage:
    python create_database.py              # uses paths from config.py
    python create_database.py --reset      # wipe and rebuild from scratch
"""

import logging
import logging.config
import os
import re
import shutil
import argparse
from typing import List

import numpy as np
from langchain_community.document_loaders import DirectoryLoader, BSHTMLLoader
from langchain_chroma import Chroma
from langchain_core.documents import Document as LCDocument
from openai import OpenAI

from config import AppConfig
from src.embeddings import Embeddings

logger = logging.getLogger(__name__)

# ── Semantic chunking config ──────────────────────────────────────────────────
SEMANTIC_THRESHOLD = 0.80   # cosine sim below this → start new chunk
MIN_CHUNK_CHARS = 150
MAX_CHUNK_CHARS = 1200
OVERLAP_SENTENCES = 1       # sentences carried into next chunk for continuity
EMBEDDING_BATCH = 100       # max sentences per OpenAI embedding batch


# ── Helpers ───────────────────────────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def split_into_sentences(text: str) -> List[str]:
    """Simple rule-based sentence splitter, with a hard length cap for embedding limits."""
    pattern = r"(?<=[.!?])\s+(?=[A-Z])"
    raw_sentences = re.split(pattern, text.strip())
    sentences = []
    for s in raw_sentences:
        s = s.strip()
        if len(s) < 20:
            continue
        # Hard cap to avoid exceeding OpenAI 8192 token limit (~32000 chars)
        max_chars = 15000
        while len(s) > max_chars:
            sentences.append(s[:max_chars])
            s = s[max_chars:]
        if len(s) > 20:
            sentences.append(s)
    return sentences


def embed_sentences(sentences: List[str], client: OpenAI) -> List[np.ndarray]:
    """Batch-embed sentences using text-embedding-3-large."""
    if not sentences:
        return []
    embeddings = []
    for i in range(0, len(sentences), EMBEDDING_BATCH):
        batch = sentences[i:i + EMBEDDING_BATCH]
        resp = client.embeddings.create(model="text-embedding-3-large", input=batch)
        embeddings.extend([np.array(item.embedding) for item in resp.data])
    return embeddings


def semantic_chunk(text: str, client: OpenAI) -> List[str]:
    """
    Split text into semantically coherent chunks.

    Algorithm:
      1. Split into sentences.
      2. Batch-embed all sentences.
      3. Walk forward; start a new chunk whenever cosine sim of consecutive
         sentences drops below SEMANTIC_THRESHOLD or chunk exceeds MAX_CHUNK_CHARS.
      4. Carry OVERLAP_SENTENCES of the previous chunk into the next.
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return []
    if len(sentences) == 1:
        return [sentences[0]] if len(sentences[0]) >= MIN_CHUNK_CHARS else []

    embeddings = embed_sentences(sentences, client)

    chunks: List[str] = []
    current: List[str] = [sentences[0]]

    for i in range(1, len(sentences)):
        sim = cosine_similarity(embeddings[i - 1], embeddings[i])
        current_text = " ".join(current)

        if sim < SEMANTIC_THRESHOLD or len(current_text) > MAX_CHUNK_CHARS:
            if len(current_text) >= MIN_CHUNK_CHARS:
                chunks.append(current_text)
                current = current[-OVERLAP_SENTENCES:] + [sentences[i]]
            else:
                current.append(sentences[i])
        else:
            current.append(sentences[i])

    last = " ".join(current)
    if last and len(last) >= MIN_CHUNK_CHARS:
        chunks.append(last)
    elif chunks and last:
        chunks[-1] += " " + last

    return chunks


# ── DatabaseGenerator ─────────────────────────────────────────────────────────

class DatabaseGenerator:
    def __init__(self, data_path: str, chroma_path: str):
        self.data_path = data_path
        self.db_path = chroma_path
        self.embeddings = Embeddings()
        self.openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def generate_database(self, reset: bool = False) -> None:
        if reset and os.path.exists(self.db_path):
            self._safe_rmtree(self.db_path)
            logger.info(f"Cleared existing Chroma store at {self.db_path}")

        documents = self.load_documents()
        logger.info(f"Loaded {len(documents)} HTML documents from {self.data_path}")

        chunks = self.semantic_split(documents)
        logger.info(f"Created {len(chunks)} semantic chunks")

        self.save_to_chroma(chunks)

    @staticmethod
    def _safe_rmtree(path: str, retries: int = 5, delay: float = 1.0) -> None:
        """Delete a directory tree, retrying on Windows PermissionError (file locks)."""
        import stat
        import time

        def on_error(func, fpath, exc_info):
            # Make read-only files writable and retry
            try:
                os.chmod(fpath, stat.S_IWRITE)
                func(fpath)
            except Exception:
                pass

        for attempt in range(retries):
            try:
                shutil.rmtree(path, onerror=on_error)
                return
            except PermissionError as e:
                if attempt < retries - 1:
                    logger.warning(
                        f"DB directory locked (attempt {attempt+1}/{retries}), "
                        f"retrying in {delay}s. Stop the Streamlit app to release locks."
                    )
                    time.sleep(delay)
                else:
                    raise RuntimeError(
                        f"Could not delete {path} after {retries} attempts. "
                        f"Please stop the Streamlit app and re-run."
                    ) from e

    def load_documents(self) -> list:
        loader = DirectoryLoader(
            self.data_path,
            glob="**/*.html",
            loader_cls=BSHTMLLoader,
            loader_kwargs={"open_encoding": "utf-8"},
            silent_errors=True,
        )
        return loader.load()

    def semantic_split(self, documents) -> List[LCDocument]:
        """Replace fixed-size splitter with semantic chunking."""
        all_chunks: List[LCDocument] = []

        for i, doc in enumerate(documents):
            logger.debug(f"  Chunking doc {i+1}/{len(documents)}: {doc.metadata.get('source', '')[:60]}")
            try:
                text_chunks = semantic_chunk(doc.page_content, self.openai_client)
            except Exception as exc:
                logger.warning(f"  Semantic chunking failed for doc {i+1}: {exc}; skipping.")
                continue

            for j, chunk in enumerate(text_chunks):
                all_chunks.append(
                    LCDocument(
                        page_content=chunk,
                        metadata={**doc.metadata, "chunk_index": j},
                    )
                )

        return all_chunks

    def save_to_chroma(self, chunks: List[LCDocument]) -> None:
        if not chunks:
            return
        batch_size = 500
        total_batches = (len(chunks) - 1) // batch_size + 1
        
        # Initialize with first batch to create/load the collection
        db = Chroma.from_documents(
            chunks[:batch_size],
            self.embeddings.embedding_function,
            persist_directory=self.db_path,
            collection_metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"Saved batch 1/{total_batches} ({len(chunks[:batch_size])} chunks)")
        
        # Add remaining batches
        for i in range(batch_size, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            db.add_documents(batch)
            logger.info(f"Saved batch {i//batch_size + 1}/{total_batches} ({len(batch)} chunks)")
            
        logger.info(f"Successfully saved {len(chunks)} total chunks to {self.db_path}.")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build the HowdyAI Chroma vector store with semantic chunking."
    )
    parser.add_argument("--reset", action="store_true",
                        help="Delete and rebuild the Chroma DB from scratch")
    args = parser.parse_args()

    config = AppConfig()
    logging.config.dictConfig(config.logging_config)

    logger.info(f"Generating database from {config.DOCUMENTS_PATH} -> {config.DATABASE_PATH}")
    gen = DatabaseGenerator(config.DOCUMENTS_PATH, config.DATABASE_PATH)
    gen.generate_database(reset=args.reset)
    logger.info("Database generation complete.")
