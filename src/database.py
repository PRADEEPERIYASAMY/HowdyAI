import logging

from langchain_chroma import Chroma
from src.embeddings import Embeddings

logger = logging.getLogger(__name__)


class Database:
    def __init__(self, db_path):
        logger.debug(f"Initializing database with path: {db_path}")
        embeddings = Embeddings()
        self.db = Chroma(
            persist_directory=db_path,
            embedding_function=embeddings.embedding_function,
            collection_metadata={"hnsw:space": "cosine"}
        )

    def search(self, query_text, k=5):
        logger.debug(f"Searching database for query: {query_text}")
        # Request k+5 extra to account for post-delete HNSW/SQLite sync lag:
        # deleted vector IDs can still surface in HNSW query results with
        # None document text until the index compacts, causing LangChain's
        # Document validator to throw. We fetch extra, filter None, return top-k.
        try:
            raw = self.db.similarity_search_with_relevance_scores(query_text, k=k + 5)
            return [(doc, score) for doc, score in raw if doc.page_content is not None][:k]
        except Exception as exc:
            logger.warning(f"Chroma search raised validation error (likely stale HNSW entry): {exc}")
            return []

