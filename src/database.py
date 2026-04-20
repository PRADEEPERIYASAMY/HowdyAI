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
        )

    def search(self, query_text, k=5):
        logger.debug(f"Searching database for query: {query_text}")
        return self.db.similarity_search_with_relevance_scores(query_text, k=k)
