import logging
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

class CrossEncoderRanker:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        logger.info(f"Loading cross-encoder model: {model_name}...")
        try:
            self.model = CrossEncoder(model_name, local_files_only=True)
        except Exception:
            self.model = CrossEncoder(model_name)
        logger.info("Cross-encoder model loaded.")

    def rank(self, query: str, documents: List[Dict[str, Any]], text_key: str = "truncated_html", top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Rerank a list of documents based on semantic relevance to the query.
        
        Args:
            query: The user's query string.
            documents: A list of result dictionaries from the retriever.
            text_key: The key in each dictionary containing the text to be scored.
            top_k: The number of top results to return.
            
        Returns:
            The reranked and truncated list of documents.
        """
        if not documents:
            return []

        # Prepare pairs for cross-encoder: [(query, doc1), (query, doc2), ...]
        pairs = []
        for doc in documents:
            text = doc.get(text_key, "")
            pairs.append((query, text))

        logger.info(f"Reranking {len(documents)} documents with Cross-Encoder...")
        # Predict scores
        scores = self.model.predict(pairs)

        # Attach scores to documents
        for i, doc in enumerate(documents):
            doc["cross_encoder_score"] = float(scores[i])

        # Sort descending by score
        ranked_docs = sorted(documents, key=lambda x: x["cross_encoder_score"], reverse=True)

        logger.debug("Cross-Encoder scores: " + ", ".join([f"{d['cross_encoder_score']:.2f}" for d in ranked_docs]))

        # Return top_k
        return ranked_docs[:top_k]
