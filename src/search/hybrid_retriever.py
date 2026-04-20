"""
src/search/hybrid_retriever.py  –  Hybrid retrieval: Web Search + Chroma,
                                    fused via Reciprocal Rank Fusion (RRF).

This module ACTIVATES the existing but unused Chroma vector store in the
base codebase (src/database.py + src/embeddings.py) and integrates it into
the live pipeline alongside DuckDuckGo Search.
"""

import logging
from typing import List, Dict

from src.database import Database
from src.search.google_search import duckduckgo_search_engine

logger = logging.getLogger(__name__)


# ── Reciprocal Rank Fusion ────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    ranked_lists: List[List[dict]],
    k: int = 60,
    top_n: int = 8,
) -> List[dict]:
    """
    Merge multiple ranked result lists using RRF.

    Each list contains dicts with at minimum a 'url' key.
    RRF score = Σ 1 / (k + rank_i)   (summed over all lists)
    """
    scores: Dict[str, float] = {}
    url_to_doc: Dict[str, dict] = {}

    for ranked_list in ranked_lists:
        for rank, doc in enumerate(ranked_list, start=1):
            key = doc.get("url", doc.get("metadata", {}).get("source", ""))
            if not key:
                continue
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in url_to_doc:
                url_to_doc[key] = doc

    sorted_keys = sorted(scores, key=lambda x: scores[x], reverse=True)
    fused = [url_to_doc[k] for k in sorted_keys[:top_n]]
    logger.debug(f"RRF fusion: {sum(len(l) for l in ranked_lists)} → {len(fused)} docs")
    return fused


# ── Chroma result adapter ─────────────────────────────────────────────────────

def chroma_results_to_search_format(chroma_results) -> List[dict]:
    """
    Convert LangChain Chroma similarity_search results into the same
    dict format used by google_custom_search_engine so they can be fused.

    Chroma returns: List[Tuple[Document, float]]
    """
    adapted = []
    for doc, score in chroma_results:
        if score < 0.40:
            continue
            
        url = doc.metadata.get("source", "")
        title = doc.metadata.get("title", url)
        adapted.append({
            "url": url,
            "title": title,
            "description": doc.page_content[:200],
            "metadata": {"content": doc.page_content},
            "chroma_score": score,
            "_source": "chroma",
        })
    return adapted


# ── Hybrid Retriever ──────────────────────────────────────────────────────────

class HybridRetriever:
    """
    Orchestrates Google Search + Chroma dense retrieval and fuses via RRF.
    Falls back gracefully if Chroma DB is not yet built.
    """

    def __init__(self, config):
        self.config = config
        self.search_num = config.NUM_SEARCH_RESULTS
        self.chroma_num = config.CHROMA_NUM_RESULTS
        self.fusion_top_n = config.FUSION_TOP_N

        # Activate the existing Chroma database (was unused in base code)
        try:
            self.chroma_db = Database(config.CHROMA_PATH)
            logger.info(f"Chroma DB loaded from {config.CHROMA_PATH}")
            self._chroma_available = True
        except Exception as exc:
            logger.warning(f"Chroma DB not available ({exc}). Running Google-only mode.")
            self._chroma_available = False

    def retrieve(self, query: str) -> List[dict]:
        """
        Run both retrievers and return RRF-fused results.

        Returns a list of result dicts compatible with the existing
        summarize_search_results_with_llm / rank_results_with_llm pipeline.
        """
        sources_used: List[List[dict]] = []

        # ── DuckDuckGo Web Search ─────────────────────────────────────────
        try:
            ddg_results = duckduckgo_search_engine(self.config, query) or []
            ddg_results = ddg_results[:self.search_num]
            if ddg_results:
                sources_used.append(ddg_results)
                logger.info(f"DuckDuckGo Search: {len(ddg_results)} results")
        except Exception as exc:
            logger.error(f"DuckDuckGo Search failed: {exc}")

        # ── Chroma (local dense) ──────────────────────────────────────────
        if self._chroma_available:
            try:
                chroma_raw = self.chroma_db.search(query, k=self.chroma_num)
                chroma_results = chroma_results_to_search_format(chroma_raw)
                if chroma_results:
                    sources_used.append(chroma_results)
                    logger.info(f"Chroma Search: {len(chroma_results)} results")
            except Exception as exc:
                logger.error(f"Chroma search failed: {exc}")

        if not sources_used:
            logger.warning("Both retrievers returned no results.")
            return []

        if len(sources_used) == 1:
            return sources_used[0][:self.fusion_top_n]

        return reciprocal_rank_fusion(sources_used, top_n=self.fusion_top_n)
