"""
src/search/hybrid_retriever.py  –  Hybrid retrieval: Web Search + Chroma,
                                    fused via Reciprocal Rank Fusion (RRF).

This module ACTIVATES the existing but unused Chroma vector store in the
base codebase (src/database.py + src/embeddings.py) and integrates it into
the live pipeline alongside Brave Search.
"""

import logging
import re
from typing import List, Dict

from src.database import Database
from src.search.brave_search import brave_search_engine
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

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
    dict format used by brave_search_engine so they can be fused.

    Chroma returns: List[Tuple[Document, float]]
    """
    adapted = []
    for doc, score in chroma_results:
        if score < 0.40:
            continue

        # Prefer the patched 'url' field (real web URL); fall back to 'source' (local path)
        url = doc.metadata.get("url", "") or doc.metadata.get("source", "")
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
    Orchestrates Web Search + Chroma dense retrieval and fuses via RRF.
    Web search provider is selected via config.SEARCH_PROVIDER:
        "brave"  (default) — Brave Search API
        "google"           — Google Custom Search JSON API
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
            logger.warning(f"Chroma DB not available ({exc}). Running Brave-only mode.")
            self._chroma_available = False

        if getattr(config, "USE_HYDE", False):
            self.llm = ChatOpenAI(
                model=config.FAST_MODEL,
                temperature=0.0,
                api_key=config.OPENAPI_API_KEY
            )

    def _generate_hyde(self, query: str) -> str:
        """Generate a hypothetical document to improve dense retrieval recall."""
        prompt = (
            "You are a helpful assistant for Texas A&M University students. "
            "Write a factual, two-sentence hypothetical answer to the following question. "
            "Do not include introductory or conversational text, just the raw hypothetical facts."
        )
        try:
            response = self.llm.invoke([
                SystemMessage(content=prompt),
                HumanMessage(content=query)
            ])
            hypothetical = response.content.strip()
            logger.debug(f"HyDE generated: {hypothetical}")
            return hypothetical
        except Exception as exc:
            logger.warning(f"HyDE generation failed: {exc}")
            return query

    def retrieve(self, query: str, broad: bool = False) -> List[dict]:
        """
        Run both retrievers and return RRF-fused results.

        Returns a list of result dicts compatible with the existing
        summarize_search_results_with_llm / rank_results_with_llm pipeline.
        """
        sources_used: List[List[dict]] = []

        chroma_high_confidence = False
        if self._chroma_available and not broad:
            try:
                use_hyde = getattr(self.config, "USE_HYDE", False)
                # if they searched for a specific course code, skip the hyde step or it messes up the search
                if use_hyde:
                    if re.search(r'\b[A-Z]{3,4}\s*\d{3}\b', query, re.IGNORECASE) or len(query.split()) <= 10:
                        use_hyde = False
                        logger.info(f"Skipping HyDE for keyword/course-dense query: {query}")

                raw_top_score = None
                if use_hyde:
                    raw_chroma_check = self.chroma_db.search(query, k=1)
                    raw_top_score = raw_chroma_check[0][1] if raw_chroma_check else None

                search_query = query
                if use_hyde:
                    search_query = self._generate_hyde(query)

                chroma_raw = self.chroma_db.search(search_query, k=self.chroma_num)
                top_score = chroma_raw[0][1] if chroma_raw else None

                raw_str = f"{raw_top_score:.3f}" if raw_top_score is not None else "N/A"
                top_str = f"{top_score:.3f}" if top_score is not None else "N/A"

                # use whichever score is better so we don't accidentally fallback to web search 
                # (hyde sometimes completely bombs on course codes like ARAB 221)
                gate_score = max(
                    s for s in [raw_top_score, top_score] if s is not None
                ) if (raw_top_score is not None or top_score is not None) else None

                # big brain fix: if they asked for SPMT 681, we better make sure 681 is actually in the text
                # otherwise the cosine similarity will just give us SPMT 481 and we fail the eval
                course_codes = re.findall(r'\b[A-Z]{3,4}\s*\d{3}\b', query, re.IGNORECASE)
                has_course_code_miss = False
                if course_codes and chroma_raw:
                    top_chunk_text = chroma_raw[0][0].page_content.lower() if hasattr(chroma_raw[0][0], "page_content") else ""
                    for code in course_codes:
                        num_match = re.search(r'\d{3}', code)
                        if num_match:
                            num = num_match.group()
                            if num not in top_chunk_text:
                                has_course_code_miss = True
                                logger.info(f"Course code mismatch! Query asked for {num} but top chunk didn't have it.")
                                break

                gate_str = f"{gate_score:.3f}" if gate_score is not None else "N/A"
                gate = "YES" if (gate_score is not None and gate_score >= 0.60 and not has_course_code_miss) else "NO"
                if use_hyde:
                    logger.info(
                        f"DIAG | raw_score={raw_str} | hyde_score={top_str} | gate_score={gate_str} | threshold=0.60 | gate_fires={gate}"
                    )
                else:
                    logger.info(
                        f"DIAG | raw_score={raw_str} | gate_score={gate_str} | threshold=0.60 | gate_fires={gate}"
                    )

                if gate == "YES":

                    chroma_high_confidence = True
                    logger.info(f"Chroma high confidence match ({top_score:.3f}). Skipping web search.")
                    # only send 3 results to the cross encoder otherwise it takes forever
                    chroma_raw = chroma_raw[:3]

                chroma_results = chroma_results_to_search_format(chroma_raw)
                if chroma_results:
                    sources_used.append(chroma_results)
                    logger.info(f"Chroma Search: {len(chroma_results)} results")
            except Exception as exc:
                logger.error(f"Chroma search failed: {exc}")

        # ── 2. Web Search (provider-aware) ──────────────────────────────────
        if not chroma_high_confidence:
            try:
                web_results = brave_search_engine(self.config, query, broad=broad) or []
                web_results = web_results[:self.search_num]
                if web_results:
                    sources_used.append(web_results)
                    logger.info(f"Brave Search: {len(web_results)} results")
            except Exception as exc:
                logger.error(f"Brave Search failed: {exc}")

        if not sources_used:
            logger.warning("Both retrievers returned no results.")
            return []

        if len(sources_used) == 1:
            return sources_used[0][:self.fusion_top_n]

        return reciprocal_rank_fusion(sources_used, top_n=self.fusion_top_n)
