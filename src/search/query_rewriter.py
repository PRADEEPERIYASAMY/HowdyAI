"""
src/search/query_rewriter.py  –  LLM-based query rewriting for improved retrieval.

Reformulates the raw user question into a search-engine-optimised query
with TAMU-specific terminology so both Brave Search and Chroma return better docs.
"""

import logging

from src.language_models.openai_language_model import OpenAILanguageModel

logger = logging.getLogger(__name__)


class QueryRewriter:
    """Rewrites user queries for improved retrieval recall."""

    def __init__(self, config):
        self.model = OpenAILanguageModel(config.REWRITE_TEMPLATE_PATH, model_name=config.FAST_MODEL)
        logger.debug("QueryRewriter initialised.")

    def rewrite(self, query: str, history: str = "") -> str:
        """
        Parameters
        ----------
        query   : Raw user question.
        history : Plain-text conversation history string.

        Returns
        -------
        Rewritten search query string (falls back to original on error).
        """
        try:
            prompt = self.model.generate_prompt(
                question=query,
                history=history if history else "(none)",
            )
            response = self.model.invoke(prompt)
            rewritten = response.content.strip()

            if not rewritten or len(rewritten) > 300:
                logger.warning("Query rewriter returned empty/too-long output; using original.")
                return query

            logger.info(f"Query rewritten: {query!r} → {rewritten!r}")
            return rewritten

        except Exception as exc:
            logger.warning(f"QueryRewriter error: {exc}; using original query.")
            return query
