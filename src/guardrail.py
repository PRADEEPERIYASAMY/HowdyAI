"""
src/guardrail.py  –  LLM-based out-of-scope query guard.

Uses a lightweight gpt-4o-mini call to classify whether the query is
TAMU-related before entering the expensive full pipeline.
"""

import logging

from src.language_models.openai_language_model import OpenAILanguageModel

logger = logging.getLogger(__name__)

OUT_OF_SCOPE_REPLY = (
    "I'm HowdyAI, a chatbot specifically for Texas A&M University. "
    "Your question doesn't appear to be related to TAMU. "
    "Please ask me about TAMU courses, policies, admissions, campus life, "
    "research, or any other university topic. Howdy! 🤠"
)


class QueryGuardrail:
    """Classifies queries as in-scope (TAMU) or out-of-scope."""

    def __init__(self, config):
        self.model = OpenAILanguageModel(config.GUARDRAIL_TEMPLATE_PATH)
        logger.debug("QueryGuardrail initialised.")

    def check(self, query: str) -> tuple[bool, str]:
        """
        Returns
        -------
        (is_in_scope: bool, message: str)
        """
        try:
            prompt = self.model.generate_prompt(question=query, history="")
            response = self.model.invoke(prompt)
            label = response.content.strip()
            logger.debug(f"Guardrail label for {query[:60]!r}: {label}")
        except Exception as exc:
            logger.warning(f"Guardrail API error: {exc}; defaulting to IN_SCOPE")
            return True, ""

        if "OUT_OF_SCOPE" in label:
            logger.info(f"Query blocked by guardrail: {query[:60]!r}")
            return False, OUT_OF_SCOPE_REPLY

        return True, ""
