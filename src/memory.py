"""
src/memory.py  –  Sliding-window conversation memory for multi-turn sessions.
"""

import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


class ConversationMemory:
    """Maintains a rolling window of (user, assistant) turn pairs."""

    def __init__(self, max_turns: int = 6):
        self.max_turns = max_turns
        self._history: List[Tuple[str, str]] = []   # [(user_msg, assistant_msg), ...]

    def add_turn(self, user_message: str, assistant_message: str) -> None:
        self._history.append((user_message, assistant_message))
        if len(self._history) > self.max_turns:
            self._history = self._history[-self.max_turns:]
        logger.debug(f"Memory now has {len(self._history)} turn(s).")

    def clear(self) -> None:
        self._history.clear()
        logger.info("Conversation memory cleared.")

    def as_string(self) -> str:
        """Return history as plain-text block for prompt injection."""
        if not self._history:
            return "(no prior conversation)"
        lines = []
        for user_msg, asst_msg in self._history:
            lines.append(f"User: {user_msg}")
            lines.append(f"Assistant: {asst_msg}")
        return "\n".join(lines)

    def as_messages(self) -> List[dict]:
        """Return history as OpenAI-style message dicts."""
        msgs = []
        for user_msg, asst_msg in self._history:
            msgs.append({"role": "user", "content": user_msg})
            msgs.append({"role": "assistant", "content": asst_msg})
        return msgs

    def is_empty(self) -> bool:
        return len(self._history) == 0

    def __len__(self) -> int:
        return len(self._history)
