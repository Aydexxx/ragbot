"""In-memory store for short-lived multi-turn conversation history.

Keyed by an opaque ``conversation_id`` the API layer mints on the first turn
and the client echoes back on follow-ups. Not persisted to disk — history
only needs to survive the process lifetime to ground a back-and-forth.
"""

from __future__ import annotations

import threading

from app.models.conversation import ConversationTurn

#: Oldest turns beyond this are dropped, bounding both memory and the size of
#: the history fed into reformulation prompts.
MAX_TURNS = 6


class ConversationStore:
    """Thread-safe, bounded history per conversation ID."""

    def __init__(self, max_turns: int = MAX_TURNS) -> None:
        self._max_turns = max_turns
        self._lock = threading.Lock()
        self._conversations: dict[str, list[ConversationTurn]] = {}

    def get_history(self, conversation_id: str) -> list[ConversationTurn]:
        """Return the turns recorded so far, oldest first. Empty if unknown."""
        with self._lock:
            return list(self._conversations.get(conversation_id, []))

    def append(self, conversation_id: str, turn: ConversationTurn) -> None:
        """Record a turn, trimming the oldest ones beyond ``max_turns``."""
        with self._lock:
            history = self._conversations.setdefault(conversation_id, [])
            history.append(turn)
            if len(history) > self._max_turns:
                del history[: len(history) - self._max_turns]
