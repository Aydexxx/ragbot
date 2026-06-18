"""Schema for a single turn in a multi-turn conversation."""

from __future__ import annotations

from pydantic import BaseModel


class ConversationTurn(BaseModel):
    """One question/answer exchange, kept as grounding context for follow-ups."""

    question: str
    answer: str | None = None
    #: The standalone form of ``question`` used for retrieval, when it was
    #: rewritten from a follow-up. None if asked standalone (no prior turns)
    #: or generation was disabled at the time.
    standalone_question: str | None = None
