"""Schemas for the RAG question-answering flow."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class GroundingStatus(str, Enum):
    """How well an answer is grounded in the indexed documents.

    The honesty contract: the system reports one of these so the UI can be
    explicit about whether it actually answered, declined for lack of
    coverage, or had nothing to draw on — never dressing up a guess as fact.
    """

    #: Retrieved sources matched the question well; an answer was produced.
    ANSWERED = "answered"
    #: Sources were retrieved but the best match is too weak to answer from;
    #: the closest related material is still returned for the user to judge.
    INSUFFICIENT_CONTEXT = "insufficient_context"
    #: Nothing was retrieved — no indexed documents (or none in scope).
    NO_DOCUMENTS = "no_documents"


class AskRequest(BaseModel):
    """Request body for ``POST /ask``."""

    question: str = Field(min_length=1, description="The question to answer.")
    document_ids: list[str] | None = Field(
        default=None,
        description="Restrict retrieval to these document IDs (all, if omitted).",
    )
    conversation_id: str | None = Field(
        default=None,
        description=(
            "Continue an existing conversation, grounding follow-ups in its "
            "prior turns. Omit to start a new conversation."
        ),
    )


class Source(BaseModel):
    """A retrieved chunk offered as grounding for an answer.

    ``citation`` is the 1-based ``[n]`` number the LLM is told to cite, so the
    frontend can map a ``[n]`` reference in the answer back to this source.
    """

    citation: int = Field(description="1-based [n] reference number for this source.")
    document_id: str = Field(description="ID of the document this chunk was retrieved from.")
    filename: str
    chunk_index: int
    page: int | None = Field(
        default=None,
        description="1-based page the passage starts on (PDF); None for TXT/MD.",
    )
    char_start: int = Field(
        default=0, description="Start offset of the passage in the document text."
    )
    char_end: int = Field(
        default=0, description="End offset of the passage in the document text."
    )
    text: str = Field(
        description="The full retrieved chunk text (not a truncated snippet)."
    )
    score: float = Field(description="Similarity score (higher is more relevant).")


class AnswerResult(BaseModel):
    """Result of asking a question against the indexed documents."""

    answer: str | None = Field(
        default=None,
        description="Grounded answer, or None when generation is disabled.",
    )
    sources: list[Source] = Field(
        default_factory=list,
        description="Retrieved chunks used as context, in rank order.",
    )
    generation_enabled: bool = Field(
        description="Whether an LLM produced (or could produce) an answer."
    )
    status: GroundingStatus = Field(
        default=GroundingStatus.ANSWERED,
        description="Grounding outcome: answered, insufficient_context, or no_documents.",
    )
    coverage: float = Field(
        default=0.0,
        description=(
            "How well the top sources match the question, as the best source "
            "similarity clamped to 0..1. 0 when nothing was retrieved."
        ),
    )
    low_confidence: bool = Field(
        default=False,
        description=(
            "True when an answer was produced but cites no valid source — a "
            "best-effort signal that the model may have gone beyond the context."
        ),
    )
    cited: list[int] = Field(
        default_factory=list,
        description="Citation numbers actually referenced by the answer text.",
    )
    conversation_id: str = Field(
        default="",
        description="ID of the conversation this turn belongs to; echo it back to continue.",
    )
    standalone_question: str | None = Field(
        default=None,
        description=(
            "The follow-up rewritten as a standalone question for retrieval, "
            "when prior turns existed. None on the first turn, when the "
            "follow-up needed no rewriting, or when generation is disabled."
        ),
    )
