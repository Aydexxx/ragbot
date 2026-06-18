"""Question-answering endpoint: retrieve relevant chunks, then ground an answer."""

from __future__ import annotations

from uuid import uuid4

from fastapi import APIRouter

from app.api.dependencies import ConversationStoreDep, RagServiceDep
from app.models.answer import AnswerResult, AskRequest
from app.models.conversation import ConversationTurn

router = APIRouter(tags=["ask"])


@router.post("/ask", response_model=AnswerResult)
async def ask(
    request: AskRequest, rag: RagServiceDep, conversations: ConversationStoreDep
) -> AnswerResult:
    """Answer a question grounded in indexed documents, with citations.

    Always returns 200. Retrieval finding nothing (no documents indexed, or
    none match the optional ``document_ids`` filter) yields an empty
    ``sources`` list and a clear "I don't know" ``answer`` — never an error,
    since there's nothing wrong with the request. When generation is
    disabled, ``answer`` is ``None`` but ``sources`` are still populated
    whenever something was retrieved.

    Pass ``conversation_id`` to continue a conversation: prior turns ground
    follow-up reformulation (see ``RagService.answer``). Omit it to start a
    new one — the server mints an ID and returns it on the response for the
    client to echo back on the next turn.
    """
    conversation_id = request.conversation_id or str(uuid4())
    history = conversations.get_history(conversation_id)

    result = await rag.answer(
        request.question, document_ids=request.document_ids, history=history
    )

    conversations.append(
        conversation_id,
        ConversationTurn(
            question=request.question,
            answer=result.answer,
            standalone_question=result.standalone_question,
        ),
    )

    return result.model_copy(update={"conversation_id": conversation_id})
