"""Tests for the in-memory multi-turn conversation history store."""

from __future__ import annotations

from app.models.conversation import ConversationTurn
from app.services.conversation import ConversationStore


def test_get_history_empty_for_unknown_conversation() -> None:
    store = ConversationStore()
    assert store.get_history("nope") == []


def test_append_then_get_history_returns_turns_in_order() -> None:
    store = ConversationStore()
    store.append("c1", ConversationTurn(question="q1", answer="a1"))
    store.append("c1", ConversationTurn(question="q2", answer="a2"))

    history = store.get_history("c1")

    assert [t.question for t in history] == ["q1", "q2"]


def test_conversations_are_isolated_by_id() -> None:
    store = ConversationStore()
    store.append("c1", ConversationTurn(question="q1", answer="a1"))
    store.append("c2", ConversationTurn(question="q2", answer="a2"))

    assert [t.question for t in store.get_history("c1")] == ["q1"]
    assert [t.question for t in store.get_history("c2")] == ["q2"]


def test_history_is_bounded_to_max_turns() -> None:
    store = ConversationStore(max_turns=3)
    for i in range(5):
        store.append("c1", ConversationTurn(question=f"q{i}", answer=f"a{i}"))

    history = store.get_history("c1")

    assert [t.question for t in history] == ["q2", "q3", "q4"]


def test_get_history_returns_a_copy_not_a_live_reference() -> None:
    store = ConversationStore()
    store.append("c1", ConversationTurn(question="q1", answer="a1"))

    history = store.get_history("c1")
    history.append(ConversationTurn(question="mutated", answer=None))

    assert len(store.get_history("c1")) == 1
