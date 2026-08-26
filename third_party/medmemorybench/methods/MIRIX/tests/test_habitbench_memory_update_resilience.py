from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from mirix.functions.function_sets.memory_tools import (
    episodic_memory_merge,
    trigger_memory_update,
)
from mirix.orm.errors import NoResultFound
from mirix.schemas.enums import MessageRole
from mirix.schemas.message import MessageCreate
from mirix.schemas.mirix_message_content import TextContent


async def test_missing_episodic_merge_target_is_preserved_as_new_event(
    monkeypatch,
):
    manager = SimpleNamespace(
        update_event=AsyncMock(side_effect=NoResultFound("missing")),
        insert_event=AsyncMock(return_value=SimpleNamespace(id="ep_new")),
    )
    agent = SimpleNamespace(
        episodic_memory_manager=manager,
        actor=SimpleNamespace(organization_id="org"),
        agent_state=SimpleNamespace(parent_id="parent", id="child"),
        occurred_at=datetime(2026, 8, 26, tzinfo=timezone.utc),
        filter_tags=None,
        use_cache=True,
        user_id="user",
    )
    monkeypatch.setenv("MIRIX_NORMALIZE_MISSING_UPDATE_IDS", "1")

    response = await episodic_memory_merge(
        agent,
        event_id="ep_semantic_suffix",
        combined_summary="complete summary",
        combined_details="complete details",
    )

    assert response == (
        "Events inserted! Now you need to check if there are repeated events "
        "shown in the system prompt."
    )
    manager.insert_event.assert_awaited_once()
    inserted = manager.insert_event.await_args.kwargs
    assert inserted["timestamp"] == agent.occurred_at
    assert inserted["event_type"] == "user_message"
    assert inserted["event_actor"] == "user"
    assert inserted["summary"] == "complete summary"
    assert inserted["details"] == "complete details"


async def test_missing_episodic_merge_target_still_fails_without_adapter_flag(
    monkeypatch,
):
    manager = SimpleNamespace(
        update_event=AsyncMock(side_effect=NoResultFound("missing")),
        insert_event=AsyncMock(),
    )
    agent = SimpleNamespace(
        episodic_memory_manager=manager,
        actor=SimpleNamespace(organization_id="org"),
        agent_state=SimpleNamespace(parent_id="parent", id="child"),
        occurred_at=datetime(2026, 8, 26, tzinfo=timezone.utc),
    )
    monkeypatch.delenv("MIRIX_NORMALIZE_MISSING_UPDATE_IDS", raising=False)

    with pytest.raises(NoResultFound):
        await episodic_memory_merge(
            agent,
            event_id="ep_missing",
            combined_summary="summary",
            combined_details="details",
        )
    manager.insert_event.assert_not_awaited()


async def test_parallel_memory_children_receive_isolated_retrieval_snapshots():
    observed: dict[str, dict] = {}
    both_started = asyncio.Event()

    class MockMemoryAgent:
        def __init__(self, agent_state, **kwargs):
            self.agent_state = agent_state

        async def step(self, *, retrieved_memories, **kwargs):
            memory_type = self.agent_state.agent_type.removesuffix(
                "_memory_agent"
            )
            observed[memory_type] = retrieved_memories
            retrieved_memories[memory_type]["child_marker"] = memory_type
            if len(observed) == 2:
                both_started.set()
            await asyncio.wait_for(both_started.wait(), timeout=1)

    child_states = [
        SimpleNamespace(
            id="episodic-child", agent_type="episodic_memory_agent", name="episodic"
        ),
        SimpleNamespace(
            id="semantic-child", agent_type="semantic_memory_agent", name="semantic"
        ),
    ]
    parent_retrieval = {
        "key_words": "travel",
        "episodic": {"text": "no exposed id"},
        "semantic": {"text": "no exposed id"},
    }
    parent = SimpleNamespace(
        agent_manager=SimpleNamespace(
            list_agents=AsyncMock(return_value=child_states)
        ),
        agent_state=SimpleNamespace(id="parent"),
        actor=SimpleNamespace(id="client"),
        user=SimpleNamespace(id="user"),
        interface=SimpleNamespace(),
    )
    message = MessageCreate(
        role=MessageRole.user,
        content=[TextContent(text="update travel memory")],
    )

    with patch("mirix.agent.EpisodicMemoryAgent", MockMemoryAgent), patch(
        "mirix.agent.SemanticMemoryAgent", MockMemoryAgent
    ), patch(
        "mirix.functions.function_sets.memory_tools.get_langfuse_client",
        return_value=None,
    ):
        await trigger_memory_update(
            parent,
            user_message={
                "message": message,
                "chaining": False,
                "retrieved_memories": parent_retrieval,
            },
            memory_types=["episodic", "semantic"],
        )

    assert observed["episodic"] is not observed["semantic"]
    assert "child_marker" not in observed["episodic"]["semantic"]
    assert "child_marker" not in observed["semantic"]["episodic"]
    assert "child_marker" not in parent_retrieval["episodic"]
    assert "child_marker" not in parent_retrieval["semantic"]
