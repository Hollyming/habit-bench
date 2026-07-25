"""Operation-level parity checks for the native and adapted A-MEM layers."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path


AMEM_SOURCE = Path(__file__).resolve().parents[1] / "methods" / "amem" / "A-mem"
if str(AMEM_SOURCE) not in sys.path:
    # Keep the project root ahead of the vendored directory so its top-level
    # ``utils`` package is not shadowed by A-MEM's standalone ``utils.py``.
    sys.path.append(str(AMEM_SOURCE))

from memory_layer import AgenticMemorySystem, MemoryNote  # noqa: E402
from memory_layer_robust import (  # noqa: E402
    RobustAgenticMemorySystem,
    RobustMemoryNote,
)


class _NativeLLM:
    def get_completion(self, prompt, response_format=None, temperature=0.7):
        del prompt, response_format, temperature
        return json.dumps(
            {
                "should_evolve": True,
                "actions": ["strengthen", "update_neighbor"],
                "suggested_connections": [0],
                "tags_to_update": ["linked", "clinical"],
                "new_context_neighborhood": ["updated zero", "updated one"],
                "new_tags_neighborhood": [["zero"], ["one"]],
            }
        )


class _RobustLLM:
    def __init__(self):
        self.responses = iter(
            [
                "DECISION: STRENGTHEN_AND_UPDATE\nREASON: related",
                "CONNECTIONS: 0\nTAGS: linked, clinical",
                (
                    "NEIGHBOR 0:\nCONTEXT: updated zero\nTAGS: zero\n\n"
                    "NEIGHBOR 1:\nCONTEXT: updated one\nTAGS: one"
                ),
            ]
        )

    def get_completion(self, prompt, temperature=0.7):
        del prompt, temperature
        return next(self.responses)


class _Controller:
    def __init__(self, llm):
        self.llm = llm


class _Retriever:
    def search(self, query, k=5):
        del query, k
        return [1, 0]


def _note(note_class, note_id, context, tags):
    return note_class(
        content=f"content {note_id}",
        id=note_id,
        keywords=[f"keyword-{note_id}"],
        links=[],
        timestamp="2024-01-01 00:00:00",
        last_accessed="2024-01-01 00:00:00",
        context=context,
        category="test",
        tags=tags,
    )


def test_native_and_robust_amem_have_matching_evolution_and_retrieval_trace():
    native = AgenticMemorySystem.__new__(AgenticMemorySystem)
    native.memories = {
        "n0": _note(MemoryNote, "n0", "original zero", ["old-zero"]),
        "n1": _note(MemoryNote, "n1", "original one", ["old-one"]),
    }
    native.llm_controller = _Controller(_NativeLLM())
    native.evolution_system_prompt = (
        "{context} {content} {keywords} {nearest_neighbors_memories} "
        "{neighbor_number}"
    )

    robust = RobustAgenticMemorySystem.__new__(RobustAgenticMemorySystem)
    robust.memories = {
        "n0": _note(RobustMemoryNote, "n0", "original zero", ["old-zero"]),
        "n1": _note(RobustMemoryNote, "n1", "original one", ["old-one"]),
    }
    robust.llm_controller = _Controller(_RobustLLM())
    robust.max_context_chars = 100000

    fixed_neighbors = lambda self, query, k=5: ("two neighbors", [0, 1])
    native.find_related_memories = types.MethodType(fixed_neighbors, native)
    robust.find_related_memories = types.MethodType(fixed_neighbors, robust)

    native_new = _note(MemoryNote, "new", "new context", ["new-old"])
    robust_new = _note(RobustMemoryNote, "new", "new context", ["new-old"])
    native_evolved, native_new = native.process_memory(native_new)
    robust_evolved, robust_new = robust.process_memory(robust_new)

    assert native_evolved is robust_evolved is True
    assert native_new.links == robust_new.links == [0]
    assert native_new.tags == robust_new.tags == ["linked", "clinical"]
    assert [note.context for note in native.memories.values()] == [
        note.context for note in robust.memories.values()
    ] == ["updated zero", "updated one"]
    assert [note.tags for note in native.memories.values()] == [
        note.tags for note in robust.memories.values()
    ] == [["zero"], ["one"]]

    native.retriever = _Retriever()
    robust.retriever = _Retriever()
    native.find_related_memories = types.MethodType(
        AgenticMemorySystem.find_related_memories, native
    )
    robust.find_related_memories = types.MethodType(
        RobustAgenticMemorySystem.find_related_memories, robust
    )
    native_context, native_indices = native.find_related_memories("query", k=2)
    robust_context, robust_indices = robust.find_related_memories("query", k=2)

    assert native_indices == robust_indices == [1, 0]
    assert native_context == robust_context
