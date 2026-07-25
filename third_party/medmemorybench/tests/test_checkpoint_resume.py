from pathlib import Path

from benchmarks.medmemorybench.checkpoint import MedMemoryBenchCheckpointManager


def _manager(root: Path) -> MedMemoryBenchCheckpointManager:
    return MedMemoryBenchCheckpointManager(
        method_name="mem0",
        model_name="Qwen3-8B",
        checkpoint_dir=root,
        config_hash="fixed",
    )


def test_restarting_same_persona_preserves_completed_queries(tmp_path):
    first = _manager(tmp_path)
    first.create(total_personas=2, total_queries=3, evaluation_mode="independent")
    first.start_persona(1)
    first.mark_session_injected("s1")
    first.mark_query_completed("q1", {"query_id": "q1", "score": 1.0})

    resumed = _manager(tmp_path)
    checkpoint = resumed.load()
    assert checkpoint is not None
    resumed.start_persona(1)

    assert resumed.is_query_completed("q1")
    assert resumed.get_current_persona_id() == 1
    assert resumed.get_completed_results()[1][0]["query_id"] == "q1"


def test_starting_next_persona_clears_current_progress(tmp_path):
    manager = _manager(tmp_path)
    manager.create(total_personas=2, total_queries=3, evaluation_mode="independent")
    manager.start_persona(1)
    manager.mark_query_completed("q1", {"query_id": "q1"})
    manager.start_persona(2)

    assert not manager.is_query_completed("q1")
    assert manager.get_current_persona_id() == 2
