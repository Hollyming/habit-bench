"""MedMemoryBench checkpoint module for resumable evaluation (independent mode only)."""

import json
import hashlib
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional


@dataclass
class MedMemoryBenchCheckpoint:
    """Checkpoint data structure for MedMemoryBench (independent mode only)."""

    checkpoint_id: str
    created_at: str
    updated_at: str

    method_name: str
    model_name: str
    dataset_name: str = "medmemorybench"
    config_hash: str = ""

    status: str = "in_progress"
    evaluation_mode: str = "independent"

    completed_personas: List[int] = field(default_factory=list)
    current_persona_id: Optional[int] = None
    current_persona_completed_queries: List[str] = field(default_factory=list)
    current_persona_injected_sessions: List[str] = field(default_factory=list)

    completed_results: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)

    total_personas: int = 0
    total_queries: int = 0
    completed_query_count: int = 0

    last_error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MedMemoryBenchCheckpoint":
        defaults = {
            "checkpoint_id": "",
            "created_at": "",
            "updated_at": "",
            "method_name": "",
            "model_name": "",
            "dataset_name": "medmemorybench",
            "config_hash": "",
            "status": "in_progress",
            "evaluation_mode": "independent",
            "completed_personas": [],
            "current_persona_id": None,
            "current_persona_completed_queries": [],
            "current_persona_injected_sessions": [],
            "completed_results": {},
            "total_personas": 0,
            "total_queries": 0,
            "completed_query_count": 0,
            "last_error": None,
        }
        merged = {**defaults, **data}
        return cls(**merged)


class MedMemoryBenchCheckpointManager:
    """Checkpoint manager for MedMemoryBench evaluation."""

    def __init__(
        self,
        method_name: str,
        model_name: str,
        checkpoint_dir: Path,
        config_hash: str = "",
    ):
        self.method_name = method_name
        self.model_name = model_name
        self.checkpoint_dir = Path(checkpoint_dir)
        self.config_hash = config_hash
        self._checkpoint: Optional[MedMemoryBenchCheckpoint] = None

    @property
    def checkpoint_path(self) -> Path:
        safe_method = self.method_name.replace("/", "-").replace("\\", "-")
        safe_model = self.model_name.replace("/", "-").replace("\\", "-")
        subdir = f"{safe_method}_{safe_model}"
        return self.checkpoint_dir / "medmemorybench" / subdir / "checkpoint.json"

    def exists(self) -> bool:
        return self.checkpoint_path.exists()

    def load(self) -> Optional[MedMemoryBenchCheckpoint]:
        if not self.exists():
            return None

        try:
            with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self._checkpoint = MedMemoryBenchCheckpoint.from_dict(data)
            return self._checkpoint
        except (json.JSONDecodeError, KeyError):
            return None

    def save(self) -> None:
        if self._checkpoint is None:
            return

        self._checkpoint.updated_at = datetime.now().isoformat()
        self.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(self._checkpoint.to_dict(), f, ensure_ascii=False, indent=2)

    def create(
        self,
        total_personas: int,
        total_queries: int,
        evaluation_mode: str,
    ) -> MedMemoryBenchCheckpoint:
        import uuid

        self._checkpoint = MedMemoryBenchCheckpoint(
            checkpoint_id=str(uuid.uuid4()),
            created_at=datetime.now().isoformat(),
            updated_at=datetime.now().isoformat(),
            method_name=self.method_name,
            model_name=self.model_name,
            config_hash=self.config_hash,
            evaluation_mode=evaluation_mode,
            total_personas=total_personas,
            total_queries=total_queries,
        )
        self.save()
        return self._checkpoint

    def validate_config(self) -> bool:
        if self._checkpoint is None:
            return False
        if not self._checkpoint.config_hash:
            return True
        return self._checkpoint.config_hash == self.config_hash

    def is_independent_mode(self) -> bool:
        if self._checkpoint is None:
            return False
        return self._checkpoint.evaluation_mode == "independent"

    def start_persona(self, persona_id: int) -> None:
        if self._checkpoint is None:
            return

        # On a mid-persona resume, the evaluator rebuilds memory from the
        # beginning and must retain the already-scored query IDs/results.  A
        # second start for the same persona is therefore idempotent.
        if self._checkpoint.current_persona_id == persona_id:
            self.save()
            return

        self._checkpoint.current_persona_id = persona_id
        self._checkpoint.current_persona_completed_queries = []
        self._checkpoint.current_persona_injected_sessions = []
        self.save()

    def mark_session_injected(self, session_id: str) -> None:
        if self._checkpoint is None:
            return

        if session_id not in self._checkpoint.current_persona_injected_sessions:
            self._checkpoint.current_persona_injected_sessions.append(session_id)
            self.save()

    def mark_query_completed(self, query_id: str, result_dict: Dict[str, Any]) -> None:
        if self._checkpoint is None:
            return

        persona_id = self._checkpoint.current_persona_id
        if persona_id is None:
            return

        persona_key = str(persona_id)

        if query_id not in self._checkpoint.current_persona_completed_queries:
            self._checkpoint.current_persona_completed_queries.append(query_id)
            self._checkpoint.completed_query_count += 1

        if persona_key not in self._checkpoint.completed_results:
            self._checkpoint.completed_results[persona_key] = []

        existing_ids = [r.get("query_id") for r in self._checkpoint.completed_results[persona_key]]
        if query_id not in existing_ids:
            self._checkpoint.completed_results[persona_key].append(result_dict)

        self.save()

    def complete_persona(self, persona_id: int) -> None:
        if self._checkpoint is None:
            return

        if persona_id not in self._checkpoint.completed_personas:
            self._checkpoint.completed_personas.append(persona_id)

        self._checkpoint.current_persona_id = None
        self._checkpoint.current_persona_completed_queries = []
        self._checkpoint.current_persona_injected_sessions = []
        self.save()

    def mark_completed(self) -> None:
        if self._checkpoint is None:
            return

        self._checkpoint.status = "completed"
        self.save()

    def mark_failed(self, error: str) -> None:
        if self._checkpoint is None:
            return

        self._checkpoint.status = "failed"
        self._checkpoint.last_error = error
        self.save()

    def is_persona_completed(self, persona_id: int) -> bool:
        if self._checkpoint is None:
            return False
        return persona_id in self._checkpoint.completed_personas

    def is_query_completed(self, query_id: str) -> bool:
        if self._checkpoint is None:
            return False
        return query_id in self._checkpoint.current_persona_completed_queries

    def get_completed_results(self) -> Dict[int, List[Dict[str, Any]]]:
        if self._checkpoint is None:
            return {}
        return {
            int(k): v
            for k, v in self._checkpoint.completed_results.items()
        }

    def get_current_persona_id(self) -> Optional[int]:
        if self._checkpoint is None:
            return None
        return self._checkpoint.current_persona_id

    def get_resume_info(self) -> Dict[str, Any]:
        if self._checkpoint is None:
            return {}

        return {
            "checkpoint_id": self._checkpoint.checkpoint_id,
            "completed_personas": len(self._checkpoint.completed_personas),
            "total_personas": self._checkpoint.total_personas,
            "completed_queries": self._checkpoint.completed_query_count,
            "total_queries": self._checkpoint.total_queries,
            "current_persona": self._checkpoint.current_persona_id,
            "current_persona_completed_queries": len(self._checkpoint.current_persona_completed_queries),
        }

    def delete(self) -> None:
        if self.checkpoint_path.exists():
            self.checkpoint_path.unlink()
        self._checkpoint = None


def compute_config_hash(method_config, dataset_config) -> str:
    """Compute config hash for checkpoint validation."""
    try:
        content = json.dumps({
            "method": method_config.raw_config if hasattr(method_config, "raw_config") else {},
            "dataset": dataset_config.raw_config if hasattr(dataset_config, "raw_config") else {},
        }, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()[:16]
    except Exception:
        return ""
