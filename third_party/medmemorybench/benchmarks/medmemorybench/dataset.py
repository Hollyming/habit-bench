"""MedMemoryBench dataset processing module."""

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List, Iterator, Optional

from benchmarks.base import BaseDataset, Session, Query, EvaluationUnit


@dataclass
class MedSession(Session):
    """MedMemoryBench session with medical dialogue data."""
    timestamp: Optional[str] = None
    is_noise: bool = False
    noise_type: Optional[str] = None
    original_index: int = 0
    persona_id: Optional[int] = None
    knowledge_points: List[Dict[str, Any]] = field(default_factory=list)
    event_info: Dict[str, Any] = field(default_factory=dict)
    family_role: Optional[Dict[str, Any]] = None

    def to_memory_text(self) -> str:
        """Convert session to memory text format."""
        lines = []

        if self.is_noise:
            if self.noise_type == "family_health_consultation" and self.family_role:
                role_name = self.family_role.get('name', 'Family member')
                relationship = self.family_role.get('relationship', '')
                lines.append(f"[Health consultation record about {role_name}({relationship})]")
            else:
                lines.append("[Health consultation record]")
        else:
            date_str = self.timestamp or self.event_info.get('date', 'N/A')
            lines.append(f"[{date_str}]")

        lines.append("")

        messages = self.metadata.get("messages", [])
        if messages:
            for msg in messages:
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                if role == "user":
                    lines.append(f"Patient: {content}")
                elif role == "assistant":
                    lines.append(f"Doctor: {content}")
                lines.append("")
        elif self.content:
            lines.append(self.content)

        return "\n".join(lines)


@dataclass
class MedQuery(Query):
    """MedMemoryBench query with medical question data."""
    session_id: Optional[int] = None
    answers_data: List[Dict[str, Any]] = field(default_factory=list)
    source_key_points: List[Dict[str, Any]] = field(default_factory=list)

    def get_correct_answers(self) -> List[str]:
        if self.answers_data:
            return [
                ans.get("content", "")
                for ans in self.answers_data
                if ans.get("is_correct", False)
            ]
        return self.expected_answers


class MedMemoryBenchDataset(BaseDataset):
    """MedMemoryBench dataset with persona-based medical dialogues."""

    NAME = "medmemorybench"

    def __init__(
        self,
        data_dir: Path,
        config: Dict[str, Any],
    ):
        super().__init__(data_dir, config)

        self.evaluation_mode = config.get("evaluation_mode", "independent")
        self.persona_ids = config.get("persona_ids")
        self.max_personas = config.get("max_personas")
        self.max_sessions_per_persona = config.get("max_sessions_per_persona")
        self.evaluation_interval = config.get("evaluation_interval", 10)
        self.inject_noise = config.get("inject_noise", True)
        self.query_types_filter = config.get("query_types")

        self._personas: Dict[int, Dict[str, Any]] = {}

    def load(self) -> None:
        if self._is_loaded:
            return

        persona_dirs = sorted(
            [d for d in self.data_dir.iterdir() if d.is_dir() and d.name.startswith("persona_")],
            key=lambda x: int(x.name.split("_")[1])
        )

        if self.persona_ids:
            persona_dirs = [
                d for d in persona_dirs
                if int(d.name.split("_")[1]) in self.persona_ids
            ]

        if self.max_personas:
            persona_dirs = persona_dirs[:self.max_personas]

        for persona_dir in persona_dirs:
            persona_id = int(persona_dir.name.split("_")[1])
            self._load_persona(persona_id, persona_dir)

        self._is_loaded = True

    def _load_persona(self, persona_id: int, persona_dir: Path) -> None:
        eval_dir = persona_dir / "eval"

        if self.inject_noise:
            dialogues_file = eval_dir / "generated_dialogues_with_noise.json"
        else:
            dialogues_file = eval_dir / "generated_dialogues.json"

        queries_file = eval_dir / "generated_queries.json"

        sessions: List[MedSession] = []
        if dialogues_file.exists():
            with open(dialogues_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                sessions_data = data.get("sessions", [])

                for i, s in enumerate(sessions_data):
                    is_noise_type1 = "noise_id" in s
                    is_noise_type2 = "noise_family_id" in s
                    is_noise = is_noise_type1 or is_noise_type2

                    noise_type = None
                    if is_noise_type2:
                        noise_type = "family_health_consultation"
                    elif is_noise_type1:
                        noise_type = "health_knowledge"

                    messages = s.get("messages", [])
                    content_lines = []
                    for msg in messages:
                        role = msg.get("role", "")
                        text = msg.get("content", "")
                        if role == "user":
                            content_lines.append(f"Patient: {text}")
                        elif role == "assistant":
                            content_lines.append(f"Doctor: {text}")
                    content = "\n\n".join(content_lines)

                    event_info = s.get("event_info", {})
                    timestamp = event_info.get("date") if event_info else None

                    session = MedSession(
                        session_id=s.get("session_id", i),
                        content=content,
                        metadata={
                            "messages": messages,
                            "event_id": s.get("event_id"),
                            "turn_count": s.get("turn_count", 0),
                            "status": s.get("status", ""),
                        },
                        timestamp=timestamp,
                        is_noise=is_noise,
                        noise_type=noise_type,
                        original_index=i,
                        persona_id=s.get("persona_id", persona_id),
                        knowledge_points=s.get("knowledge_points", []),
                        event_info=event_info,
                        family_role=s.get("family_role"),
                    )
                    sessions.append(session)

        if self.max_sessions_per_persona:
            sessions = sessions[:self.max_sessions_per_persona]

        queries: List[MedQuery] = []
        queries_by_session: Dict[int, List[MedQuery]] = {}

        if queries_file.exists():
            with open(queries_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                queries_data = data.get("queries", [])

                for q in queries_data:
                    query_type = q.get("query_type", "unknown")

                    if self.query_types_filter and query_type not in self.query_types_filter:
                        continue

                    answers_data = q.get("answers", [])

                    expected_answers = [
                        ans.get("content", "")
                        for ans in answers_data
                        if ans.get("is_correct", False)
                    ]

                    query = MedQuery(
                        query_id=q.get("query_id", ""),
                        question=q.get("question", ""),
                        query_type=query_type,
                        expected_answers=expected_answers,
                        metadata=q.get("metadata", {}),
                        session_id=q.get("session_id"),
                        answers_data=answers_data,
                        source_key_points=q.get("source_key_points", []),
                    )
                    queries.append(query)

                    if query.session_id is not None:
                        if query.session_id not in queries_by_session:
                            queries_by_session[query.session_id] = []
                        queries_by_session[query.session_id].append(query)

        self._personas[persona_id] = {
            "persona_id": persona_id,
            "sessions": sessions,
            "queries": queries,
            "queries_by_session": queries_by_session,
        }

    def get_evaluation_units(self) -> Iterator[EvaluationUnit]:
        if self.evaluation_mode == "independent":
            yield from self._get_independent_units()
        else:
            yield from self._get_merged_units()

    def _get_independent_units(self) -> Iterator[EvaluationUnit]:
        """Independent mode: evaluate each persona separately."""
        unit_id = 0

        for persona_id, persona_data in self._personas.items():
            sessions = persona_data["sessions"]
            queries_by_session = persona_data["queries_by_session"]

            pending_sessions: List[Session] = []
            regular_session_count = 0

            for session in sessions:
                pending_sessions.append(session)

                if not session.is_noise:
                    regular_session_count += 1

                    if regular_session_count % self.evaluation_interval == 0:
                        queries = queries_by_session.get(session.session_id, [])

                        if queries:
                            yield EvaluationUnit(
                                unit_id=unit_id,
                                sessions_to_inject=pending_sessions.copy(),
                                queries_to_evaluate=queries,
                                context_id=persona_id,
                                metadata={
                                    "persona_id": persona_id,
                                    "eval_session_id": session.session_id,
                                    "total_sessions_injected": len(pending_sessions),
                                    "noise_sessions_count": sum(1 for s in pending_sessions if isinstance(s, MedSession) and s.is_noise),
                                }
                            )
                            unit_id += 1

                        pending_sessions = []

    def _get_merged_units(self) -> Iterator[EvaluationUnit]:
        """Merged mode: evaluate all personas continuously."""
        unit_id = 0
        pending_sessions: List[Session] = []
        total_session_count = 0

        for persona_id, persona_data in self._personas.items():
            sessions = persona_data["sessions"]
            queries_by_session = persona_data["queries_by_session"]

            for session in sessions:
                pending_sessions.append(session)

                if not session.is_noise:
                    total_session_count += 1

                    if total_session_count % self.evaluation_interval == 0:
                        queries = queries_by_session.get(session.session_id, [])

                        if queries:
                            yield EvaluationUnit(
                                unit_id=unit_id,
                                sessions_to_inject=pending_sessions.copy(),
                                queries_to_evaluate=queries,
                                context_id=0,
                                metadata={
                                    "persona_id": persona_id,
                                    "eval_session_id": session.session_id,
                                }
                            )
                            unit_id += 1

                        pending_sessions = []

    def get_total_sessions(self) -> int:
        return sum(len(p["sessions"]) for p in self._personas.values())

    def get_total_queries(self) -> int:
        return sum(len(p["queries"]) for p in self._personas.values())

    def get_persona_ids(self) -> List[int]:
        return list(self._personas.keys())

    def get_persona_info(self, persona_id: int) -> Dict[str, Any]:
        if persona_id in self._personas:
            return {
                "persona_id": persona_id,
                "total_sessions": len(self._personas[persona_id]["sessions"]),
                "total_queries": len(self._personas[persona_id]["queries"]),
            }
        return {}

    def get_query_type_distribution(self) -> Dict[str, int]:
        distribution: Dict[str, int] = {}
        for persona_data in self._personas.values():
            for query in persona_data["queries"]:
                query_type = query.query_type
                distribution[query_type] = distribution.get(query_type, 0) + 1
        return distribution
