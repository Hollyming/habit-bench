"""LoCoMo dataset processing module."""

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, Any, List, Iterator, Optional

from benchmarks.base import BaseDataset, Session, Query, EvaluationUnit


LOCOMO_CATEGORY_MAPPING = {
    1: "multi_hop",
    2: "temporal",
    3: "open_domain",
    4: "single_hop",
    5: "adversarial",
}

LOCOMO_CATEGORY_REVERSE = {v: k for k, v in LOCOMO_CATEGORY_MAPPING.items()}


@dataclass
class LoCoMoSession(Session):
    date_time: str = ""
    speaker_a: str = ""
    speaker_b: str = ""
    dialogues: List[Dict[str, Any]] = field(default_factory=list)

    def to_memory_text(self) -> str:
        lines = []
        lines.append(f"DATE: {self.date_time}")
        lines.append("CONVERSATION:")

        for turn in self.dialogues:
            speaker = turn.get("speaker", "Unknown")
            text = turn.get("text", "")
            turn_text = f'{speaker} said, "{text}"'

            if "blip_caption" in turn:
                turn_text += f' and shared {turn["blip_caption"]}'

            lines.append(turn_text)

        return "\n".join(lines)


@dataclass
class LoCoMoQuery(Query):
    category: int = 1
    evidence: List[str] = field(default_factory=list)
    adversarial_answer: Optional[str] = None

    def get_correct_answers(self) -> List[str]:
        return self.expected_answers


class LoCoMoDataset(BaseDataset):
    NAME = "locomo"

    def __init__(self, data_dir: Path, config: Dict[str, Any]):
        super().__init__(data_dir, config)

        self.data_file = config.get("data_file", "locomo10.json")
        self.sample_ids = config.get("sample_ids")
        self.max_samples = config.get("max_samples")
        self.category_filter = config.get("category_filter")
        self.include_images = config.get("include_images", True)

        self._samples: Dict[str, Dict[str, Any]] = {}

    def load(self) -> None:
        if self._is_loaded:
            return

        data_path = self.data_dir / self.data_file
        with open(data_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        for idx, sample in enumerate(raw_data):
            sample_id = sample.get("sample_id", f"sample_{idx}")

            if self.sample_ids and sample_id not in self.sample_ids:
                continue

            if self.max_samples and len(self._samples) >= self.max_samples:
                break

            sessions, queries = self._parse_sample(sample, sample_id)

            conversation = sample.get("conversation", {})
            self._samples[sample_id] = {
                "sample_id": sample_id,
                "sessions": sessions,
                "queries": queries,
                "speaker_a": conversation.get("speaker_a", "Speaker A"),
                "speaker_b": conversation.get("speaker_b", "Speaker B"),
            }

        self._is_loaded = True

    def _parse_sample(
        self, sample: Dict[str, Any], sample_id: str
    ) -> tuple[List[LoCoMoSession], List[LoCoMoQuery]]:
        conversation = sample.get("conversation", {})
        qa_list = sample.get("qa", [])

        sessions = self._parse_sessions(conversation, sample_id)
        queries = self._parse_queries(qa_list, sample_id)

        return sessions, queries

    def _parse_sessions(
        self, conversation: Dict[str, Any], sample_id: str
    ) -> List[LoCoMoSession]:
        sessions = []
        speaker_a = conversation.get("speaker_a", "Speaker A")
        speaker_b = conversation.get("speaker_b", "Speaker B")

        session_keys = sorted([
            k for k in conversation.keys()
            if k.startswith("session_") and not k.endswith("_date_time")
        ], key=lambda x: int(x.split("_")[1]))

        for session_key in session_keys:
            session_num = int(session_key.split("_")[1])
            date_time_key = f"session_{session_num}_date_time"
            date_time = conversation.get(date_time_key, "")

            dialogues = conversation.get(session_key, [])

            formatted_dialogues = []
            for turn in dialogues:
                turn_data = {
                    "speaker": turn.get("speaker", "Unknown"),
                    "text": turn.get("text", ""),
                    "dia_id": turn.get("dia_id", ""),
                }
                if self.include_images and "blip_caption" in turn:
                    turn_data["blip_caption"] = turn["blip_caption"]
                formatted_dialogues.append(turn_data)

            content_lines = []
            for turn in formatted_dialogues:
                line = f'{turn["speaker"]}: {turn["text"]}'
                content_lines.append(line)

            session = LoCoMoSession(
                session_id=session_num,
                content="\n".join(content_lines),
                metadata={
                    "sample_id": sample_id,
                    "session_key": session_key,
                    "turn_count": len(dialogues),
                },
                date_time=date_time,
                speaker_a=speaker_a,
                speaker_b=speaker_b,
                dialogues=formatted_dialogues,
            )
            sessions.append(session)

        return sessions

    def _parse_queries(
        self, qa_list: List[Dict[str, Any]], sample_id: str
    ) -> List[LoCoMoQuery]:
        queries = []

        for idx, qa in enumerate(qa_list):
            category = qa.get("category", 1)

            if self.category_filter and category not in self.category_filter:
                continue

            query_type = LOCOMO_CATEGORY_MAPPING.get(category, "unknown")

            answer = qa.get("answer", "")
            if category == 5 and "adversarial_answer" in qa:
                expected_answers = ["Not mentioned", "No information available"]
            else:
                if isinstance(answer, (int, float)):
                    expected_answers = [str(answer)]
                else:
                    expected_answers = [str(answer)]

            evidence = qa.get("evidence", [])
            if isinstance(evidence, str):
                evidence = [evidence]

            query = LoCoMoQuery(
                query_id=f"{sample_id}_q{idx}",
                question=qa.get("question", ""),
                query_type=query_type,
                expected_answers=expected_answers,
                metadata={
                    "sample_id": sample_id,
                    "original_index": idx,
                    "original_answer": answer,
                },
                category=category,
                evidence=evidence,
                adversarial_answer=qa.get("adversarial_answer"),
            )
            queries.append(query)

        return queries

    def get_evaluation_units(self) -> Iterator[EvaluationUnit]:
        unit_id = 0
        for sample_id, sample_data in self._samples.items():
            yield EvaluationUnit(
                unit_id=unit_id,
                sessions_to_inject=sample_data["sessions"],
                queries_to_evaluate=sample_data["queries"],
                context_id=sample_id,
                metadata={
                    "sample_id": sample_id,
                    "total_sessions": len(sample_data["sessions"]),
                    "total_queries": len(sample_data["queries"]),
                    "speaker_a": sample_data["speaker_a"],
                    "speaker_b": sample_data["speaker_b"],
                }
            )
            unit_id += 1

    def get_total_sessions(self) -> int:
        return sum(len(s["sessions"]) for s in self._samples.values())

    def get_total_queries(self) -> int:
        return sum(len(s["queries"]) for s in self._samples.values())

    def get_sample_ids(self) -> List[str]:
        return list(self._samples.keys())

    def get_sample_info(self, sample_id: str) -> Dict[str, Any]:
        if sample_id in self._samples:
            sample = self._samples[sample_id]
            return {
                "sample_id": sample_id,
                "total_sessions": len(sample["sessions"]),
                "total_queries": len(sample["queries"]),
                "speaker_a": sample["speaker_a"],
                "speaker_b": sample["speaker_b"],
            }
        return {}

    def get_category_distribution(self) -> Dict[str, int]:
        distribution: Dict[str, int] = {}
        for sample_data in self._samples.values():
            for query in sample_data["queries"]:
                query_type = query.query_type
                distribution[query_type] = distribution.get(query_type, 0) + 1
        return distribution
