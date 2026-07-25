"""String matching metrics."""

import re
import string
from typing import List, Set

from .base import BaseMetric, MetricResult


def normalize_text(text: str) -> str:
    """Normalize text: remove punctuation, whitespace, convert to lowercase."""
    punctuation = string.punctuation + '，。！？、；：""''（）【】《》·…—～－–·'
    text = text.translate(str.maketrans('', '', punctuation))
    text = ''.join(text.split())
    return text.strip().lower()


def extract_option_letters(text: str) -> Set[str]:
    """Extract option letters from model output."""
    text_upper = text.upper()
    options = set()

    patterns = [
        r'\b([A-F])\b',
        r'选([A-F])',
        r'答案[是为：:]*\s*([A-F])',
        r'choose\s*([A-F])',
        r'answer[:\s]*([A-F])',
        r'([A-F])选项',
    ]

    for pattern in patterns:
        options.update(re.findall(pattern, text_upper))

    return options


class StringContainMetric(BaseMetric):
    """String containment metric - checks if normalized output contains expected answer."""

    NAME = "string_contain"

    def compute(
        self,
        query_id: str,
        query_type: str,
        model_output: str,
        expected_answers: List[str],
        question: str = "",
        **kwargs
    ) -> MetricResult:
        model_normalized = normalize_text(model_output)

        matched = []
        for answer in expected_answers:
            answer_normalized = normalize_text(answer)
            if answer_normalized and answer_normalized in model_normalized:
                matched.append(answer)

        is_correct = len(matched) == len(expected_answers) and len(expected_answers) > 0

        return MetricResult(
            query_id=query_id,
            query_type=query_type,
            score=1.0 if is_correct else 0.0,
            is_correct=is_correct,
            model_output=model_output,
            expected_answer=", ".join(expected_answers),
            question=question,
            details={
                "matched_answers": matched,
                "total_expected": len(expected_answers),
                "total_matched": len(matched),
                "metric": self.NAME,
            }
        )


class ExactMatchMetric(BaseMetric):
    """Exact match metric - checks if normalized output exactly matches expected answer."""

    NAME = "exact_match"

    def compute(
        self,
        query_id: str,
        query_type: str,
        model_output: str,
        expected_answers: List[str],
        question: str = "",
        **kwargs
    ) -> MetricResult:
        model_normalized = normalize_text(model_output)

        is_correct = False
        matched_answer = ""
        for answer in expected_answers:
            answer_normalized = normalize_text(answer)
            if model_normalized == answer_normalized:
                is_correct = True
                matched_answer = answer
                break

        return MetricResult(
            query_id=query_id,
            query_type=query_type,
            score=1.0 if is_correct else 0.0,
            is_correct=is_correct,
            model_output=model_output,
            expected_answer=expected_answers[0] if expected_answers else "",
            question=question,
            details={
                "matched_answer": matched_answer,
                "metric": self.NAME,
            }
        )


class OptionMatchMetric(BaseMetric):
    """Option match metric for multiple choice questions."""

    NAME = "option_match"

    def compute(
        self,
        query_id: str,
        query_type: str,
        model_output: str,
        expected_answers: List[str],
        question: str = "",
        answers_data: List[dict] = None,
        **kwargs
    ) -> MetricResult:
        selected_options = extract_option_letters(model_output)

        correct_options = set()

        if answers_data:
            for ans in answers_data:
                if ans.get("is_correct", False):
                    content = ans.get("content", "")
                    match = re.match(r'^([A-F])[.、:\s]', content.upper())
                    if match:
                        correct_options.add(match.group(1))

        if not correct_options:
            for answer in expected_answers:
                match = re.match(r'^([A-F])[.、:\s]', answer.upper())
                if match:
                    correct_options.add(match.group(1))
                else:
                    correct_options.update(extract_option_letters(answer))

        is_correct = selected_options == correct_options

        return MetricResult(
            query_id=query_id,
            query_type=query_type,
            score=1.0 if is_correct else 0.0,
            is_correct=is_correct,
            model_output=model_output,
            expected_answer=", ".join(sorted(correct_options)),
            question=question,
            details={
                "selected_options": sorted(list(selected_options)),
                "correct_options": sorted(list(correct_options)),
                "metric": self.NAME,
            }
        )
