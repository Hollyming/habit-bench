"""LLM-as-Judge metrics for MedMemoryBench."""

import re
import json
import logging
from typing import List, Dict, Any, Optional

from .base import BaseMetric, MetricResult
from utils.templates import get_prompt_manager

logger = logging.getLogger(__name__)

EMPTY_OUTPUT_REASON = "Model provided no response"


class LLMJudge:
    """LLM judge using 0/1 binary scoring."""

    def __init__(self, dataset: str = "medmemorybench", client=None,
                 judge_model: str = None, judge_api_key: str = None, judge_base_url: str = None,
                 language: str = "zh"):
        self._client = client
        self._initialized = False
        self._prompt_manager = get_prompt_manager(dataset, language=language)
        self._judge_model = judge_model
        self._judge_api_key = judge_api_key
        self._judge_base_url = judge_base_url

    def _ensure_client(self):
        if not self._initialized:
            if self._client is None:
                from utils.llm_client import create_llm_client
                from src.config import get_api_config

                api_config = get_api_config()

                model = self._judge_model or api_config.get_judge_model()
                api_key = self._judge_api_key or api_config.get_judge_api_key()
                base_url = self._judge_base_url or api_config.get_judge_base_url()

                self._client = create_llm_client(
                    provider="openai",
                    model=model,
                    temperature=1.0,
                    max_tokens=10000,
                    api_key=api_key,
                    base_url=base_url,
                )
            self._initialized = True

    def _extract_json_from_text(self, text: str) -> Optional[Dict[str, Any]]:
        """Extract JSON object from text, handling nested structures correctly.

        Uses bracket matching to find the complete outermost JSON object,
        which is necessary for complex nested JSON like MCD evaluation results.
        """
        # Find the first '{' character
        start_idx = text.find('{')
        if start_idx == -1:
            return None

        # Use bracket counting to find the matching '}'
        bracket_count = 0
        in_string = False
        escape_next = False

        for i, char in enumerate(text[start_idx:], start=start_idx):
            if escape_next:
                escape_next = False
                continue

            if char == '\\' and in_string:
                escape_next = True
                continue

            if char == '"' and not escape_next:
                in_string = not in_string
                continue

            if not in_string:
                if char == '{':
                    bracket_count += 1
                elif char == '}':
                    bracket_count -= 1
                    if bracket_count == 0:
                        # Found the complete JSON object
                        json_str = text[start_idx:i + 1]
                        try:
                            return json.loads(json_str)
                        except json.JSONDecodeError as e:
                            logger.warning(f"[LLMJudge] JSON parse error: {e}")
                            return None

        return None

    def _call_llm(self, prompt: str, max_tokens: int = 500) -> Optional[Dict[str, Any]]:
        self._ensure_client()

        try:
            response = self._client.chat(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
            )
            result_text = response.content.strip()

            # First try direct JSON parsing
            try:
                return json.loads(result_text)
            except json.JSONDecodeError:
                pass

            # Use bracket-matching extraction for nested JSON structures
            result = self._extract_json_from_text(result_text)
            if result is not None:
                return result

            logger.warning(f"[LLMJudge] Failed to extract JSON from response: {result_text[:200]}...")
        except Exception as e:
            print(f"[LLMJudge] API call failed: {e}")
        return None

    def _is_empty_output(self, model_output: str) -> bool:
        return not model_output or not model_output.strip()

    def judge_temporal_localization(
        self,
        question: str,
        model_output: str,
        expected_answer: str,
        explanation: str = "",
    ) -> Dict[str, Any]:
        if self._is_empty_output(model_output):
            logger.warning(f"[LLMJudge] Empty model output for temporal_localization")
            return {"is_correct": False, "score": 0.0, "reason": EMPTY_OUTPUT_REASON}

        prompt = self._prompt_manager.format_judge(
            query_type="temporal_localization",
            question=question,
            model_output=model_output,
            expected_answer=expected_answer,
            explanation=explanation,
        )

        result = self._call_llm(prompt)
        if result:
            is_correct = result.get("is_correct", False)
            return {
                "is_correct": is_correct,
                "score": 1.0 if is_correct else 0.0,
                "reason": result.get("reason", ""),
            }
        return {"is_correct": False, "score": 0.0, "reason": "Judge failed"}

    def judge_state_update(
        self,
        question: str,
        model_output: str,
        expected_answer: str,
        explanation: str = "",
    ) -> Dict[str, Any]:
        if self._is_empty_output(model_output):
            logger.warning(f"[LLMJudge] Empty model output for state_update")
            return {"is_correct": False, "score": 0.0, "reason": EMPTY_OUTPUT_REASON}

        prompt = self._prompt_manager.format_judge(
            query_type="state_update",
            question=question,
            model_output=model_output,
            expected_answer=expected_answer,
            explanation=explanation,
        )

        result = self._call_llm(prompt)
        if result:
            is_correct = result.get("is_correct", False)
            return {
                "is_correct": is_correct,
                "score": 1.0 if is_correct else 0.0,
                "reason": result.get("reason", ""),
            }
        return {"is_correct": False, "score": 0.0, "reason": "Judge failed"}

    def judge_inference_generation(
        self,
        question: str,
        model_output: str,
        expected_answer: str,
        explanation: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if self._is_empty_output(model_output):
            logger.warning(f"[LLMJudge] Empty model output for inference_generation")
            return {"is_correct": False, "score": 0.0, "reason": EMPTY_OUTPUT_REASON}

        metadata_info = ""
        if metadata:
            if "inference_type" in metadata:
                metadata_info += f"\nInference type: {metadata['inference_type']}"
            if "trap_design" in metadata:
                trap = metadata["trap_design"]
                if "trap_mechanism" in trap:
                    metadata_info += f"\nTrap mechanism: {trap['trap_mechanism']}"
                if "required_patient_info" in trap:
                    metadata_info += f"\nRequired patient info: {', '.join(trap['required_patient_info'])}"
            if "common_wrong_answer" in metadata:
                wrong = metadata["common_wrong_answer"]
                metadata_info += f"\nCommon wrong answer: {wrong.get('content', '')}"
                metadata_info += f"\nError reason: {wrong.get('why_wrong', '')}"

        prompt = self._prompt_manager.format_judge(
            query_type="inference_generation",
            question=question,
            model_output=model_output,
            expected_answer=expected_answer,
            explanation=explanation,
            metadata_info=metadata_info,
        )

        result = self._call_llm(prompt)
        if result:
            is_correct = result.get("is_correct", False)
            return {
                "is_correct": is_correct,
                "score": 1.0 if is_correct else 0.0,
                "reason": result.get("reason", ""),
            }
        return {"is_correct": False, "score": 0.0, "reason": "Judge failed"}

    def judge_multi_hop_clinical_deduction(
        self,
        question: str,
        model_output: str,
        expected_answer: str,
        explanation: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if self._is_empty_output(model_output):
            logger.warning(f"[LLMJudge] Empty model output for multi_hop_clinical_deduction")
            return {
                "is_correct": False,
                "score": 0.0,
                "ncr_score": 0.0,
                "crc_score": 0.0,
                "cc_score": 0.0,
                "node_validations": [],
                "reason": EMPTY_OUTPUT_REASON,
            }

        reasoning_chain = []
        required_memory_nodes = []
        hop_count = 0
        reasoning_pattern = ""

        if metadata:
            reasoning_chain = metadata.get("reasoning_chain", [])
            required_memory_nodes = metadata.get("required_memory_nodes", [])
            hop_count = metadata.get("hop_count", 0)
            reasoning_pattern = metadata.get("reasoning_pattern", "")

        nodes_for_validation = ""
        if reasoning_chain:
            nodes_for_validation = "\n[Reasoning chain nodes (to be verified one by one)]\n"
            for i, node in enumerate(reasoning_chain):
                node_id = node.get("node_id", i + 1)
                session_id = node.get("session_id", "?")
                content = node.get("content", "")
                role = node.get("role", "")
                source_info = node.get("source_info", "")

                nodes_for_validation += f"""
Node {node_id}:
  - Source: Session {session_id} ({source_info})
  - Role: {role}
  - Content: {content}
"""

        required_nodes_str = ""
        if required_memory_nodes:
            required_nodes_str = "\n[Information that must be recalled from memory]\n"
            for node in required_memory_nodes:
                required_nodes_str += f"- {node}\n"

        prompt = self._prompt_manager.format_judge(
            query_type="multi_hop_clinical_deduction",
            question=question,
            model_output=model_output,
            expected_answer=expected_answer,
            explanation=explanation,
            nodes_for_validation=nodes_for_validation,
            required_nodes_str=required_nodes_str,
            hop_count=hop_count,
            reasoning_pattern=reasoning_pattern,
        )

        result = self._call_llm(prompt, max_tokens=2000)
        if result:
            is_correct = result.get("is_correct", False)
            ncr_score = result.get("ncr_score", 0.0)
            crc_score = result.get("crc_score", 0.0)
            cc_score = result.get("cc_score", 0.0)
            node_validations = result.get("node_validations", [])
            uses_patient_specific_info = result.get("uses_patient_specific_info", False)
            memory_retrieval_quality = result.get("memory_retrieval_quality", "none")

            # Strict composite score calculation
            # Weight: NCR 35%, CRC 35%, CC 30%
            composite_score = ncr_score * 0.35 + crc_score * 0.35 + cc_score * 0.30

            # Apply penalty if model doesn't use patient-specific information
            if not uses_patient_specific_info:
                composite_score *= 0.5  # 50% penalty

            # Apply penalty based on memory retrieval quality
            quality_multiplier = {
                "excellent": 1.0,
                "good": 0.9,
                "partial": 0.7,
                "poor": 0.4,
                "none": 0.1
            }
            composite_score *= quality_multiplier.get(memory_retrieval_quality, 0.5)

            return {
                "is_correct": is_correct,
                "score": composite_score,
                "ncr_score": ncr_score,
                "crc_score": crc_score,
                "cc_score": cc_score,
                "node_validations": node_validations,
                "uses_patient_specific_info": uses_patient_specific_info,
                "memory_retrieval_quality": memory_retrieval_quality,
                "reason": result.get("reason", ""),
            }
        return {
            "is_correct": False,
            "score": 0.0,
            "ncr_score": 0.0,
            "crc_score": 0.0,
            "cc_score": 0.0,
            "node_validations": [],
            "reason": "Judge failed",
        }

    def judge_locomo_open_domain(
        self,
        question: str,
        model_output: str,
        expected_answer: str,
    ) -> Dict[str, Any]:
        if self._is_empty_output(model_output):
            logger.warning(f"[LLMJudge] Empty model output for locomo_open_domain")
            return {"is_correct": False, "score": 0.0, "reason": EMPTY_OUTPUT_REASON}

        prompt = self._prompt_manager.format_judge(
            query_type="locomo_open_domain",
            question=question,
            model_output=model_output,
            expected_answer=expected_answer,
        )

        result = self._call_llm(prompt)
        if result:
            is_correct = result.get("is_correct", False)
            return {
                "is_correct": is_correct,
                "score": 1.0 if is_correct else 0.0,
                "reason": result.get("reason", ""),
            }
        return {"is_correct": False, "score": 0.0, "reason": "Judge failed"}


# LLM Judge Metric Classes

class LLMJudgeMetric(BaseMetric):
    """LLM-as-Judge metric for TLA, SUA, IG query types."""

    NAME = "llm_judge"

    def __init__(self, dataset: str = "medmemorybench",
                 judge_model: str = None, judge_api_key: str = None, judge_base_url: str = None,
                 language: str = "zh"):
        self._dataset = dataset
        self._judge_model = judge_model
        self._judge_api_key = judge_api_key
        self._judge_base_url = judge_base_url
        self._language = language
        self._judge: Optional[LLMJudge] = None

    @property
    def judge(self) -> LLMJudge:
        if self._judge is None:
            self._judge = LLMJudge(
                dataset=self._dataset,
                judge_model=self._judge_model,
                judge_api_key=self._judge_api_key,
                judge_base_url=self._judge_base_url,
                language=self._language,
            )
        return self._judge

    def compute(
        self,
        query_id: str,
        query_type: str,
        model_output: str,
        expected_answers: List[str],
        question: str = "",
        answers_data: List[dict] = None,
        metadata: Dict[str, Any] = None,
        **kwargs
    ) -> MetricResult:
        expected_answer = expected_answers[0] if expected_answers else ""

        explanation = ""
        if answers_data:
            for ans in answers_data:
                if ans.get("is_correct", False):
                    explanation = ans.get("explanation", "")
                    break

        if query_type == "temporal_localization":
            result = self.judge.judge_temporal_localization(
                question, model_output, expected_answer, explanation
            )
        elif query_type == "state_update":
            result = self.judge.judge_state_update(
                question, model_output, expected_answer, explanation
            )
        elif query_type == "inference_generation":
            result = self.judge.judge_inference_generation(
                question, model_output, expected_answer, explanation, metadata
            )
        elif query_type == "open_domain":
            result = self.judge.judge_locomo_open_domain(
                question, model_output, expected_answer
            )
        else:
            result = self.judge.judge_state_update(
                question, model_output, expected_answer, explanation
            )

        return MetricResult(
            query_id=query_id,
            query_type=query_type,
            score=result["score"],
            is_correct=result["is_correct"],
            model_output=model_output,
            expected_answer=expected_answer,
            question=question,
            details={
                "judge_reason": result.get("reason", ""),
                "explanation": explanation,
                "metric": self.NAME,
            }
        )


class LLMJudgeMCDMetric(BaseMetric):
    """LLM Judge metric for multi-hop clinical deduction (MCD) with NCR/CRC/CC scoring."""

    NAME = "llm_judge_mcd"

    def __init__(self, dataset: str = "medmemorybench",
                 judge_model: str = None, judge_api_key: str = None, judge_base_url: str = None,
                 language: str = "zh"):
        self._dataset = dataset
        self._judge_model = judge_model
        self._judge_api_key = judge_api_key
        self._judge_base_url = judge_base_url
        self._language = language
        self._judge: Optional[LLMJudge] = None

    @property
    def judge(self) -> LLMJudge:
        if self._judge is None:
            self._judge = LLMJudge(
                dataset=self._dataset,
                judge_model=self._judge_model,
                judge_api_key=self._judge_api_key,
                judge_base_url=self._judge_base_url,
                language=self._language,
            )
        return self._judge

    def compute(
        self,
        query_id: str,
        query_type: str,
        model_output: str,
        expected_answers: List[str],
        question: str = "",
        answers_data: List[dict] = None,
        metadata: Dict[str, Any] = None,
        **kwargs
    ) -> MetricResult:
        expected_answer = expected_answers[0] if expected_answers else ""

        explanation = ""
        if answers_data:
            for ans in answers_data:
                if ans.get("is_correct", False):
                    explanation = ans.get("explanation", "")
                    break

        result = self.judge.judge_multi_hop_clinical_deduction(
            question, model_output, expected_answer, explanation, metadata
        )

        return MetricResult(
            query_id=query_id,
            query_type=query_type,
            score=result["score"],
            is_correct=result["is_correct"],
            model_output=model_output,
            expected_answer=expected_answer,
            question=question,
            details={
                "judge_reason": result.get("reason", ""),
                "ncr_score": result.get("ncr_score", 0.0),
                "crc_score": result.get("crc_score", 0.0),
                "cc_score": result.get("cc_score", 0.0),
                "node_validations": result.get("node_validations", []),
                "uses_patient_specific_info": result.get("uses_patient_specific_info", False),
                "memory_retrieval_quality": result.get("memory_retrieval_quality", "none"),
                "explanation": explanation,
                "metric": self.NAME,
            }
        )
