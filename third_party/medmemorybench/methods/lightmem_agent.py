"""LightMem agent adapter for MedMemoryBench."""

from __future__ import annotations

import importlib
import os
import sys
import json
import uuid
import tempfile
import concurrent.futures
import threading
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List, Literal

from .base import BaseAgent, MemoryBuildResult, AgentResponse, MemoryRetrievalResult
from utils.llm_client import create_llm_client, format_messages, BaseLLMClient, get_usage_tracker


class TrackedMemoryManager:
    """
    A Memory Manager that wraps LightMem's LLM calls through our llm_client
    for token tracking. This replaces LightMem's OpenaiManager while maintaining
    the same interface.
    """

    def __init__(
        self,
        llm_client: BaseLLMClient,
        model: str,
        temperature: float = 0.1,
        max_tokens: int = 2000,
        top_p: float = 0.1,
    ):
        self.llm_client = llm_client
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

        # Import LightMem utilities
        lightmem_src = Path(__file__).resolve().parent / "LightMem" / "src"
        if str(lightmem_src) not in sys.path:
            sys.path.insert(0, str(lightmem_src))

        from lightmem.memory.prompts import EXTRACTION_PROMPTS, METADATA_GENERATE_PROMPT, UPDATE_PROMPT
        from lightmem.memory.utils import clean_response

        self.EXTRACTION_PROMPTS = EXTRACTION_PROMPTS
        self.METADATA_GENERATE_PROMPT = METADATA_GENERATE_PROMPT
        self.UPDATE_PROMPT = UPDATE_PROMPT
        self.clean_response = clean_response

        # For compatibility with LightMem's config access
        class ConfigProxy:
            def __init__(self, model, temperature, max_tokens, top_p):
                self.model = model
                self.temperature = temperature
                self.max_tokens = max_tokens
                self.top_p = top_p

        self.config = ConfigProxy(model, temperature, max_tokens, top_p)

    def generate_response(
        self,
        messages: List[Dict[str, str]],
        response_format: Optional[Dict[str, str]] = None,
        tools: Optional[List[Dict]] = None,
        tool_choice: str = "auto",
    ) -> tuple:
        """
        Generate a response using our tracked llm_client.
        Returns (parsed_response, usage_info) to match LightMem's interface.
        """
        # Build kwargs for special parameters
        kwargs = {}
        if response_format:
            kwargs["response_format"] = response_format

        # Call through our tracked client
        response = self.llm_client.chat(
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            **kwargs
        )

        usage_info = {
            "prompt_tokens": response.input_tokens,
            "completion_tokens": response.output_tokens,
            "total_tokens": response.input_tokens + response.output_tokens,
        }

        return response.content, usage_info

    def meta_text_extract(
        self,
        extract_list: List[List[List[Dict]]],
        messages_use: Literal["user_only", "assistant_only", "hybrid"] = "user_only",
        topic_id_mapping: Optional[List[List[int]]] = None,
        extraction_mode: Literal["flat", "event"] = "flat",
        custom_prompts: Optional[Dict[str, str]] = None
    ) -> List[Optional[Dict]]:
        """
        Extract metadata from text segments using parallel processing.
        This mirrors LightMem's OpenaiManager.meta_text_extract but uses our llm_client.
        """
        if not extract_list:
            return []

        default_prompts = self.EXTRACTION_PROMPTS.get(extraction_mode, {})

        if custom_prompts is None:
            prompts = default_prompts
        else:
            prompts = {**default_prompts, **custom_prompts}

        if extraction_mode == "flat":
            return self._extract_with_prompt(
                system_prompt=prompts.get("factual", self.METADATA_GENERATE_PROMPT),
                extract_list=extract_list,
                messages_use=messages_use,
                topic_id_mapping=topic_id_mapping,
                entry_type="factual"
            )

        elif extraction_mode == "event":
            factual_results = self._extract_with_prompt(
                system_prompt=prompts["factual"],
                extract_list=extract_list,
                messages_use=messages_use,
                topic_id_mapping=topic_id_mapping,
                entry_type="factual"
            )

            relational_results = self._extract_with_prompt(
                system_prompt=prompts["relational"],
                extract_list=extract_list,
                messages_use=messages_use,
                topic_id_mapping=topic_id_mapping,
                entry_type="relational"
            )

            return self._merge_dual_perspective_results(factual_results, relational_results)

        else:
            raise ValueError(f"Unknown extraction_mode: {extraction_mode}")

    def _merge_dual_perspective_results(
        self,
        factual_results: List[Optional[Dict]],
        relational_results: List[Optional[Dict]]
    ) -> List[Optional[Dict]]:
        """Merge factual and relational extraction results."""
        merged_results = []

        for factual, relational in zip(factual_results, relational_results):
            if factual is None and relational is None:
                merged_results.append(None)
                continue

            merged = {
                "input_prompt": [],
                "output_prompt": "",
                "cleaned_result": [],
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0
                }
            }

            if factual is not None:
                merged["input_prompt"].extend(factual.get("input_prompt", []))
                merged["cleaned_result"].extend(factual.get("cleaned_result", []))
                if factual.get("usage"):
                    for key in merged["usage"]:
                        merged["usage"][key] += factual["usage"].get(key, 0)

            if relational is not None:
                merged["input_prompt"].extend(relational.get("input_prompt", []))
                merged["cleaned_result"].extend(relational.get("cleaned_result", []))
                if relational.get("usage"):
                    for key in merged["usage"]:
                        merged["usage"][key] += relational["usage"].get(key, 0)

            merged["output_prompt"] = (
                f"Factual: {factual.get('output_prompt', 'N/A') if factual else 'N/A'}\n"
                f"Relational: {relational.get('output_prompt', 'N/A') if relational else 'N/A'}"
            )

            merged_results.append(merged)

        return merged_results

    def _extract_with_prompt(
        self,
        system_prompt: str,
        extract_list: List[List[List[Dict]]],
        messages_use: str,
        topic_id_mapping: Optional[List[List[int]]],
        entry_type: str = "factual"
    ) -> List[Optional[Dict]]:
        """Extract information with a specific prompt using parallel processing."""

        def concatenate_messages(segment: List[Dict], messages_use: str) -> str:
            """Concatenate messages based on usage strategy."""
            role_filter = {
                "user_only": {"user"},
                "assistant_only": {"assistant"},
                "hybrid": {"user", "assistant"}
            }

            if messages_use not in role_filter:
                raise ValueError(f"Invalid messages_use value: {messages_use}")

            allowed_roles = role_filter[messages_use]
            message_lines = []

            for mes in segment:
                if mes.get("role") in allowed_roles:
                    sequence_id = mes.get("sequence_number", 0)
                    role = mes["role"]
                    content = mes.get("content", "")
                    speaker_name = mes.get("speaker_name", "")
                    time_stamp = mes.get("time_stamp", "")
                    weekday = mes.get("weekday", "")

                    time_prefix = ""
                    if time_stamp and weekday:
                        time_prefix = f"[{time_stamp}, {weekday}] "

                    if speaker_name:
                        message_lines.append(f"{time_prefix}{sequence_id // 2}.{speaker_name}: {content}")
                    else:
                        message_lines.append(f"{time_prefix}{sequence_id // 2}.{role}: {content}")

            return "\n".join(message_lines)

        max_workers = min(len(extract_list), 5)

        def process_segment_wrapper(args):
            api_call_idx, api_call_segments = args
            try:
                user_prompt_parts: List[str] = []

                global_topic_ids: List[int] = []
                if topic_id_mapping and api_call_idx < len(topic_id_mapping):
                    global_topic_ids = topic_id_mapping[api_call_idx]

                for topic_idx, topic_segment in enumerate(api_call_segments):
                    if topic_idx < len(global_topic_ids):
                        global_topic_id = global_topic_ids[topic_idx]
                    else:
                        global_topic_id = topic_idx + 1

                    topic_text = concatenate_messages(topic_segment, messages_use)
                    user_prompt_parts.append(f"--- Topic {global_topic_id} ---\n{topic_text}")

                user_prompt = "\n".join(user_prompt_parts)

                metadata_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ]

                raw_response, usage_info = self.generate_response(
                    messages=metadata_messages,
                    response_format={"type": "json_object"},
                )
                metadata_facts = self.clean_response(raw_response)

                for entry in metadata_facts:
                    entry["entry_type"] = entry_type

                return {
                    "input_prompt": metadata_messages,
                    "output_prompt": raw_response,
                    "cleaned_result": metadata_facts,
                    "usage": usage_info,
                    "entry_type": entry_type
                }

            except Exception as e:
                print(f"Error processing API call {api_call_idx}: {e}")
                return {
                    "input_prompt": [],
                    "output_prompt": "",
                    "cleaned_result": [],
                    "usage": None,
                    "entry_type": entry_type
                }

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            try:
                results = list(executor.map(process_segment_wrapper, enumerate(extract_list)))
            except Exception as e:
                print(f"Error in parallel processing: {e}")
                results = [None] * len(extract_list)

        return results

    def _call_update_llm(self, system_prompt, target_entry, candidate_sources):
        """Call LLM for update decisions."""
        target_memory = target_entry["payload"]["memory"]
        candidate_memories = [c["payload"]["memory"] for c in candidate_sources]

        user_prompt = (
            f"Target memory:{target_memory}\n"
            f"Candidate memories:\n" + "\n".join([f"- {m}" for m in candidate_memories])
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        response_text, usage_info = self.generate_response(
            messages=messages,
            response_format={"type": "json_object"}
        )

        try:
            result = json.loads(response_text)
            if "action" not in result:
                result = {"action": "ignore"}
            result["usage"] = usage_info
            return result
        except Exception:
            return {"action": "ignore", "usage": usage_info if 'usage_info' in locals() else None}


class LightMemAgent(BaseAgent):
    """Adapter that bridges LightMem memory system to BaseAgent interface."""

    METHOD_TYPE = "agentic_memory"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 1.0,
        max_tokens: int = 2000,
        provider: str = "openai",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        # LightMem specific parameters
        retrieve_num: int = 5,
        pre_compress: bool = False,
        topic_segment: bool = True,
        index_strategy: str = "embedding",
        retrieve_strategy: str = "embedding",
        update_mode: str = "offline",
        extraction_mode: str = "flat",
        messages_use: str = "user_only",
        # Embedding configuration
        embedding_provider: str = "local",
        embedding_model: str = "BAAI/bge-small-zh-v1.5",
        embedding_model_path: Optional[str] = None,
        embedding_dim: Optional[int] = None,
        embedding_api_key: Optional[str] = None,
        embedding_base_url: Optional[str] = None,
        topic_segmenter_model_path: Optional[str] = None,
        # Memory manager LLM settings
        lightmem_temperature: float = 0.1,
        lightmem_max_tokens: int = 2000,
        lightmem_top_p: float = 0.1,
        # Token limits
        max_context_tokens: int = 120000,
        **kwargs,
    ):
        super().__init__(model, temperature, max_tokens, **kwargs)

        # Save configuration
        self.retrieve_num = retrieve_num
        self.pre_compress = pre_compress
        self.topic_segment = topic_segment
        self.index_strategy = index_strategy
        self.retrieve_strategy = retrieve_strategy
        self.update_mode = update_mode
        self.extraction_mode = extraction_mode
        self.messages_use = messages_use
        self.embedding_provider = embedding_provider
        self.embedding_model = embedding_model
        self.embedding_model_path = embedding_model_path
        self.embedding_dim = embedding_dim
        self.embedding_api_key = embedding_api_key
        self.embedding_base_url = embedding_base_url
        self.topic_segmenter_model_path = topic_segmenter_model_path
        self.lightmem_temperature = lightmem_temperature
        self.lightmem_max_tokens = lightmem_max_tokens
        self.lightmem_top_p = lightmem_top_p
        self.max_context_tokens = max_context_tokens

        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._base_url = base_url or os.environ.get("OPENAI_BASE_URL")

        # Initialize LLM client for final QA
        self._llm_client: BaseLLMClient = create_llm_client(
            provider=provider,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            api_key=api_key,
            base_url=base_url,
        )

        # Initialize LLM client for LightMem internal operations (memory extraction)
        self._lightmem_llm_client: BaseLLMClient = create_llm_client(
            provider=provider,
            model=model,
            temperature=lightmem_temperature,
            max_tokens=lightmem_max_tokens,
            api_key=api_key,
            base_url=base_url,
        )

        # LightMem instances per context_id
        self._lightmem_instances: Dict[int, Any] = {}

        # Load LightMem modules
        self._setup_lightmem_path()
        self._LightMemory = None
        self._lightmem_modules_loaded = False

    def _setup_lightmem_path(self):
        """Add LightMem src to Python path."""
        lightmem_src = Path(__file__).resolve().parent / "LightMem" / "src"
        if not lightmem_src.exists():
            raise ImportError(f"LightMem source folder not found at {lightmem_src}")

        lightmem_src_str = str(lightmem_src)
        if lightmem_src_str not in sys.path:
            sys.path.insert(0, lightmem_src_str)

    def _load_lightmem_modules(self):
        """Lazy load LightMem modules."""
        if self._lightmem_modules_loaded:
            return

        from lightmem.memory.lightmem import LightMemory
        self._LightMemory = LightMemory
        self._lightmem_modules_loaded = True

    def _build_lightmem_config(self) -> Dict[str, Any]:
        """Build LightMem configuration dictionary."""
        config = {
            # Pre-processing
            "pre_compress": self.pre_compress,
            "topic_segment": self.topic_segment,
            "precomp_topic_shared": False,  # Must be False when pre_compress=False
            "messages_use": self.messages_use,

            # Index and retrieval strategy
            "index_strategy": self.index_strategy,
            "retrieve_strategy": self.retrieve_strategy,
            "update": self.update_mode,
            "extraction_mode": self.extraction_mode,

            # Metadata and summary
            "metadata_generate": True,
            "text_summary": True,

            # Memory Manager (LLM) - will be replaced with our tracked manager
            "memory_manager": {
                "model_name": "openai",
                "configs": {
                    "model": self.model,
                    "api_key": self._api_key,
                    "openai_base_url": self._base_url,
                    "temperature": self.lightmem_temperature,
                    "max_tokens": self.lightmem_max_tokens,
                    "top_p": self.lightmem_top_p,
                }
            },
        }

        # Topic segmenter configuration (required when topic_segment=True)
        if self.topic_segment:
            # When pre_compress=False, topic_segmenter needs its own model
            # Default to a multilingual BERT model for topic segmentation
            # Note: Don't use device_map to avoid accelerate dependency
            # NOTE: buffer_len is limited by BERT's max_position_embeddings (512)
            # Long messages must be handled at the adapter level before sending to LightMem
            config["topic_segmenter"] = {
                "model_name": "llmlingua-2",
                "configs": {
                    "model_name": self.topic_segmenter_model_path
                    or "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
                    "buffer_len": 512,  # Keep at 512 due to BERT model limit
                    # Don't set device_map - let PyTorch handle device placement
                }
            }

        # Text Embedder configuration
        if self.embedding_provider == "local":
            # Local HuggingFace model
            model_path = self.embedding_model_path or self.embedding_model
            config["text_embedder"] = {
                "model_name": "huggingface",
                "configs": {
                    "model": model_path,
                    "embedding_dims": self.embedding_dim,
                    "model_kwargs": {"trust_remote_code": True},
                }
            }
        elif self.embedding_provider == "openai":
            # OpenAI API embedding
            config["text_embedder"] = {
                "model_name": "openai",
                "configs": {
                    "model": self.embedding_model,
                    "api_key": self.embedding_api_key or self._api_key,
                    "openai_base_url": self.embedding_base_url or self._base_url,
                    "embedding_dims": self.embedding_dim,
                }
            }
        elif self.embedding_provider == "huggingface_api":
            # HuggingFace TEI API
            config["text_embedder"] = {
                "model_name": "huggingface",
                "configs": {
                    "model": self.embedding_model,
                    "huggingface_base_url": self.embedding_base_url,
                    "embedding_dims": self.embedding_dim,
                }
            }

        # Embedding retriever (Qdrant in-memory/local)
        if self.retrieve_strategy in ["embedding", "hybrid"]:
            # Use a unique temp path for each instance to avoid conflicts
            qdrant_path = tempfile.mkdtemp(prefix="lightmem_qdrant_")
            config["embedding_retriever"] = {
                "model_name": "qdrant",
                "configs": {
                    "collection_name": f"lightmem_ctx_{id(self)}",
                    "embedding_model_dims": self.embedding_dim or 512,
                    "path": qdrant_path,  # Use local file-based storage
                }
            }

        return config

    def _get_context_id(self) -> int:
        return self._context_id if self._context_id is not None else 0

    def _get_lightmem_instance(self, context_id: int):
        """Get or create LightMem instance for a specific context."""
        if context_id in self._lightmem_instances:
            return self._lightmem_instances[context_id]

        self._load_lightmem_modules()

        config = self._build_lightmem_config()
        # Add unique collection name for this context
        if "embedding_retriever" in config:
            config["embedding_retriever"]["configs"]["collection_name"] = f"lightmem_ctx_{context_id}_{id(self)}"

        instance = self._LightMemory.from_config(config)

        # Increase ShortMemBuffer threshold to reduce LLM API calls
        # Default is 512, we increase to 4096 to batch more segments before extraction
        if hasattr(instance, 'shortmem_buffer_manager'):
            instance.shortmem_buffer_manager.max_tokens = 4096
            print(f"[LightMem] ShortMemBuffer max_tokens set to 4096")

        # Replace the manager with our tracked version
        instance.manager = TrackedMemoryManager(
            llm_client=self._lightmem_llm_client,
            model=self.model,
            temperature=self.lightmem_temperature,
            max_tokens=self.lightmem_max_tokens,
            top_p=self.lightmem_top_p,
        )

        self._lightmem_instances[context_id] = instance
        return instance

    def _extract_timestamp_from_text(self, text: str) -> Optional[str]:
        """
        Extract timestamp from the input text.
        Looks for patterns like [2024-01-05] or [2024/01/05] at the beginning.
        Returns LightMem compatible format: "YYYY/MM/DD (Weekday) HH:MM"
        """
        import re

        # Match date patterns like [2024-01-05] or [2024/01/05]
        date_pattern = r'\[(\d{4})[-/](\d{1,2})[-/](\d{1,2})\]'
        match = re.search(date_pattern, text)

        if match:
            year, month, day = match.groups()
            try:
                from datetime import datetime as dt
                date_obj = dt(int(year), int(month), int(day))
                weekday = date_obj.strftime('%a')  # e.g., "Fri"
                return f"{year}/{month.zfill(2)}/{day.zfill(2)} ({weekday}) 00:00"
            except ValueError:
                pass

        return None

    def _split_long_message(self, content: str, max_tokens: int = 250) -> List[str]:
        """
        Split a long message into smaller chunks that fit within BERT token limits.

        LightMem uses BERT with max_position_embeddings=512, so we need to ensure
        individual user messages don't exceed this limit.

        BERT tokenizer (multilingual) produces ~1.5-2x more tokens for Chinese text
        compared to GPT tokenizer. We use a conservative default of 250 GPT tokens
        which should correspond to ~375-500 BERT tokens.

        Args:
            content: The message content to split
            max_tokens: Maximum GPT/LLM tokens per chunk (default 250)

        Returns:
            List of message chunks, each within the token limit
        """
        if not content:
            return [""]

        content_tokens = self._llm_client.count_tokens(content)

        if content_tokens <= max_tokens:
            return [content]

        # Split by sentences first
        chunks = []
        sentences = content.replace('。', '。\n').replace('！', '！\n').replace('？', '？\n').split('\n')

        current_chunk = ""
        current_tokens = 0

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            sentence_tokens = self._llm_client.count_tokens(sentence)

            if sentence_tokens > max_tokens:
                # Single sentence is too long, split by characters using binary search
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                    current_tokens = 0

                # Split long sentence into smaller parts using efficient binary search
                remaining = sentence
                while remaining:
                    chunk_part = self._find_max_chunk_by_tokens(remaining, max_tokens)
                    if not chunk_part:
                        # Fallback: take first 100 characters if binary search fails
                        chunk_part = remaining[:100]

                    if len(chunk_part) < len(remaining):
                        chunks.append(chunk_part.strip())
                        remaining = remaining[len(chunk_part):]
                    else:
                        # Last part, add to current_chunk for potential merging
                        current_chunk = chunk_part
                        current_tokens = self._llm_client.count_tokens(current_chunk)
                        break

            elif current_tokens + sentence_tokens > max_tokens:
                # Adding this sentence would exceed limit
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
                current_tokens = sentence_tokens
            else:
                current_chunk += sentence
                current_tokens += sentence_tokens

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        return chunks if chunks else [content[:1000]]  # Fallback

    def _find_max_chunk_by_tokens(self, text: str, max_tokens: int) -> str:
        """
        Use binary search to find the maximum prefix of text that fits within max_tokens.

        This is O(log n) token counting calls instead of O(n) for character-by-character.

        Args:
            text: The text to split
            max_tokens: Maximum tokens allowed

        Returns:
            The maximum prefix that fits within the token limit
        """
        if not text:
            return ""

        text_tokens = self._llm_client.count_tokens(text)
        if text_tokens <= max_tokens:
            return text

        # Binary search for the maximum length
        left, right = 1, len(text)
        result = ""

        while left <= right:
            mid = (left + right) // 2
            prefix = text[:mid]
            prefix_tokens = self._llm_client.count_tokens(prefix)

            if prefix_tokens <= max_tokens:
                result = prefix
                left = mid + 1
            else:
                right = mid - 1

        # Ensure we don't cut in the middle of a multi-byte character
        # Try to find a natural break point (space or punctuation)
        if result and len(result) < len(text):
            # Look for last natural break point in the result
            for i in range(len(result) - 1, max(0, len(result) - 20), -1):
                if result[i] in ' \t\n，。！？；：、':
                    return result[:i+1]

        return result

    def _format_messages_for_lightmem(self, text: str, timestamp: str = None) -> List[Dict[str, Any]]:
        """
        Format text into LightMem's expected message format.

        IMPORTANT: LightMem expects user+assistant message pairs. Each user message
        must be followed by an assistant message (can be empty or real). This is required
        by the sensory_memory buffer which processes messages in pairs.

        This method parses the medical dialogue format:
        - "患者: xxx" -> user message
        - "医生: xxx" -> assistant message

        Also extracts timestamp from text (e.g., [2024-01-05]) and uses it.
        """
        # Try to extract timestamp from the text itself (e.g., [2024-01-05])
        extracted_timestamp = self._extract_timestamp_from_text(text)

        if extracted_timestamp:
            timestamp = extracted_timestamp
        elif timestamp is None:
            now = datetime.now()
            timestamp = now.strftime("%Y/%m/%d") + f" ({now.strftime('%a')}) " + now.strftime("%H:%M")

        messages = []

        # Parse the medical dialogue format
        # Split by lines and identify speaker patterns
        lines = text.split('\n')

        current_role = None
        current_content = []
        dialogue_turns = []  # List of (role, content) tuples

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Check for speaker patterns
            if line.startswith('患者:') or line.startswith('患者：'):
                # Save previous content if exists
                if current_role is not None and current_content:
                    content = '\n'.join(current_content).strip()
                    if content:
                        dialogue_turns.append((current_role, content))

                # Start new user turn
                current_role = 'user'
                content_start = line[3:].strip()  # Remove "患者:" prefix
                current_content = [content_start] if content_start else []

            elif line.startswith('医生:') or line.startswith('医生：'):
                # Save previous content if exists
                if current_role is not None and current_content:
                    content = '\n'.join(current_content).strip()
                    if content:
                        dialogue_turns.append((current_role, content))

                # Start new assistant turn
                current_role = 'assistant'
                content_start = line[3:].strip()  # Remove "医生:" prefix
                current_content = [content_start] if content_start else []

            elif line.startswith('[') and line.endswith(']'):
                # Skip timestamp lines like [2024-01-05]
                continue
            elif line.startswith('[') and '健康咨询' in line:
                # Skip header lines like [健康咨询记录] or [关于xxx的健康咨询记录]
                continue
            else:
                # Continue current speaker's content
                if current_role is not None:
                    current_content.append(line)

        # Don't forget the last turn
        if current_role is not None and current_content:
            content = '\n'.join(current_content).strip()
            if content:
                dialogue_turns.append((current_role, content))

        # If no dialogue structure found, fall back to treating all text as user message
        if not dialogue_turns:
            # Remove header/timestamp lines and use remaining content
            content_lines = []
            for line in lines:
                line = line.strip()
                if line and not (line.startswith('[') and (line.endswith(']') or '健康咨询' in line)):
                    content_lines.append(line)

            if content_lines:
                dialogue_turns = [('user', '\n'.join(content_lines))]

        # Convert dialogue turns to LightMem message format
        # LightMem expects user+assistant pairs
        #
        # IMPORTANT: LightMem's SenMemBufferManager uses BERT tokenizer with max_position_embeddings=512.
        # If a SINGLE user message exceeds 512 BERT tokens, the segmenter will fail.
        # We need to split long messages to ensure each user message is within BERT's limit.
        #
        # BERT tokenizer for Chinese typically produces ~1.5-2x more tokens than GPT tokenizer,
        # so we use a conservative threshold of 250 GPT tokens (≈375-500 BERT tokens).
        MAX_USER_TOKENS = 250  # Conservative threshold for single user message

        i = 0
        while i < len(dialogue_turns):
            role, content = dialogue_turns[i]

            if role == 'user':
                # Check if next is assistant
                if i + 1 < len(dialogue_turns) and dialogue_turns[i + 1][0] == 'assistant':
                    assistant_content = dialogue_turns[i + 1][1]

                    # Check if user content needs splitting
                    user_tokens = self._llm_client.count_tokens(content)
                    if user_tokens > MAX_USER_TOKENS:
                        # Split long user message
                        user_chunks = self._split_long_message(content, max_tokens=MAX_USER_TOKENS)
                        # Pair each user chunk with empty assistant, except last one gets real assistant
                        for j, chunk in enumerate(user_chunks):
                            messages.append({
                                "time_stamp": timestamp,
                                "role": "user",
                                "content": chunk,
                                "speaker_id": "patient",
                                "speaker_name": "患者",
                            })
                            # Only last chunk gets the real assistant response
                            if j == len(user_chunks) - 1:
                                messages.append({
                                    "time_stamp": timestamp,
                                    "role": "assistant",
                                    "content": assistant_content,
                                    "speaker_id": "doctor",
                                    "speaker_name": "医生",
                                })
                            else:
                                messages.append({
                                    "time_stamp": timestamp,
                                    "role": "assistant",
                                    "content": "",
                                    "speaker_id": "doctor",
                                    "speaker_name": "医生",
                                })
                    else:
                        messages.append({
                            "time_stamp": timestamp,
                            "role": "user",
                            "content": content,
                            "speaker_id": "patient",
                            "speaker_name": "患者",
                        })
                        messages.append({
                            "time_stamp": timestamp,
                            "role": "assistant",
                            "content": assistant_content,
                            "speaker_id": "doctor",
                            "speaker_name": "医生",
                        })
                    i += 2
                else:
                    # No paired assistant
                    user_tokens = self._llm_client.count_tokens(content)
                    if user_tokens > MAX_USER_TOKENS:
                        user_chunks = self._split_long_message(content, max_tokens=MAX_USER_TOKENS)
                        for chunk in user_chunks:
                            messages.append({
                                "time_stamp": timestamp,
                                "role": "user",
                                "content": chunk,
                                "speaker_id": "patient",
                                "speaker_name": "患者",
                            })
                            messages.append({
                                "time_stamp": timestamp,
                                "role": "assistant",
                                "content": "",
                                "speaker_id": "doctor",
                                "speaker_name": "医生",
                            })
                    else:
                        messages.append({
                            "time_stamp": timestamp,
                            "role": "user",
                            "content": content,
                            "speaker_id": "patient",
                            "speaker_name": "患者",
                        })
                        messages.append({
                            "time_stamp": timestamp,
                            "role": "assistant",
                            "content": "",
                            "speaker_id": "doctor",
                            "speaker_name": "医生",
                        })
                    i += 1

            elif role == 'assistant':
                # Orphan assistant message - pair with empty user
                messages.append({
                    "time_stamp": timestamp,
                    "role": "user",
                    "content": "",
                    "speaker_id": "patient",
                    "speaker_name": "患者",
                })
                messages.append({
                    "time_stamp": timestamp,
                    "role": "assistant",
                    "content": content,
                    "speaker_id": "doctor",
                    "speaker_name": "医生",
                })
                i += 1
            else:
                i += 1

        # Ensure we have at least one message pair
        if not messages:
            messages = [
                {
                    "time_stamp": timestamp,
                    "role": "user",
                    "content": text[:5000] if len(text) > 5000 else text,
                    "speaker_id": "patient",
                    "speaker_name": "患者",
                },
                {
                    "time_stamp": timestamp,
                    "role": "assistant",
                    "content": "",
                    "speaker_id": "doctor",
                    "speaker_name": "医生",
                }
            ]

        return messages

    def memorize(self, text: str, **kwargs) -> MemoryBuildResult:
        """
        Store text into LightMem memory system using official turn-by-turn approach.

        This follows EXACTLY the official LightMem usage pattern from add_locomo.py:
        1. Each turn (user + assistant pair) is passed to add_memory separately
        2. Only the LAST turn sets force_segment=True, force_extract=True
        3. LightMem internally handles buffer overflow and BERT limits automatically

        We do NOT add any custom BERT limit handling - LightMem handles this internally
        in SenMemBufferManager.add_messages() which auto-triggers cut_with_segmenter
        when token_count exceeds max_tokens.
        """
        context_id = self._get_context_id()
        lightmem = self._get_lightmem_instance(context_id)

        # Log memorize start
        print(f"[LightMem] Starting memorize for context_id={context_id}")
        print(f"[LightMem] Input text length: {len(text)} chars, ~{self._llm_client.count_tokens(text)} tokens")

        # Format as LightMem messages (returns list of user+assistant pairs)
        messages = self._format_messages_for_lightmem(text)
        total_turns = len(messages) // 2
        used_timestamp = messages[0].get("time_stamp", "N/A") if messages else "N/A"
        print(f"[LightMem] Formatted {total_turns} turns, timestamp={used_timestamp}")

        all_memory_entries = []
        total_api_calls = 0

        # Process turn by turn, exactly like official add_locomo.py
        for turn_idx in range(total_turns):
            turn_start = turn_idx * 2
            turn_messages = messages[turn_start:turn_start + 2]

            # Validate turn structure
            if len(turn_messages) < 2:
                continue
            if turn_messages[0].get("role") != "user" or turn_messages[1].get("role") != "assistant":
                continue

            # Only force on the last turn - exactly like official implementation
            is_last_turn = (turn_idx == total_turns - 1)

            try:
                result = lightmem.add_memory(
                    messages=turn_messages,
                    force_segment=is_last_turn,
                    force_extract=is_last_turn,
                )

                # Collect results only when extraction was triggered
                api_calls = result.get("api_call_nums", 0)
                total_api_calls += api_calls

                if result.get("add_output_prompt"):
                    for output in result.get("add_output_prompt", []):
                        if output:
                            all_memory_entries.append({
                                "turn_index": turn_idx,
                                "output": str(output)[:500],
                            })

            except Exception as e:
                print(f"[LightMem] Error in add_memory for turn {turn_idx}: {e}")
                import traceback
                traceback.print_exc()
                raise RuntimeError(
                    f"LightMem add_memory failed for turn {turn_idx} of {total_turns}"
                ) from e

        self._memory_chunks.append(text)
        self._is_initialized = True

        # Get token statistics from LightMem
        token_stats = lightmem.get_token_statistics()
        print(f"[LightMem] Memorize complete: total_api_calls={total_api_calls}, memory_entries={len(all_memory_entries)}")
        print(f"[LightMem] Token stats: {token_stats.get('summary', {})}")

        return MemoryBuildResult(
            success=True,
            method="lightmem",
            action="add_memory",
            input_content=text,
            stored_content=text,
            memory_entries=all_memory_entries,
            all_passages=all_memory_entries,
            chunk_count=len(self._memory_chunks),
            extraction_result=str(all_memory_entries)[:2000],
            extra={
                "context_id": context_id,
                "total_turns": total_turns,
                "api_calls": total_api_calls,
                "lightmem_token_stats": token_stats,
            },
        )

    def query(
        self,
        question: str,
        system_message: Optional[str] = None,
        **kwargs,
    ) -> AgentResponse:
        """Query LightMem and generate response using official implementation."""
        context_id = self._get_context_id()
        lightmem = self._get_lightmem_instance(context_id)

        print(f"[LightMem] Starting query for context_id={context_id}")
        print(f"[LightMem] Question: {question[:100]}...")

        # Use LightMem's official retrieve method
        retrieved_memories = []
        memory_context = ""

        try:
            # LightMem's retrieve returns formatted string
            related = lightmem.retrieve(query=question, limit=self.retrieve_num)

            if related and related.strip():
                memory_context = related
                # Parse retrieved memories for logging
                for line in related.split('\n'):
                    if line.strip():
                        retrieved_memories.append({
                            "memory": line.strip()[:500],
                            "type": "lightmem_retrieval",
                        })
                print(f"[LightMem] Retrieved {len(retrieved_memories)} memories")
            else:
                print(f"[LightMem] No memories retrieved")
        except Exception as e:
            print(f"[LightMem] Error in retrieve: {e}")
            import traceback
            traceback.print_exc()
            raise RuntimeError("LightMem native retrieval failed") from e

        # Build full question with memory context
        if memory_context:
            full_question = f"[Retrieved Memories]\n{memory_context}\n\n[Question]\n{question}"
        else:
            full_question = question

        # Truncate if too long
        question_tokens = self._llm_client.count_tokens(full_question)
        if question_tokens > self.max_context_tokens - self.max_tokens - 500:
            # Truncate memory context
            max_memory_tokens = self.max_context_tokens - self.max_tokens - 500 - self._llm_client.count_tokens(question)
            if max_memory_tokens > 0 and memory_context:
                memory_tokens = self._llm_client.count_tokens(memory_context)
                if memory_tokens > max_memory_tokens:
                    # Simple truncation
                    ratio = max_memory_tokens / memory_tokens
                    truncated_len = int(len(memory_context) * ratio * 0.9)
                    memory_context = memory_context[:truncated_len] + "\n... [truncated]"
                    full_question = f"[Retrieved Memories]\n{memory_context}\n\n[Question]\n{question}"
                    print(f"[LightMem] Memory context truncated to {truncated_len} chars")

        # Generate response using our tracked LLM client
        messages = format_messages(full_question, system_message)
        response = self._llm_client.chat(messages)

        print(f"[LightMem] Query complete, response length: {len(response.content)} chars")

        return AgentResponse(
            output=response.content,
            query_time=0.0,
            retrieved_count=len(retrieved_memories),
            retrieved_memories=retrieved_memories,  # Fix: properly set field
            extra={
                "method": "lightmem",
            },
        )

    def retrieve(self, question: str, **kwargs) -> MemoryRetrievalResult:
        """Return LightMem's native formatted retrieval without answering."""
        context_id = self._get_context_id()
        lightmem = self._get_lightmem_instance(context_id)
        related = lightmem.retrieve(query=question, limit=self.retrieve_num) or ""
        retrieved_memories = [
            {"memory": line.strip(), "type": "lightmem_retrieval"}
            for line in related.splitlines()
            if line.strip()
        ]
        return MemoryRetrievalResult(
            memory_context=related,
            retrieved_count=len(retrieved_memories),
            retrieved_memories=retrieved_memories,
            extra={"method": "lightmem", "context_id": context_id},
        )

    def reset(self) -> None:
        """Reset agent state."""
        super().reset()
        self._lightmem_instances = {}

    def set_context_id(self, context_id: int) -> None:
        """Set context ID."""
        super().set_context_id(context_id)
