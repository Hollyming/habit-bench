"""Zep Agent - knowledge graph memory and thread memory using Zep Cloud."""

import os
import time
import json
from pathlib import Path
from typing import Optional, List, Dict, Any

from .base import BaseAgent, MemoryBuildResult, AgentResponse
from utils.llm_client import create_llm_client, BaseLLMClient


ZEP_CONTEXT_TEMPLATE = """
FACTS and ENTITIES represent relevant context to the current conversation.

# These are the most relevant facts and their valid date ranges. If the fact is about an event, the event takes place during this time.
# format: FACT (Date range: from - to)

{facts}


# These are the most relevant entities
# ENTITY_NAME: entity summary

{entities}


# These are the most relevant episodes.
# format: EPISODE

{episodes}

"""


def format_edge_date_range(edge) -> str:
    """Format edge date range."""
    return f"{edge.valid_at if edge.valid_at else 'date unknown'} - {(edge.invalid_at if edge.invalid_at else 'present')}"


def compose_search_context(
    edges: Optional[List] = None,
    nodes: Optional[List] = None,
    context_block: str = "",
    episodes: Optional[List] = None
) -> str:
    """Compose Zep search results into context string."""
    edges = edges or []
    nodes = nodes or []
    episodes = episodes or []

    facts = [f'  - {edge.fact} ({format_edge_date_range(edge)})' for edge in edges if edge]
    entities = [f'  - {node.name}: {node.summary}' for node in nodes if node]
    episode_lines = [f'  - Content: {episode.content}' for episode in episodes if episode]

    return ZEP_CONTEXT_TEMPLATE.format(
        facts='\n'.join(facts),
        entities='\n'.join(entities),
        episodes='\n'.join(episode_lines)
    )


class ZepAgent(BaseAgent):
    """Zep Agent for knowledge graph memory management with Facts, Entities, and Episodes."""

    METHOD_TYPE = "agentic_memory"

    ZEP_GRAPH_LIMIT = 9000
    ZEP_MESSAGE_LIMIT = 2000
    ZEP_MAX_CONTEXT_TOKENS = 120000

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 1.0,
        max_tokens: int = 2000,
        provider: str = "openai",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        zep_api_key: Optional[str] = None,
        retrieve_num: int = 5,
        chunk_size: int = 512,
        # Azure OpenAI config
        azure_endpoint: Optional[str] = None,
        azure_api_key: Optional[str] = None,
        azure_api_version: Optional[str] = None,
        use_azure: bool = False,
        **kwargs
    ):
        super().__init__(model, temperature, max_tokens, **kwargs)

        self.retrieve_num = retrieve_num
        self.chunk_size = chunk_size

        # API config
        self._api_key = api_key
        self._base_url = base_url
        self._provider = provider

        # Azure config
        self._use_azure = use_azure or bool(os.environ.get("AZURE_OPENAI_ENDPOINT"))
        self._azure_endpoint = azure_endpoint or os.environ.get("AZURE_OPENAI_ENDPOINT")
        self._azure_api_key = azure_api_key or os.environ.get("AZURE_OPENAI_API_KEY")
        self._azure_api_version = azure_api_version or os.environ.get("AZURE_OPENAI_API_VERSION", "2024-02-01")

        # Zep config
        self._zep_api_key = zep_api_key or os.environ.get("ZEP_API_KEY")
        if not self._zep_api_key:
            raise ValueError("ZEP_API_KEY not configured. Set ZEP_API_KEY in .env or pass as parameter.")

        # Initialize Zep client
        self._init_zep_client()

        # Initialize LLM client
        self._init_llm_client()

        # Internal state
        self._agent_start_time = time.time()
        self._current_graph_id: Optional[str] = None
        self._current_thread_id: Optional[str] = None
        self._current_user_id: Optional[str] = None

    def _init_zep_client(self) -> None:
        """Initialize Zep Cloud client."""
        from zep_cloud import Zep
        self._zep_client = Zep(api_key=self._zep_api_key)

    def _init_llm_client(self) -> None:
        """Initialize LLM client."""
        if self._use_azure:
            self._llm_client = create_llm_client(
                provider="azure",
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                api_key=self._azure_api_key,
                endpoint=self._azure_endpoint,
                api_version=self._azure_api_version,
            )
        else:
            self._llm_client = create_llm_client(
                provider=self._provider,
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                api_key=self._api_key,
                base_url=self._base_url,
            )

    def _get_ids(self, sub_dataset: str = "default") -> tuple:
        """Get Zep resource IDs."""
        context_id = self._context_id or 0
        user_id = f"user_{context_id}_{sub_dataset}"
        graph_id = f"graph_{context_id}_{sub_dataset}"
        thread_id = f"thread_{context_id}_{sub_dataset}"
        return user_id, graph_id, thread_id

    def _ensure_zep_resources(self, user_id: str, graph_id: str, thread_id: str) -> None:
        """Ensure Zep resources are created (User, Graph, Thread)."""
        if (self._current_user_id == user_id and
            self._current_graph_id == graph_id and
            self._current_thread_id == thread_id):
            return

        try:
            self._zep_client.user.add(user_id=user_id)
        except Exception as e:
            if "already exists" not in str(e).lower():
                print(f"[Zep] User creation warning: {e}")

        try:
            self._zep_client.thread.create(thread_id=thread_id, user_id=user_id)
        except Exception as e:
            if "already exists" not in str(e).lower():
                print(f"[Zep] Thread creation warning: {e}")

        try:
            self._zep_client.graph.create(graph_id=graph_id)
        except Exception as e:
            if "already exists" not in str(e).lower():
                print(f"[Zep] Graph creation warning: {e}")

        self._current_user_id = user_id
        self._current_graph_id = graph_id
        self._current_thread_id = thread_id

    def _split_text_into_chunks(self, text: str, max_size: int) -> List[str]:
        """Split text into token-bounded chunks."""
        if not text.strip():
            return []

        token_count = self._llm_client.count_tokens(text)
        if token_count <= max_size:
            return [text]

        tokens = self._tokenizer.encode(text)
        chunks: List[str] = []
        start = 0
        while start < len(tokens):
            end = min(start + max_size, len(tokens))
            chunks.append(self._tokenizer.decode(tokens[start:end]))
            if end >= len(tokens):
                break
            start = end

        return chunks

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        if not text or max_tokens <= 0:
            return ""
        tokens = self._tokenizer.encode(text)
        if len(tokens) <= max_tokens:
            return text
        return self._tokenizer.decode(tokens[:max_tokens]) + "\n... [truncated due to length]"

    def memorize(self, text: str, sub_dataset: str = "default", **kwargs) -> MemoryBuildResult:
        """Add text to Zep memory with batch processing for long content."""
        user_id, graph_id, thread_id = self._get_ids(sub_dataset)
        self._ensure_zep_resources(user_id, graph_id, thread_id)

        graph_chunks = self._split_text_into_chunks(text, self.ZEP_GRAPH_LIMIT)
        message_chunks = self._split_text_into_chunks(text, self.ZEP_MESSAGE_LIMIT)

        graph_success = 0
        for chunk in graph_chunks:
            try:
                self._zep_client.graph.add(
                    graph_id=graph_id,
                    type="text",
                    data=chunk,
                )
                graph_success += 1
            except Exception as e:
                print(f"[Zep] Graph add failed for chunk: {e}")

        message_success = 0
        from zep_cloud import Message
        for i, chunk in enumerate(message_chunks):
            try:
                messages = [
                    Message(
                        name=user_id,
                        content=chunk,
                        role="user",
                    ),
                    Message(
                        name="AI Assistant",
                        content=f"I have memorized part {i+1}/{len(message_chunks)}.",
                        role="assistant",
                    )
                ]
                self._zep_client.thread.add_messages(thread_id=thread_id, messages=messages)
                message_success += 1
            except Exception as e:
                print(f"[Zep] Thread add messages failed for chunk {i+1}: {e}")

        self._memory_chunks.append(text)
        self._is_initialized = True

        return MemoryBuildResult(
            success=True,
            method="zep",
            action="add_to_memory",
            input_content=text,
            stored_content=text,
            memory_entries=[],
            chunk_count=len(self._memory_chunks),
            extra={
                "graph_id": graph_id,
                "thread_id": thread_id,
                "graph_chunks": len(graph_chunks),
                "graph_success": graph_success,
                "message_chunks": len(message_chunks),
                "message_success": message_success,
            }
        )

    def _retrieve(self, query: str, sub_dataset: str = "default") -> Dict[str, Any]:
        """Retrieve relevant memories from Zep."""
        user_id, graph_id, thread_id = self._get_ids(sub_dataset)

        # Zep query length limit 400 chars
        retrieval_query = self._extract_retrieval_query(query)[:399]

        edges_results = None
        node_results = None
        episode_results = None
        context_block = ""

        try:
            edges_response = self._zep_client.graph.search(
                graph_id=graph_id,
                query=retrieval_query,
                scope='edges',
                limit=self.retrieve_num
            )
            edges_results = edges_response.edges if edges_response else None
        except Exception as e:
            print(f"[Zep] Edges search failed: {e}")

        try:
            nodes_response = self._zep_client.graph.search(
                graph_id=graph_id,
                query=retrieval_query,
                scope='nodes',
                limit=self.retrieve_num
            )
            node_results = nodes_response.nodes if nodes_response else None
        except Exception as e:
            print(f"[Zep] Nodes search failed: {e}")

        try:
            episodes_response = self._zep_client.graph.search(
                graph_id=graph_id,
                query=retrieval_query,
                scope='episodes',
                limit=self.retrieve_num
            )
            episode_results = episodes_response.episodes if episodes_response else None
        except Exception as e:
            print(f"[Zep] Episodes search failed: {e}")

        try:
            memory = self._zep_client.thread.get_user_context(thread_id=thread_id)
            context_block = memory.context if memory else ""
        except Exception as e:
            print(f"[Zep] Thread context retrieval failed: {e}")

        return {
            "edges": edges_results,
            "nodes": node_results,
            "episodes": episode_results,
            "context_block": context_block,
        }

    def _extract_retrieval_query(self, message: str) -> str:
        """Extract retrieval query from message."""
        import re

        # Prefer extracting event description part
        start_marker = "These are the events"
        end_markers = [
            "Your task is to",
            "Below is a list of possible subsequent events:",
        ]
        end_indices = [idx for m in end_markers if (idx := message.find(m)) != -1]
        if end_indices:
            end_idx = min(end_indices)
            start_idx = message.rfind(start_marker, 0, end_idx)
            if start_idx != -1:
                return message[start_idx:end_idx].strip()

        # Other patterns
        patterns = [
            r"Now Answer the Question:\s*(.*)",
            r"Here is the conversation:\s*(.*)",
        ]

        for pattern in patterns:
            match = re.search(pattern, message, re.DOTALL)
            if match:
                return ''.join(match.groups())

        return message

    def _llm_response(self, context: str, question: str) -> str:
        """Call LLM to generate response."""
        system_prompt = "You are a helpful expert assistant answering questions from users based on the provided context."

        # Reserve output and prompt overhead, then distribute remaining budget between question and context.
        system_tokens = self._llm_client.count_tokens(system_prompt)
        reserved_tokens = self.max_tokens + 500 + system_tokens
        available_tokens = max(self.ZEP_MAX_CONTEXT_TOKENS - reserved_tokens, 0)

        question_budget = min(4096, max(available_tokens // 3, 512))
        question_text = self._truncate_to_tokens(question, question_budget)
        question_tokens = self._llm_client.count_tokens(question_text)

        context_budget = max(available_tokens - question_tokens, 0)
        context_text = self._truncate_to_tokens(context, context_budget)

        prompt = f"""Your task is to briefly answer the question. You are given the following context from the previous conversation. If you don't know how to answer the question, abstain from answering.

    {context_text}

    {question_text}

Answer:"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]

        response = self._llm_client.chat(messages)
        return response.content

    def query(
        self,
        question: str,
        system_message: Optional[str] = None,
        sub_dataset: str = "default",
        query_id: Optional[str] = None,
        **kwargs
    ) -> AgentResponse:
        """Query the agent."""
        # Retrieve relevant memories
        retrieval_results = self._retrieve(question, sub_dataset)

        # Compose context
        retrieved_context = compose_search_context(
            edges=retrieval_results["edges"],
            nodes=retrieval_results["nodes"],
            context_block=retrieval_results["context_block"],
            episodes=retrieval_results["episodes"],
        )

        # Call LLM to generate response
        response_text = self._llm_response(retrieved_context, question)

        # Save retrieval results
        if query_id is not None:
            self._save_retrieval_context(
                query_id=query_id,
                context_id=self._context_id,
                sub_dataset=sub_dataset,
                retrieved_context=retrieved_context,
                response=response_text,
            )

        return AgentResponse(
            output=response_text,
            query_time=0.0,
            retrieved_count=self.retrieve_num,
            retrieved_memories=[
                {"memory": retrieved_context, "type": "zep_retrieval"}
            ],  # Fix: properly set field
            extra={
                "method": "zep",
                "graph_id": self._current_graph_id,
                "thread_id": self._current_thread_id,
            }
        )

    def _save_retrieval_context(
        self,
        query_id: str,
        context_id: Optional[int],
        sub_dataset: str,
        retrieved_context: str,
        response: str,
    ) -> None:
        """Save retrieval context to file."""
        save_dir = Path(f"./outputs/rag_retrieved/zep/k_{self.retrieve_num}/{sub_dataset}/chunksize_{self.chunk_size}")
        save_dir.mkdir(parents=True, exist_ok=True)

        save_path = save_dir / f"query_{query_id}_context_{context_id}.json"
        paragraphs = [p for p in retrieved_context.replace("\r\n", "\n").split("\n") if p.strip()]

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump({
                "retrieved_context_paragraphs": paragraphs,
                "response": response,
            }, f, ensure_ascii=False, indent=2)

    def reset(self) -> None:
        """Reset agent."""
        super().reset()
        self._agent_start_time = time.time()

    def set_context_id(self, context_id: int) -> None:
        """Set context ID."""
        super().set_context_id(context_id)
        self._agent_start_time = time.time()

    def save_agent(self, save_path: str) -> None:
        """Save agent state."""
        save_dir = Path(save_path)
        save_dir.mkdir(parents=True, exist_ok=True)

        with open(save_dir / "messages.txt", "w") as f:
            f.write("agent finished memorization")

        metadata = {
            "user_id": self._current_user_id,
            "graph_id": self._current_graph_id,
            "thread_id": self._current_thread_id,
            "memory_chunk_count": len(self._memory_chunks),
            "context_id": self._context_id,
        }
        with open(save_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"[Zep] Agent state saved to: {save_dir}")

    def load_agent(self, load_path: str) -> bool:
        """Load agent state."""
        load_dir = Path(load_path)
        if not load_dir.exists():
            print(f"[Zep] Load path not found: {load_dir}")
            return False

        metadata_path = load_dir / "metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                metadata = json.load(f)

            self._current_user_id = metadata.get("user_id")
            self._current_graph_id = metadata.get("graph_id")
            self._current_thread_id = metadata.get("thread_id")
            self._context_id = metadata.get("context_id")

        print(f"[Zep] Agent state loaded from: {load_dir}")
        return True

    @property
    def memory_count(self) -> int:
        """Get current memory count."""
        return len(self._memory_chunks)

    def get_info(self) -> Dict[str, Any]:
        """Get agent info."""
        info = super().get_info()
        info.update({
            "retrieve_num": self.retrieve_num,
            "chunk_size": self.chunk_size,
            "use_azure": self._use_azure,
            "current_graph_id": self._current_graph_id,
            "current_thread_id": self._current_thread_id,
        })
        return info
