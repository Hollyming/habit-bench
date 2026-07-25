"""GraphRAG - Knowledge Graph based Retrieval-Augmented Generation."""

import os
import sys
import time
import logging
import heapq
from typing import Optional, List, Tuple, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
from dotenv import load_dotenv
from pydantic import Field, BaseModel
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import tiktoken
import spacy
from spacy.cli import download
from nltk.stem import WordNetLemmatizer
import nltk

from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import LLMChainExtractor
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from .base import BaseAgent, MemoryBuildResult, AgentResponse
from utils.llm_client import get_usage_tracker

# Setup
load_dotenv()
logger = logging.getLogger(__name__)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)

DEFAULT_CHUNK_SIZE = 4096
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_EDGES_THRESHOLD = 0.8
DEFAULT_RETRIEVE_NUM = 5
DEFAULT_MAX_WORKERS = 3
LLM_CONTEXT_LIMIT = 127000

def _get_embeddings(embedding_model: str = None):
    """Create Embeddings instance supporting OpenAI API and local HuggingFace models."""
    provider = os.environ.get("EMBEDDING_PROVIDER", "openai")
    model_name = embedding_model or os.environ.get("DEFAULT_EMBEDDING_MODEL")

    if provider in ("local", "huggingface"):
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
            logger.debug(f"[GraphRAG] Using local HuggingFace Embedding model: {model_name}")
            return HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        except ImportError:
            logger.warning("[GraphRAG] langchain_huggingface not installed, falling back to OpenAI")
            provider = "openai"

    # OpenAI or compatible API
    kwargs = {}
    base_url = os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("OPENAI_API_KEY")

    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    if model_name:
        kwargs["model"] = model_name

    return OpenAIEmbeddings(**kwargs)


def _get_chat_model(model_name: str, temperature: float = 0.7, max_tokens: int = 100, callbacks=None):
    """Create Chat model instance with provider auto-detection."""
    model_lower = model_name.lower()

    if 'gemini' in model_lower:
        return ChatGoogleGenerativeAI(
            temperature=temperature,
            model=model_name,
            max_tokens=max_tokens,
            callbacks=callbacks,
        )

    base_url = os.environ.get("OPENAI_BASE_URL")
    api_key = os.environ.get("OPENAI_API_KEY")

    kwargs = {
        "model_name": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if api_key:
        kwargs["api_key"] = api_key
    if base_url:
        kwargs["base_url"] = base_url
    if callbacks:
        kwargs["callbacks"] = callbacks

    return ChatOpenAI(**kwargs)


class Concepts(BaseModel):
    """Extracted concepts from text."""
    concepts_list: List[str] = Field(description="List of concepts")


class AnswerCheck(BaseModel):
    """Result of answer completeness check."""
    is_complete: bool = Field(description="Whether the current context provides a complete answer")
    answer: str = Field(description="The current answer based on the context")

class DocumentProcessor:
    """Processes documents into chunks and creates embeddings."""

    def __init__(self, embedding_model: str = None, chunk_size: int = DEFAULT_CHUNK_SIZE,
                 chunk_overlap: int = DEFAULT_CHUNK_OVERLAP):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        self.embeddings = _get_embeddings(embedding_model)
        self.vector_store = None

    def process_documents(self, documents: List[Document]) -> Tuple[List, Any]:
        """Split documents and create vector store."""
        splits = self.text_splitter.split_documents(documents)
        self.vector_store = FAISS.from_documents(splits, self.embeddings)
        return splits, self.vector_store

class KnowledgeGraph:
    """Builds and manages a knowledge graph from document chunks."""

    def __init__(self, edges_threshold: float = DEFAULT_EDGES_THRESHOLD):
        self.graph = nx.Graph()
        self.lemmatizer = WordNetLemmatizer()
        self.concept_cache = {}
        self.nlp = self._load_spacy_model()
        self.edges_threshold = edges_threshold
        self._all_embeddings = []

    def _load_spacy_model(self):
        """Load spaCy model, downloading if necessary."""
        try:
            return spacy.load("en_core_web_sm")
        except OSError:
            logger.info("[GraphRAG] Downloading spaCy model...")
            download("en_core_web_sm")
            return spacy.load("en_core_web_sm")

    def build_graph(self, splits: List, llm, embedding_model) -> None:
        """Build the knowledge graph from document splits."""
        self._add_nodes(splits)
        embeddings = self._create_embeddings(splits, embedding_model)
        self._all_embeddings = list(embeddings)
        self._extract_concepts(splits, llm)
        self._add_edges(embeddings)

    def _add_nodes(self, splits: List) -> None:
        """Add nodes to the graph."""
        for i, split in enumerate(splits):
            self.graph.add_node(i, content=split.page_content)

    def _create_embeddings(self, splits: List, embedding_model) -> np.ndarray:
        """Create embeddings for document splits."""
        texts = [split.page_content for split in splits]
        return embedding_model.embed_documents(texts)

    def _extract_concepts(self, splits: List, llm) -> None:
        """Extract concepts from all splits using multi-threading."""
        max_workers = int(os.environ.get("GRAPH_RAG_MAX_WORKERS", str(DEFAULT_MAX_WORKERS)))

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_node = {
                executor.submit(self._extract_concepts_and_entities, split.page_content, llm): i
                for i, split in enumerate(splits)
            }

            for future in tqdm(as_completed(future_to_node), total=len(splits),
                               desc="Extracting concepts"):
                node = future_to_node[future]
                try:
                    concepts = future.result()
                    self.graph.nodes[node]['concepts'] = concepts
                except Exception as e:
                    logger.warning(f"[GraphRAG] Concept extraction failed for node {node}: {e}")
                    self.graph.nodes[node]['concepts'] = []

    def _extract_concepts_and_entities(self, content: str, llm) -> List[str]:
        """Extract concepts and named entities from content."""
        if content in self.concept_cache:
            return self.concept_cache[content]

        doc = self.nlp(content)
        named_entities = [
            ent.text for ent in doc.ents
            if ent.label_ in ["PERSON", "ORG", "GPE", "WORK_OF_ART"]
        ]

        concept_prompt = PromptTemplate(
            input_variables=["text"],
            template="Extract key concepts (excluding named entities) from the following text:\n\n{text}\n\nKey concepts:"
        )
        concept_chain = concept_prompt | llm.with_structured_output(Concepts)

        general_concepts = []
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = concept_chain.invoke({"text": content})
                general_concepts = result.concepts_list
                break
            except Exception as e:
                error_msg = str(e).lower()
                if "rate limit" in error_msg or "429" in error_msg:
                    if attempt < max_retries - 1:
                        wait_time = (attempt + 1) * 5
                        logger.warning(f"[GraphRAG] Rate limit hit, waiting {wait_time}s...")
                        time.sleep(wait_time)
                    else:
                        logger.warning(f"[GraphRAG] Concept extraction failed after {max_retries} retries")
                else:
                    logger.warning(f"[GraphRAG] Concept extraction error: {e}")
                    break

        all_concepts = list(set(named_entities + general_concepts))
        self.concept_cache[content] = all_concepts
        return all_concepts

    def _add_edges(self, embeddings: np.ndarray) -> None:
        """Add edges based on similarity and shared concepts."""
        similarity_matrix = cosine_similarity(embeddings)
        num_nodes = len(self.graph.nodes)

        for node1 in tqdm(range(num_nodes), desc="Adding edges"):
            for node2 in range(node1 + 1, num_nodes):
                similarity_score = similarity_matrix[node1][node2]
                if similarity_score > self.edges_threshold:
                    shared_concepts = (
                        set(self.graph.nodes[node1].get('concepts', [])) &
                        set(self.graph.nodes[node2].get('concepts', []))
                    )
                    edge_weight = self._calculate_edge_weight(
                        node1, node2, similarity_score, shared_concepts
                    )
                    self.graph.add_edge(
                        node1, node2,
                        weight=edge_weight,
                        similarity=similarity_score,
                        shared_concepts=list(shared_concepts)
                    )

    def _calculate_edge_weight(self, node1: int, node2: int, similarity_score: float,
                               shared_concepts: set, alpha: float = 0.7, beta: float = 0.3) -> float:
        """Calculate edge weight from similarity and shared concepts."""
        concepts1 = self.graph.nodes[node1].get('concepts', [])
        concepts2 = self.graph.nodes[node2].get('concepts', [])
        max_possible = min(len(concepts1), len(concepts2))
        normalized_shared = len(shared_concepts) / max_possible if max_possible > 0 else 0
        return alpha * similarity_score + beta * normalized_shared

    def _lemmatize_concept(self, concept: str) -> str:
        """Lemmatize a concept."""
        return ' '.join([self.lemmatizer.lemmatize(word) for word in concept.lower().split()])


class QueryEngine:
    """Handles queries using vector store and knowledge graph traversal."""

    def __init__(self, vector_store, knowledge_graph: KnowledgeGraph, llm,
                 retrieve_num: int = DEFAULT_RETRIEVE_NUM):
        self.vector_store = vector_store
        self.knowledge_graph = knowledge_graph
        self.llm = llm
        self.retrieve_num = retrieve_num
        self._tokenizer = tiktoken.encoding_for_model("gpt-4o-mini")
        self.answer_check_chain = self._create_answer_check_chain()

    def _create_answer_check_chain(self):
        """Create chain to check if context provides complete answer."""
        prompt = PromptTemplate(
            input_variables=["query", "context"],
            template=(
                "Given the query: '{query}'\n\n"
                "And the current context:\n{context}\n\n"
                "Does this context provide a complete answer to the query? "
                "If yes, provide the answer. If no, state that the answer is incomplete.\n\n"
                "Is complete answer (Yes/No):\nAnswer (if complete):"
            )
        )
        return prompt | self.llm.with_structured_output(AnswerCheck)

    def _truncate_context(self, text: str, max_tokens: int = LLM_CONTEXT_LIMIT) -> str:
        """Truncate text to fit within token limit."""
        tokens = self._tokenizer.encode(text)
        if len(tokens) > max_tokens:
            logger.warning(f"[GraphRAG] Context too long ({len(tokens)} tokens), truncating to {max_tokens}")
            return self._tokenizer.decode(tokens[:max_tokens])
        return text

    def query(self, query: str) -> Tuple[str, str, List[int], Dict[int, str]]:
        """Process a query and return answer with context."""
        # Extract the actual question for retrieval
        retrieval_query = self._extract_retrieval_query(query)
        logger.debug(f"[GraphRAG] Retrieval query: {retrieval_query[:100]}...")

        # Retrieve relevant documents
        relevant_docs = self._retrieve_relevant_documents(retrieval_query)

        # Expand context using graph traversal
        expanded_context, traversal_path, filtered_content, final_answer = self._expand_context(
            query, relevant_docs
        )

        # Generate final answer if not found during traversal
        if not final_answer:
            logger.debug("[GraphRAG] Generating final answer with LLM...")
            final_answer = self._generate_answer(query, expanded_context)

        return final_answer, expanded_context, traversal_path, filtered_content

    def _extract_retrieval_query(self, query: str) -> str:
        """Extract the actual question from formatted query."""
        import re

        patterns = [
            r"Now Answer the Question:\s*(.*)",
            r"Here is the conversation:\s*(.*)",
        ]

        for pattern in patterns:
            match = re.search(pattern, query, re.DOTALL)
            if match:
                return ''.join(match.groups())

        return query

    def _retrieve_relevant_documents(self, query: str) -> List:
        """Retrieve relevant documents using compression retriever."""
        retriever = self.vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": self.retrieve_num}
        )
        compressor = LLMChainExtractor.from_llm(self.llm)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=retriever
        )
        return compression_retriever.invoke(query)

    def _expand_context(self, query: str, relevant_docs: List) -> Tuple[str, List[int], Dict[int, str], str]:
        """Expand context using Dijkstra-like graph traversal."""
        expanded_context = ""
        traversal_path = []
        visited_concepts = set()
        filtered_content = {}
        final_answer = ""

        priority_queue = []
        distances = {}

        # Initialize with relevant docs
        for doc in relevant_docs:
            closest_nodes = self.vector_store.similarity_search_with_score(doc.page_content, k=1)
            if not closest_nodes:
                continue

            closest_node_content, similarity_score = closest_nodes[0]

            # Find corresponding node in knowledge graph
            closest_node = None
            for n in self.knowledge_graph.graph.nodes:
                if self.knowledge_graph.graph.nodes[n].get('content') == closest_node_content.page_content:
                    closest_node = n
                    break

            if closest_node is not None:
                priority = 1 / max(similarity_score, 1e-10)
                heapq.heappush(priority_queue, (priority, closest_node))
                distances[closest_node] = priority

        # Graph traversal
        while priority_queue:
            current_priority, current_node = heapq.heappop(priority_queue)

            if current_priority > distances.get(current_node, float('inf')):
                continue

            if current_node not in traversal_path:
                traversal_path.append(current_node)
                node_content = self.knowledge_graph.graph.nodes[current_node].get('content', '')
                node_concepts = self.knowledge_graph.graph.nodes[current_node].get('concepts', [])

                filtered_content[current_node] = node_content
                expanded_context = f"{expanded_context}\n{node_content}" if expanded_context else node_content

                # Check for complete answer
                is_complete, answer = self._check_answer(query, expanded_context)
                if is_complete:
                    final_answer = answer
                    break

                # Process concepts and explore neighbors
                node_concepts_set = set(
                    self.knowledge_graph._lemmatize_concept(c) for c in node_concepts
                )

                if not node_concepts_set.issubset(visited_concepts):
                    visited_concepts.update(node_concepts_set)

                    for neighbor in self.knowledge_graph.graph.neighbors(current_node):
                        edge_data = self.knowledge_graph.graph[current_node][neighbor]
                        edge_weight = edge_data.get('weight', 0.5)
                        distance = current_priority + (1 / max(edge_weight, 1e-10))

                        if distance < distances.get(neighbor, float('inf')):
                            distances[neighbor] = distance
                            heapq.heappush(priority_queue, (distance, neighbor))

            if final_answer:
                break

        return expanded_context, traversal_path, filtered_content, final_answer

    def _check_answer(self, query: str, context: str) -> Tuple[bool, str]:
        """Check if context provides a complete answer."""
        try:
            response = self.answer_check_chain.invoke({"query": query, "context": context})
            return response.is_complete, response.answer
        except Exception as e:
            logger.warning(f"[GraphRAG] Answer check error: {e}")
            return False, ""

    def _generate_answer(self, query: str, context: str) -> str:
        """Generate final answer using LLM."""
        response_prompt = PromptTemplate(
            input_variables=["query", "context"],
            template=(
                "Based on the following context, please answer the query very concisely.\n\n"
                "Context: {context}\n\nQuery: {query}"
            )
        )
        response_chain = response_prompt | self.llm

        truncated_context = self._truncate_context(context)
        response = response_chain.invoke({"query": query, "context": truncated_context})

        return response

class GraphRAG:
    """Main GraphRAG system coordinating document processing, graph building, and querying."""

    def __init__(
        self,
        temperature: float = 0.7,
        model_name: str = "gpt-4o-mini",
        retrieve_num: int = DEFAULT_RETRIEVE_NUM,
        max_tokens: int = 100,
        embedding_model: str = None,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        edges_threshold: float = DEFAULT_EDGES_THRESHOLD,
        callbacks: List = None,
    ):
        self.retrieve_num = retrieve_num
        self._callbacks = callbacks or []

        # Initialize LLM
        self.llm = _get_chat_model(
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            callbacks=self._callbacks,
        )

        # Initialize components
        self.embedding_model = _get_embeddings(embedding_model)
        self.document_processor = DocumentProcessor(embedding_model, chunk_size, chunk_overlap)
        self.knowledge_graph = KnowledgeGraph(edges_threshold)
        self.query_engine = None

    def process_documents(self, documents: List[Document]) -> None:
        """Process documents and build knowledge graph."""
        splits, vector_store = self.document_processor.process_documents(documents)
        self.knowledge_graph.build_graph(splits, self.llm, self.embedding_model)
        self.query_engine = QueryEngine(
            vector_store,
            self.knowledge_graph,
            self.llm,
            retrieve_num=self.retrieve_num
        )

    def query(self, query: str) -> Tuple[str, str]:
        """Query the system and return answer with context."""
        if self.query_engine is None:
            raise RuntimeError("GraphRAG not initialized. Call process_documents first.")

        final_answer, expanded_context, _, _ = self.query_engine.query(query)

        # Extract text from LLM response if needed
        if hasattr(final_answer, 'content'):
            final_answer = final_answer.content

        return str(final_answer), expanded_context

class GraphRAGAgent(BaseAgent):
    """GraphRAG Agent - Knowledge graph based RAG method adapter.

    Evaluation flow:
    - For each evaluation unit, accumulate sessions then build/rebuild graph
    - On is_last_session=True, build complete graph from ALL accumulated documents
    - Graph is rebuilt for each evaluation point to include all historical data

    This ensures each query is evaluated against a graph containing all
    sessions up to that evaluation point.
    """

    METHOD_TYPE = "rag"

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 1.0,
        max_tokens: int = 2000,
        provider: str = "openai",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        top_k: int = DEFAULT_RETRIEVE_NUM,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        edges_threshold: float = DEFAULT_EDGES_THRESHOLD,
        embedding_model: Optional[str] = None,
        embedding_provider: Optional[str] = None,
        embedding_model_path: Optional[str] = None,
        **kwargs
    ):
        super().__init__(model, temperature, max_tokens, **kwargs)

        self.top_k = top_k
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.edges_threshold = edges_threshold

        self._api_key = api_key
        self._base_url = base_url
        self._provider = provider

        self._embedding_model = embedding_model or embedding_model_path
        self._embedding_provider = embedding_provider

        # State
        self._graph_rag: Optional[GraphRAG] = None
        self._all_documents: List[Document] = []  # ALL documents accumulated so far (persists across units)
        self._pending_documents: List[Document] = []  # Documents in current unit waiting for build
        self._graph_built = False

    def _setup_environment(self) -> None:
        """Setup environment variables for API access."""
        if self._api_key:
            os.environ["OPENAI_API_KEY"] = self._api_key
        if self._base_url:
            os.environ["OPENAI_BASE_URL"] = self._base_url
        if self._embedding_provider:
            os.environ["EMBEDDING_PROVIDER"] = self._embedding_provider
        if self._embedding_model:
            os.environ["DEFAULT_EMBEDDING_MODEL"] = self._embedding_model

    def _init_graph_rag(self) -> None:
        """Initialize GraphRAG instance with token tracking callback."""
        self._setup_environment()

        from utils.langchain_callback import TokenUsageCallbackHandler
        usage_callback = TokenUsageCallbackHandler(model_name=self.model)

        self._graph_rag = GraphRAG(
            temperature=self.temperature,
            model_name=self.model,
            retrieve_num=self.top_k,
            max_tokens=self.max_tokens,
            embedding_model=self._embedding_model,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            edges_threshold=self.edges_threshold,
            callbacks=[usage_callback],
        )

    def memorize(self, text: str, is_last_session: bool = False, **kwargs) -> MemoryBuildResult:
        """Accumulate text and build graph on last session of each evaluation unit.

        The graph is rebuilt from ALL accumulated documents (not just current unit)
        to ensure queries are evaluated against complete historical context.

        Args:
            text: Text to memorize
            is_last_session: If True, triggers graph construction from all documents

        Returns:
            MemoryBuildResult with build status
        """
        # Ensure token usage is attributed to memorize phase
        get_usage_tracker().set_phase("memorize")

        # Always accumulate the text
        self._memory_chunks.append(text)
        self._is_initialized = True

        # Create document from text
        new_doc = Document(
            page_content=text,
            metadata={"source": "memory", "chunk": len(self._memory_chunks) - 1}
        )

        # Add to both lists
        self._all_documents.append(new_doc)
        self._pending_documents.append(new_doc)

        action = "accumulate"
        num_nodes = 0
        num_edges = 0

        # Build/rebuild graph when this is the last session of current evaluation unit
        if is_last_session and self._all_documents:
            total_docs = len(self._all_documents)
            new_docs = len(self._pending_documents)
            logger.info(f"[GraphRAG] Last session received, building graph from {total_docs} total documents ({new_docs} new)...")

            # Always reinitialize GraphRAG to rebuild from scratch
            # This ensures clean graph construction with all documents
            self._init_graph_rag()

            # Process ALL accumulated documents
            self._graph_rag.process_documents(self._all_documents)
            self._graph_built = True
            action = "build_knowledge_graph"

            num_nodes = len(self._graph_rag.knowledge_graph.graph.nodes)
            num_edges = len(self._graph_rag.knowledge_graph.graph.edges)
            logger.info(f"[GraphRAG] Knowledge graph built: {num_nodes} nodes, {num_edges} edges from {total_docs} documents")

            # Clear pending documents (but keep all_documents for next unit)
            self._pending_documents = []

        return MemoryBuildResult(
            success=True,
            method="graph_rag",
            action=action,
            input_content=text,
            stored_content=text,
            memory_entries=[],
            chunk_count=len(self._memory_chunks),
            extra={
                "total_documents": len(self._all_documents),
                "pending_documents": len(self._pending_documents),
                "graph_built": self._graph_built,
                "graph_nodes": num_nodes,
                "graph_edges": num_edges,
            }
        )

    def query(
        self,
        question: str,
        system_message: Optional[str] = None,
        **kwargs
    ) -> AgentResponse:
        """Query the knowledge graph.

        Args:
            question: The question to answer
            system_message: Optional system message (unused)

        Returns:
            AgentResponse with answer and retrieval info
        """
        # Ensure token usage is attributed to query phase
        get_usage_tracker().set_phase("query")

        start_time = time.time()

        if not self._graph_built or self._graph_rag is None:
            logger.error("[GraphRAG] Graph not built, cannot query")
            return AgentResponse(
                output="Error: Knowledge graph not built. Please ensure memorize() was called with is_last_session=True.",
                query_time=time.time() - start_time,
                retrieved_count=0,
                retrieved_memories=[],
                extra={"error": "graph_not_built"}
            )

        try:
            response, retrieval_context = self._graph_rag.query(question)
            response_text = response
        except Exception as e:
            logger.error(f"[GraphRAG] Query failed: {e}")
            response_text = f"Error: {e}"
            retrieval_context = ""

        query_time = time.time() - start_time

        return AgentResponse(
            output=response_text,
            query_time=query_time,
            retrieved_count=self.top_k,
            retrieved_memories=[
                {"memory": retrieval_context, "type": "graph_rag_retrieval"}
            ],
            extra={
                "method": "graph_rag",
                "chunk_size": self.chunk_size,
                "edges_threshold": self.edges_threshold,
                "graph_nodes": len(self._graph_rag.knowledge_graph.graph.nodes) if self._graph_rag else 0,
                "graph_edges": len(self._graph_rag.knowledge_graph.graph.edges) if self._graph_rag else 0,
            }
        )

    def reset(self) -> None:
        """Reset agent state for new evaluation context (new persona).

        This method is called when switching to a new persona.
        It clears ALL accumulated state to prevent cross-persona data contamination.
        """
        old_doc_count = len(self._all_documents)
        old_graph_built = self._graph_built

        super().reset()
        self._graph_rag = None
        self._all_documents = []
        self._pending_documents = []
        self._graph_built = False

        if old_doc_count > 0 or old_graph_built:
            logger.info(
                f"[GraphRAG] Agent reset: cleared {old_doc_count} documents, "
                f"graph_built={old_graph_built} -> False. "
                f"All state wiped for new persona."
            )

    def set_context_id(self, context_id: int) -> None:
        """Set context ID with cross-persona contamination guard.

        If context_id changes while documents exist, log a warning
        since this could indicate potential data leakage.
        """
        old_context_id = self._context_id

        if (old_context_id is not None
                and old_context_id != context_id
                and len(self._all_documents) > 0):
            logger.warning(
                f"[GraphRAG] context_id changed from {old_context_id} to {context_id} "
                f"while {len(self._all_documents)} documents are accumulated. "
                f"This may indicate cross-persona data leakage. "
                f"Clearing all state to prevent contamination."
            )
            # Defensive cleanup to prevent any cross-persona data leakage
            self._graph_rag = None
            self._all_documents = []
            self._pending_documents = []
            self._graph_built = False

        super().set_context_id(context_id)
        logger.debug(f"[GraphRAG] Context ID set to {context_id}")

    @property
    def has_graph(self) -> bool:
        """Check if knowledge graph has been built."""
        return self._graph_built and self._graph_rag is not None

    def get_info(self) -> Dict[str, Any]:
        """Get agent configuration and state info."""
        info = super().get_info()
        info.update({
            "top_k": self.top_k,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "edges_threshold": self.edges_threshold,
            "has_graph": self.has_graph,
            "total_documents": len(self._all_documents),
            "pending_documents": len(self._pending_documents),
        })
        if self._graph_rag and self._graph_rag.knowledge_graph:
            info["graph_nodes"] = len(self._graph_rag.knowledge_graph.graph.nodes)
            info["graph_edges"] = len(self._graph_rag.knowledge_graph.graph.edges)
        return info
