"""Methods module - memory agent implementations."""

from .base import BaseAgent, MemoryBuildResult, AgentResponse

__all__ = [
    "BaseAgent",
    "MemoryBuildResult",
    "AgentResponse",
]


def get_hipporag():
    """Lazy import HippoRAG."""
    from .hipporag import HippoRAG
    return HippoRAG


def get_graph_rag():
    """Lazy import GraphRAG."""
    from .graph_rag import GraphRAG
    return GraphRAG


def get_raptor():
    """Lazy import RAPTORMethod."""
    from .raptor import RAPTORMethod
    return RAPTORMethod


def get_self_rag():
    """Lazy import SelfRAG."""
    from .self_rag import SelfRAG
    return SelfRAG


def get_memorag():
    """Lazy import MemoRAG."""
    from .memorag import MemoRAG
    return MemoRAG
