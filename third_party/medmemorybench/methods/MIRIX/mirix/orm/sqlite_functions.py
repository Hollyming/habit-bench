import base64
import sqlite3
from typing import Optional, Union

import numpy as np
from sqlalchemy import event
from sqlalchemy.engine import Engine

from mirix.constants import MAX_EMBEDDING_DIM


def adapt_array(arr):
    """
    Converts numpy array to binary for SQLite storage
    """
    if arr is None:
        return None

    if isinstance(arr, list):
        arr = np.array(arr, dtype=np.float32)
    elif not isinstance(arr, np.ndarray):
        raise ValueError(f"Unsupported type: {type(arr)}")

    # Convert to bytes and then base64 encode
    bytes_data = arr.tobytes()
    base64_data = base64.b64encode(bytes_data)
    return sqlite3.Binary(base64_data)


def convert_array(text):
    """
    Converts binary back to numpy array
    """
    if text is None:
        return None
    if isinstance(text, list):
        return np.array(text, dtype=np.float32)
    if isinstance(text, np.ndarray):
        return text

    # Handle both bytes and sqlite3.Binary
    binary_data = bytes(text) if isinstance(text, sqlite3.Binary) else text

    try:
        # First decode base64
        decoded_data = base64.b64decode(binary_data)
        # Then convert to numpy array
        return np.frombuffer(decoded_data, dtype=np.float32)
    except Exception:
        return None


def verify_embedding_dimension(embedding: np.ndarray, expected_dim: int = MAX_EMBEDDING_DIM) -> bool:
    """
    Verifies that an embedding has the expected dimension

    Args:
        embedding: Input embedding array
        expected_dim: Expected embedding dimension (default: 4096)

    Returns:
        bool: True if dimension matches, False otherwise
    """
    if embedding is None:
        return False
    return embedding.shape[0] == expected_dim


def validate_and_transform_embedding(
    embedding: Union[bytes, sqlite3.Binary, list, np.ndarray],
    expected_dim: int = MAX_EMBEDDING_DIM,
    dtype: np.dtype = np.float32,
) -> Optional[np.ndarray]:
    """
    Validates and transforms embeddings to ensure correct dimensionality.

    Args:
        embedding: Input embedding in various possible formats
        expected_dim: Expected embedding dimension (default 4096)
        dtype: NumPy dtype for the embedding (default float32)

    Returns:
        np.ndarray: Validated and transformed embedding

    Raises:
        ValueError: If embedding dimension doesn't match expected dimension
    """
    if embedding is None:
        return None

    # Convert to numpy array based on input type
    if isinstance(embedding, (bytes, sqlite3.Binary)):
        vec = convert_array(embedding)
    elif isinstance(embedding, list):
        vec = np.array(embedding, dtype=dtype)
    elif isinstance(embedding, np.ndarray):
        vec = embedding.astype(dtype)
    else:
        raise ValueError(f"Unsupported embedding type: {type(embedding)}")

    # Validate dimension
    if vec.shape[0] != expected_dim:
        raise ValueError(f"Invalid embedding dimension: got {vec.shape[0]}, expected {expected_dim}")

    return vec


def cosine_distance(embedding1, embedding2, expected_dim=MAX_EMBEDDING_DIM):
    """
    Calculate cosine distance between two embeddings

    Args:
        embedding1: First embedding
        embedding2: Second embedding
        expected_dim: Expected embedding dimension (default 4096)

    Returns:
        float: Cosine distance
    """

    if embedding1 is None or embedding2 is None:
        return 1.0

    def to_vector(value):
        if isinstance(value, (bytes, sqlite3.Binary)):
            return convert_array(value)
        return np.asarray(value, dtype=np.float32)

    try:
        vec1 = to_vector(embedding1)
        vec2 = to_vector(embedding2)
    except (TypeError, ValueError):
        return 1.0
    # MIRIX supports configurable local embedding dimensions (the benchmark
    # uses BGE-small at 512).  MAX_EMBEDDING_DIM is a storage ceiling, not a
    # requirement for every vector.  The previous 4096-only validation made
    # every valid BGE vector compare as distance 0 and destroyed ranking.
    if vec1 is None or vec2 is None or vec1.ndim != 1 or vec2.ndim != 1 or vec1.shape != vec2.shape or not vec1.size:
        return 1.0
    denominator = np.linalg.norm(vec1) * np.linalg.norm(vec2)
    if denominator == 0:
        return 1.0
    similarity = np.dot(vec1, vec2) / denominator
    distance = float(1.0 - similarity)

    return distance


def _is_sqlite_connection(dbapi_connection):
    """Recognize both synchronous sqlite3 and SQLAlchemy's aiosqlite adapter."""
    if isinstance(dbapi_connection, sqlite3.Connection):
        return True
    driver_connection = getattr(dbapi_connection, "driver_connection", None)
    return (
        driver_connection is not None
        and type(driver_connection).__module__.startswith("aiosqlite")
    )


@event.listens_for(Engine, "connect")
def register_functions(dbapi_connection, connection_record):
    """Register SQLite functions"""
    if _is_sqlite_connection(dbapi_connection):
        dbapi_connection.create_function("cosine_distance", 2, cosine_distance)


@event.listens_for(Engine, "connect")
def configure_sqlite_connection(dbapi_connection, connection_record):
    """Configure SQLite connection for better concurrency and performance"""
    if _is_sqlite_connection(dbapi_connection):
        # Enable WAL mode for better concurrency
        dbapi_connection.execute("PRAGMA journal_mode=WAL")

        # Set busy timeout for handling locked database
        dbapi_connection.execute("PRAGMA busy_timeout=30000")  # 30 seconds

        # Enable foreign key constraints
        dbapi_connection.execute("PRAGMA foreign_keys=ON")

        # Configure synchronous mode for better performance while maintaining safety
        dbapi_connection.execute("PRAGMA synchronous=NORMAL")

        # Set cache size (negative value = KB, positive = pages)
        dbapi_connection.execute("PRAGMA cache_size=-64000")  # 64MB cache

        # Configure temp store to memory for better performance
        dbapi_connection.execute("PRAGMA temp_store=MEMORY")

        # Set mmap size for better I/O performance (256MB)
        dbapi_connection.execute("PRAGMA mmap_size=268435456")

        # Configure checkpoint settings for WAL mode
        dbapi_connection.execute("PRAGMA wal_autocheckpoint=1000")

        # Configure page size for better performance (must be set before any tables are created)
        dbapi_connection.execute("PRAGMA page_size=4096")

        # Enable query optimization
        dbapi_connection.execute("PRAGMA optimize")

        # Configure locking mode for better concurrency
        dbapi_connection.execute("PRAGMA locking_mode=NORMAL")

        # Set journal size limit to prevent WAL from growing too large
        dbapi_connection.execute("PRAGMA journal_size_limit=67108864")  # 64MB limit

        # Enable automatic index creation for better query performance
        dbapi_connection.execute("PRAGMA automatic_index=ON")

        # Configure read uncommitted isolation for better concurrency (only for read operations)
        # This is safe for most read operations and improves performance
        dbapi_connection.execute("PRAGMA read_uncommitted=ON")

        # Commit all pragma changes
        dbapi_connection.commit()


# Register adapters and converters for numpy arrays
sqlite3.register_adapter(np.ndarray, adapt_array)
sqlite3.register_converter("ARRAY", convert_array)
