"""
Lightweight utilities for Mirix client package.

Contains only essential utility functions needed by the client,
without heavy dependencies.
"""

import json
from datetime import datetime, timezone


def get_utc_time() -> datetime:
    """Get the current UTC time"""
    return datetime.now(timezone.utc)


def json_dumps(data, indent=2):
    """
    JSON serializer that handles datetime objects.

    Args:
        data: Data to serialize
        indent: JSON indentation level

    Returns:
        str: JSON string
    """

    def safe_serializer(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Type {type(obj)} not serializable")

    return json.dumps(data, indent=indent, default=safe_serializer, ensure_ascii=False)
