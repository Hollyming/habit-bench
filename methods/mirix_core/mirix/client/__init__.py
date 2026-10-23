"""
Mirix Client Module

This module provides client implementations for interacting with Mirix agents:
- AbstractClient: Base interface for all clients
- MirixClient: For cloud deployments (server accessed via REST API)

For embedded/local deployments, use mirix.LocalClient instead.
"""

from mirix.client.client import AbstractClient
from mirix.client.remote_client import MirixClient

__all__ = ["AbstractClient", "MirixClient"]
