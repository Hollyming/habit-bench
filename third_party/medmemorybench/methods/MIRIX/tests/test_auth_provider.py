"""
Authentication Provider Tests for Mirix

Tests the custom authentication provider system for dynamic header injection
in LLM requests. This system supports claims-based tickets and other short-lived
authentication tokens.

Test Coverage:
- Auth provider registration and retrieval
- Stateful providers and token refresh logic
- Integration with LLMConfig
- Error handling

Usage:
    pytest tests/test_auth_provider.py -v
"""

import time
from datetime import datetime, timedelta
from typing import Dict

import pytest

from mirix.llm_api.auth_provider import (
    AuthProvider,
    get_auth_provider,
    list_auth_providers,
    register_auth_provider,
    unregister_auth_provider,
)
from mirix.schemas.llm_config import LLMConfig

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture(autouse=True)
def cleanup_registry():
    """Clean up auth provider registry before and after each test."""
    # Clean up before test
    for provider_name in list_auth_providers():
        unregister_auth_provider(provider_name)

    yield

    # Clean up after test
    for provider_name in list_auth_providers():
        unregister_auth_provider(provider_name)


@pytest.fixture
def simple_auth_provider():
    """Simple auth provider that returns static headers."""

    class SimpleAuthProvider(AuthProvider):
        def get_auth_headers(self) -> Dict[str, str]:
            return {
                "Authorization": "Bearer test-token",
                "X-Custom-Header": "test-value",
            }

    return SimpleAuthProvider()


class TestAuthProvider:
    """Test suite for AuthProvider abstract base class."""

    def test_cannot_instantiate_abstract_class(self):
        """Verify AuthProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            AuthProvider()  # type: ignore[abstract]

    def test_subclass_must_implement_get_auth_headers(self):
        """Verify subclasses must implement get_auth_headers."""

        class IncompleteProvider(AuthProvider):
            pass

        with pytest.raises(TypeError):
            IncompleteProvider()  # type: ignore[abstract]


class TestProviderRegistration:
    """Test suite for auth provider registration and retrieval."""

    def test_register_retrieve_and_use_provider(self, simple_auth_provider):
        """Test registering, retrieving, and using an auth provider."""
        register_auth_provider("test_provider", simple_auth_provider)

        provider = get_auth_provider("test_provider")
        assert provider is not None
        assert provider is simple_auth_provider

        # Verify headers are returned correctly
        headers = provider.get_auth_headers()
        assert headers == {
            "Authorization": "Bearer test-token",
            "X-Custom-Header": "test-value",
        }

    def test_overwrite_existing_provider(self, simple_auth_provider):
        """Test that registering same name overwrites existing provider."""
        register_auth_provider("test_provider", simple_auth_provider)

        class NewAuthProvider(AuthProvider):
            def get_auth_headers(self) -> Dict[str, str]:
                return {"Authorization": "Bearer new-token"}

        new_provider = NewAuthProvider()
        register_auth_provider("test_provider", new_provider)

        provider = get_auth_provider("test_provider")
        headers = provider.get_auth_headers()

        assert headers["Authorization"] == "Bearer new-token"

    def test_stateful_provider_with_caching(self):
        """Test provider that maintains state with token caching."""

        class StatefulProvider(AuthProvider):
            def __init__(self):
                self.token = None
                self.call_count = 0

            def get_auth_headers(self) -> Dict[str, str]:
                self.call_count += 1
                if self.token is None:
                    self.token = f"token-{self.call_count}"
                return {"Authorization": f"Bearer {self.token}"}

        provider = StatefulProvider()
        register_auth_provider("stateful", provider)

        retrieved = get_auth_provider("stateful")

        # First call creates token
        headers1 = retrieved.get_auth_headers()
        assert headers1["Authorization"] == "Bearer token-1"

        # Second call reuses cached token
        headers2 = retrieved.get_auth_headers()
        assert headers2["Authorization"] == "Bearer token-1"
        assert retrieved.call_count == 2


class TestTokenRefreshProvider:
    """Test suite for providers with token refresh logic."""

    def test_token_refresh_on_expiry(self):
        """Test that tokens are refreshed when expired, simulating real-world claims-based auth."""

        class RefreshingProvider(AuthProvider):
            def __init__(self, lifetime_seconds=1):
                self.lifetime_seconds = lifetime_seconds
                self.token = None
                self.expiry = None
                self.refresh_count = 0

            def _is_expired(self):
                if self.expiry is None:
                    return True
                return datetime.now() >= self.expiry

            def _refresh_token(self):
                # Simulate HTTP call to auth service
                time.sleep(0.01)
                self.refresh_count += 1
                self.token = f"token-{self.refresh_count}"
                self.expiry = datetime.now() + timedelta(seconds=self.lifetime_seconds)

            def get_auth_headers(self) -> Dict[str, str]:
                if self._is_expired():
                    self._refresh_token()
                return {"Authorization": f"Bearer {self.token}"}

        provider = RefreshingProvider(lifetime_seconds=1)
        register_auth_provider("refreshing", provider)

        retrieved = get_auth_provider("refreshing")

        # First call should refresh
        headers1 = retrieved.get_auth_headers()
        assert headers1["Authorization"] == "Bearer token-1"
        assert retrieved.refresh_count == 1

        # Second call should NOT refresh (token still valid)
        headers2 = retrieved.get_auth_headers()
        assert headers2["Authorization"] == "Bearer token-1"
        assert retrieved.refresh_count == 1

        # Wait for expiry and verify refresh happens
        time.sleep(1.1)
        headers3 = retrieved.get_auth_headers()
        assert headers3["Authorization"] == "Bearer token-2"
        assert retrieved.refresh_count == 2


class TestRegistryManagement:
    """Test suite for provider registry management."""

    def test_list_providers(self, simple_auth_provider):
        """Test listing registered providers."""
        assert list_auth_providers() == []

        register_auth_provider("provider1", simple_auth_provider)
        assert list_auth_providers() == ["provider1"]

        class Provider2(AuthProvider):
            def get_auth_headers(self) -> Dict[str, str]:
                return {}

        register_auth_provider("provider2", Provider2())
        providers = list_auth_providers()
        assert len(providers) == 2
        assert "provider1" in providers
        assert "provider2" in providers

    def test_unregister_provider(self, simple_auth_provider):
        """Test unregistering a provider."""
        register_auth_provider("test_provider", simple_auth_provider)
        assert "test_provider" in list_auth_providers()

        result = unregister_auth_provider("test_provider")
        assert result is True
        assert "test_provider" not in list_auth_providers()

    def test_unregister_nonexistent_provider(self):
        """Test unregistering a provider that doesn't exist."""
        result = unregister_auth_provider("nonexistent")
        assert result is False

    def test_get_nonexistent_provider(self):
        """Test retrieving a provider that doesn't exist."""
        provider = get_auth_provider("nonexistent")
        assert provider is None


class TestLLMConfigIntegration:
    """Test suite for integration with LLMConfig."""

    def test_llm_config_with_auth_provider(self, simple_auth_provider):
        """Test that LLMConfig can reference and use auth provider."""
        register_auth_provider("config_provider", simple_auth_provider)

        config = LLMConfig(
            model="gpt-4",
            model_endpoint_type="openai",
            context_window=8192,
            model_endpoint="https://custom-endpoint.com/v1",
            auth_provider="config_provider",
        )

        assert config.auth_provider == "config_provider"

        # Verify we can retrieve and use the provider
        provider = get_auth_provider(config.auth_provider)
        assert provider is not None

        headers = provider.get_auth_headers()
        assert "Authorization" in headers


class TestErrorHandling:
    """Test suite for error handling."""

    def test_provider_that_raises_exception(self):
        """Test handling of provider that raises an exception during auth."""

        class FailingProvider(AuthProvider):
            def get_auth_headers(self) -> Dict[str, str]:
                raise ValueError("Simulated auth failure")

        provider = FailingProvider()
        register_auth_provider("failing", provider)

        retrieved = get_auth_provider("failing")

        with pytest.raises(ValueError, match="Simulated auth failure"):
            retrieved.get_auth_headers()
