"""
Integration tests for provider setup functionality.
"""

import socket

import pytest

from climate_ref.provider_registry import ProviderRegistry
from climate_ref_core.testing import NetworkBlockedError, block_network


class TestProviderSetup:
    """Integration tests for provider lifecycle hooks."""

    def test_setup_is_idempotent(self, config, db):
        """Test that running setup multiple times produces the same result."""
        provider_registry = ProviderRegistry.build_from_config(config, db)

        # Get the example provider (which has no special requirements)
        example_providers = [p for p in provider_registry.providers if p.slug == "example"]
        if not example_providers:
            pytest.skip("Example provider not available")

        provider = example_providers[0]

        # Run setup twice - should not raise
        provider.setup(config)
        provider.setup(config)

        # Validate setup should return True
        assert provider.validate_setup(config) is True

    def test_setup_environment_is_idempotent(self, config, db):
        """Test that setup_environment can be called multiple times."""
        provider_registry = ProviderRegistry.build_from_config(config, db)

        for provider in provider_registry.providers:
            # Should not raise when called multiple times
            provider.setup_environment(config)
            provider.setup_environment(config)

    def test_fetch_data_is_idempotent(self, config, db):
        """Test that fetch_data can be called multiple times."""
        provider_registry = ProviderRegistry.build_from_config(config, db)

        for provider in provider_registry.providers:
            # Should not raise when called multiple times
            # Note: This may take time on first run if data needs downloading
            provider.fetch_data(config)
            provider.fetch_data(config)

    def test_validate_setup_returns_bool(self, config, db):
        """Test that validate_setup returns a boolean."""
        provider_registry = ProviderRegistry.build_from_config(config, db)

        for provider in provider_registry.providers:
            result = provider.validate_setup(config)
            assert isinstance(result, bool)


class TestBlockNetwork:
    """Tests for the network blocking utility."""

    def test_block_network_blocks_connections(self):
        """Test that block_network prevents socket connections."""
        with pytest.raises(NetworkBlockedError):
            with block_network():
                socket.socket()

    def test_block_network_restores_after_exit(self):
        """Test that network is restored after context manager exits."""
        original_socket = socket.socket

        with block_network():
            pass

        # Should be restored
        assert socket.socket is original_socket

    def test_block_network_restores_on_exception(self):
        """Test that network is restored even when an exception occurs."""
        original_socket = socket.socket

        try:
            with block_network():
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Should be restored
        assert socket.socket is original_socket
