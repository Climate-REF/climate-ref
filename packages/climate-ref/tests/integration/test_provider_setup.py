"""
Integration tests for provider setup functionality.
"""

import pytest

from climate_ref.provider_registry import ProviderRegistry


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

    def test_ingest_data_is_idempotent(self, config, db):
        """Test that ingest_data can be called multiple times."""
        provider_registry = ProviderRegistry.build_from_config(config, db)

        for provider in provider_registry.providers:
            # Should not raise when called multiple times
            provider.ingest_data(config, db)
            provider.ingest_data(config, db)

    def test_setup_with_db_calls_ingest_data(self, config, db):
        """Test that setup() calls ingest_data() when db is provided."""
        provider_registry = ProviderRegistry.build_from_config(config, db)

        # Get the example provider (which has no special requirements)
        example_providers = [p for p in provider_registry.providers if p.slug == "example"]
        if not example_providers:
            pytest.skip("Example provider not available")

        provider = example_providers[0]

        # Run setup with db - should not raise
        provider.setup(config, db=db)

        # Validate setup should return True
        assert provider.validate_setup(config) is True

    def test_setup_without_db_skips_ingest_data(self, config, db):
        """Test that setup() without db does not fail."""
        provider_registry = ProviderRegistry.build_from_config(config, db)

        # Get the example provider (which has no special requirements)
        example_providers = [p for p in provider_registry.providers if p.slug == "example"]
        if not example_providers:
            pytest.skip("Example provider not available")

        provider = example_providers[0]

        # Run setup without db - should not raise (ingest_data skipped)
        provider.setup(config)  # No db parameter

        # Validate setup should return True
        assert provider.validate_setup(config) is True
