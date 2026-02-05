import pytest

from climate_ref.provider_registry import ProviderRegistry


class TestProviderRegistry:
    def test_create(self, config, mocker):
        db = mocker.MagicMock()

        mock_import = mocker.patch("climate_ref.provider_registry.import_provider")
        mock_register = mocker.patch("climate_ref.provider_registry._register_provider")

        registry = ProviderRegistry.build_from_config(config, db)
        assert len(registry.providers) == 1
        assert registry.providers[0] == mock_import.return_value

        assert mock_import.call_count == 1
        mock_register.assert_called_once_with(db, mock_import.return_value)

    def test_get_provider_found(self, mocker):
        """Test get() returns provider when found."""
        mock_provider = mocker.MagicMock()
        mock_provider.slug = "test-provider"

        registry = ProviderRegistry(providers=[mock_provider])
        result = registry.get("test-provider")
        assert result == mock_provider

    def test_get_provider_not_found(self, mocker):
        """Test get() raises KeyError when provider not found."""
        mock_provider = mocker.MagicMock()
        mock_provider.slug = "test-provider"

        registry = ProviderRegistry(providers=[mock_provider])
        with pytest.raises(KeyError, match="No provider with slug matching: nonexistent"):
            registry.get("nonexistent")

    def test_get_metric(self, mocker):
        """Test get_metric() retrieves diagnostic from provider."""
        mock_diagnostic = mocker.MagicMock()
        mock_provider = mocker.MagicMock()
        mock_provider.slug = "test-provider"
        mock_provider.get.return_value = mock_diagnostic

        registry = ProviderRegistry(providers=[mock_provider])
        result = registry.get_metric("test-provider", "test-diagnostic")

        assert result == mock_diagnostic
        mock_provider.get.assert_called_once_with("test-diagnostic")
