import pytest
import sqlalchemy

from climate_ref.database import Database
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

    def test_build_from_config_no_network(self, config, mocker):
        """Building the registry must not fetch the ignore datasets file."""
        db = mocker.MagicMock()
        mocker.patch("climate_ref.provider_registry.import_provider")
        mocker.patch("climate_ref.provider_registry._register_provider")
        get_mock = mocker.patch("climate_ref.config.requests.get")

        ProviderRegistry.build_from_config(config, db)

        get_mock.assert_not_called()

    def test_create_no_register(self, config, mocker):
        """``register=False`` builds the registry without touching the database."""
        db = mocker.MagicMock()

        mock_import = mocker.patch("climate_ref.provider_registry.import_provider")
        mock_register = mocker.patch("climate_ref.provider_registry._register_provider")

        registry = ProviderRegistry.build_from_config(config, db, register=False)
        assert registry.providers == [mock_import.return_value]

        mock_register.assert_not_called()
        db.session.begin.assert_not_called()


class TestProviderRegistryReadOnly:
    """
    The API serves a database that may be mounted read-only.

    ``build_from_config`` registers provider/diagnostic rows, which is the only
    step that writes to the database. These tests pin down that a read-only
    consumer can build the registry when (and only when) registration is skipped,
    guarding the read-only API startup path against regressions.

    See https://github.com/Climate-REF/ref-app/issues/31.
    """

    @staticmethod
    def _read_only_db(config, *, register: bool) -> Database:
        """Build a writable DB (optionally registering providers), then reopen it read-only."""
        writable = Database.from_config(config, run_migrations=True)
        if register:
            # Populate provider/diagnostic rows as ``ref providers setup`` would.
            ProviderRegistry.build_from_config(config, writable)
        writable.close()
        return Database.from_config(config, read_only=True)

    def test_read_only_registry_builds_without_writing(self, config):
        """The API path: build the registry against a fully-set-up, read-only DB."""
        ro_db = self._read_only_db(config, register=True)
        try:
            registry = ProviderRegistry.build_from_config(config, ro_db, register=False)

            assert [p.slug for p in registry.providers] == ["example"]
            # A pure reader must never open a write transaction.
            assert not ro_db.session.in_transaction()
        finally:
            ro_db.close()

    def test_read_only_registry_with_register_raises(self, config):
        """Registering against a read-only DB is exactly what crashed startup in #31."""
        ro_db = self._read_only_db(config, register=False)
        try:
            with pytest.raises(sqlalchemy.exc.OperationalError, match="readonly database"):
                ProviderRegistry.build_from_config(config, ro_db, register=True)
        finally:
            ro_db.close()

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
