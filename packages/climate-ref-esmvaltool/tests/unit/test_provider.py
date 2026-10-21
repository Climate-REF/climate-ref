import importlib.metadata
from pathlib import Path

import pooch
from climate_ref_esmvaltool import _DATASETS_REGISTRY_NAME, ESMValToolProvider, __version__, provider

from climate_ref_core.providers import DataIngestRequest
from climate_ref_core.source_types import SourceDatasetType


def test_provider():
    assert provider.name == "ESMValTool"
    assert provider.slug == "esmvaltool"
    assert provider.version == __version__

    diagnostic_modules = importlib.resources.files("climate_ref_esmvaltool").glob("diagnostics/*.py")
    diagnostics_per_module = {
        "__init__.py": 0,
        "base.py": 0,
        "cloud_scatterplots.py": 5,
        "enso.py": 2,
        "regional_historical_changes.py": 3,
        "ozone.py": 5,
    }
    n_diagnostics = sum(diagnostics_per_module.get(f.name, 1) for f in diagnostic_modules)
    assert len(provider) == n_diagnostics


class TestESMValToolProviderHooks:
    """Tests for ESMValToolProvider lifecycle hooks."""

    def test_get_data_path(self, monkeypatch):
        """Test that get_data_path returns the pooch cache path."""
        monkeypatch.delenv("REF_DATASET_CACHE_DIR", raising=False)

        data_path = provider.get_data_path()
        assert data_path is not None
        assert isinstance(data_path, Path)
        assert data_path == Path(pooch.os_cache("climate_ref")) / "esmvaltool"

    def test_fetch_data(self, mocker):
        """Test that fetch_data calls fetch_all_files."""
        mock_fetch = mocker.patch("climate_ref_esmvaltool.fetch_all_files")
        mock_config = mocker.Mock()

        provider.fetch_data(mock_config)

        mock_fetch.assert_called()
        # Check it's using the right registry name
        call = mock_fetch.mock_calls[0]
        assert call.args[1] == _DATASETS_REGISTRY_NAME
        assert call.kwargs["output_dir"] is None

    def test_data_to_ingest_path_not_exists(self, mocker, tmp_path, caplog):
        """Nothing has been fetched yet, so there is nothing to ingest."""
        mocker.patch(
            "climate_ref_esmvaltool.registry_data_root",
            return_value=tmp_path / "nonexistent",
        )

        assert provider.data_to_ingest(mocker.Mock()) == []

        assert "No ESMValTool reference data has been fetched" in caplog.text
        # The remediation cannot be the command this hook already runs under.
        assert "ref datasets fetch-data" in caplog.text

    def test_data_to_ingest(self, mocker, tmp_path):
        """The fetched tree is requested for ingestion so the solver can select from it."""
        mocker.patch("climate_ref_esmvaltool.registry_data_root", return_value=tmp_path)

        assert provider.data_to_ingest(mocker.Mock()) == [
            DataIngestRequest(SourceDatasetType.ESMValToolReference, tmp_path)
        ]

    def test_validate_setup_env_missing(self, mocker):
        """Test validate_setup returns False when conda env is missing."""
        mock_config = mocker.Mock()
        # Mock the parent class to return False
        mocker.patch.object(
            ESMValToolProvider.__bases__[0],
            "validate_setup",
            return_value=False,
        )

        result = provider.validate_setup(mock_config)
        assert result is False

    def test_validate_setup_data_invalid(self, mocker):
        """Test validate_setup returns False when data validation fails."""
        mock_config = mocker.Mock()
        # Mock parent class to return True (conda env exists)
        mocker.patch.object(
            ESMValToolProvider.__bases__[0],
            "validate_setup",
            return_value=True,
        )
        # Mock data validation to return errors
        mocker.patch(
            "climate_ref_esmvaltool.validate_registry_cache",
            return_value=["File missing: test.nc"],
        )

        result = provider.validate_setup(mock_config)
        assert result is False

    def test_validate_setup_all_valid(self, mocker):
        """Test validate_setup returns True when all validation passes."""
        mock_config = mocker.Mock()
        # Mock parent class to return True (conda env exists)
        mocker.patch.object(
            ESMValToolProvider.__bases__[0],
            "validate_setup",
            return_value=True,
        )
        # Mock data validation to return no errors
        mocker.patch(
            "climate_ref_esmvaltool.validate_registry_cache",
            return_value=[],
        )

        result = provider.validate_setup(mock_config)
        assert result is True
