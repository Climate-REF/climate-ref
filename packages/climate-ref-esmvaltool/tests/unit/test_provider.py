import builtins
import importlib.metadata
from pathlib import Path

import pooch
from climate_ref_esmvaltool import _DATASETS_REGISTRY_NAME, ESMValToolProvider, __version__, provider


def test_provider():
    assert provider.name == "ESMValTool"
    assert provider.slug == "esmvaltool"
    assert provider.version == __version__

    diagnostic_modules = importlib.resources.files("climate_ref_esmvaltool").glob("diagnostics/*.py")
    diagnostics_per_module = {
        "__init__.py": 0,
        "base.py": 0,
        "reference.py": 0,
        "cloud_scatterplots.py": 5,
        "enso.py": 2,
        "regional_historical_changes.py": 3,
        "ozone.py": 5,
    }
    n_diagnostics = sum(diagnostics_per_module.get(f.name, 1) for f in diagnostic_modules)
    assert len(provider) == n_diagnostics


class TestESMValToolProviderHooks:
    """Tests for ESMValToolProvider lifecycle hooks."""

    def test_get_data_path(self):
        """Test that get_data_path returns the pooch cache path."""
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

    def _mock_registry(self, mocker, abspath):
        """Point the ESMValTool data registry at ``abspath`` for ingest_data tests."""
        mock_registry = mocker.Mock()
        mock_registry.abspath = abspath
        mocker.patch(
            "climate_ref_esmvaltool.dataset_registry_manager",
            {_DATASETS_REGISTRY_NAME: mock_registry},
        )

    def test_ingest_data_skips_when_climate_ref_not_installed(self, mocker, caplog):
        """Test ingest_data gracefully skips when climate-ref package is not installed."""
        mock_config = mocker.Mock()
        mock_db = mocker.Mock()

        original_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name.startswith("climate_ref.datasets"):
                raise ImportError("No module named 'climate_ref'")
            return original_import(name, *args, **kwargs)

        mocker.patch.object(builtins, "__import__", side_effect=mock_import)

        provider.ingest_data(mock_config, mock_db)

        assert "climate-ref package not installed" in caplog.text

    def test_ingest_data_path_not_exists(self, mocker, tmp_path, caplog):
        """Test ingest_data logs a warning when the ESMValTool data directory is absent."""
        mock_config = mocker.Mock()
        mock_db = mocker.Mock()

        # ESMValTool subdirectory does not exist under the registry cache.
        self._mock_registry(mocker, tmp_path)

        provider.ingest_data(mock_config, mock_db)

        assert "ESMValTool reference data not found" in caplog.text

    def test_ingest_data_no_valid_datasets(self, mocker, tmp_path, caplog):
        """Test ingest_data handles the case when no valid datasets are found."""
        mock_config = mocker.Mock()
        mock_db = mocker.Mock()

        (tmp_path / "ESMValTool").mkdir()
        self._mock_registry(mocker, tmp_path)
        mocker.patch(
            "climate_ref.datasets.ingest_datasets",
            side_effect=ValueError("No valid datasets found"),
        )

        provider.ingest_data(mock_config, mock_db)

        assert "No valid ESMValTool reference datasets found" in caplog.text

    def test_ingest_data_success(self, mocker, tmp_path):
        """Test ingest_data calls ingest_datasets with the expected parameters."""
        mock_config = mocker.Mock()
        mock_db = mocker.Mock()

        (tmp_path / "ESMValTool").mkdir()
        self._mock_registry(mocker, tmp_path)

        mock_stats = mocker.Mock()
        mock_ingest = mocker.patch(
            "climate_ref.datasets.ingest_datasets",
            return_value=mock_stats,
        )

        provider.ingest_data(mock_config, mock_db)

        mock_ingest.assert_called_once()
        assert mock_ingest.call_args.kwargs["skip_invalid"] is True
        mock_stats.log_summary.assert_called_once()
