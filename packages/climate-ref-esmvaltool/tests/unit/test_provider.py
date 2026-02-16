import importlib.metadata
from pathlib import Path

import pooch
from climate_ref_esmvaltool import ESMValToolProvider, __version__, provider


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
        assert data_path == Path(pooch.os_cache("climate_ref"))

    def test_fetch_data(self, mocker):
        """Test that fetch_data calls fetch_all_files."""
        mock_fetch = mocker.patch("climate_ref_esmvaltool.fetch_all_files")
        mock_config = mocker.Mock()

        provider.fetch_data(mock_config)

        mock_fetch.assert_called_once()
        # Check it's using the right registry name
        call_args = mock_fetch.call_args
        assert call_args[0][1] == "esmvaltool"
        assert call_args[1]["output_dir"] is None

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
