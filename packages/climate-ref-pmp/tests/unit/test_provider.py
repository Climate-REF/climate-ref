from pathlib import Path

import pooch
from climate_ref_pmp import PMPDiagnosticProvider, __version__, provider


def test_provider():
    assert provider.name == "PMP"
    assert provider.slug == "pmp"
    assert provider.version == __version__

    assert len(provider) >= 1


class TestPMPProviderHooks:
    """Tests for PMPDiagnosticProvider lifecycle hooks."""

    def test_get_data_path(self):
        """Test that get_data_path returns the pooch cache path."""
        data_path = provider.get_data_path()
        assert data_path is not None
        assert isinstance(data_path, Path)
        assert data_path == Path(pooch.os_cache("climate_ref"))

    def test_fetch_data(self, mocker):
        """Test that fetch_data calls fetch_all_files."""
        mock_fetch = mocker.patch("climate_ref_pmp.fetch_all_files")
        mock_config = mocker.Mock()

        provider.fetch_data(mock_config)

        mock_fetch.assert_called_once()
        # Check it's using the right registry name
        call_args = mock_fetch.call_args
        assert call_args[0][1] == "pmp-climatology"
        assert call_args[1]["output_dir"] is None

    def test_validate_setup_env_missing(self, mocker):
        """Test validate_setup returns False when conda env is missing."""
        mock_config = mocker.Mock()
        # Mock the parent class to return False
        mocker.patch.object(
            PMPDiagnosticProvider.__bases__[0],
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
            PMPDiagnosticProvider.__bases__[0],
            "validate_setup",
            return_value=True,
        )
        # Mock data validation to return errors
        mocker.patch(
            "climate_ref_pmp.validate_registry_cache",
            return_value=["File missing: test.nc"],
        )

        result = provider.validate_setup(mock_config)
        assert result is False

    def test_validate_setup_all_valid(self, mocker):
        """Test validate_setup returns True when all validation passes."""
        mock_config = mocker.Mock()
        # Mock parent class to return True (conda env exists)
        mocker.patch.object(
            PMPDiagnosticProvider.__bases__[0],
            "validate_setup",
            return_value=True,
        )
        # Mock data validation to return no errors
        mocker.patch(
            "climate_ref_pmp.validate_registry_cache",
            return_value=[],
        )

        result = provider.validate_setup(mock_config)
        assert result is True

    def test_configure_sets_env_vars(self, mocker, tmp_path):
        """Test that configure sets the required environment variables."""
        test_provider = PMPDiagnosticProvider("PMP-Test", "1.0")
        mock_config = mocker.Mock()
        mock_config.paths.software = tmp_path / "software"
        mock_config.ignore_datasets_file = tmp_path / "ignore.yaml"
        mock_config.ignore_datasets_file.touch()

        mocker.patch.object(test_provider, "get_conda_exe", return_value=Path("/path/to/conda"))

        test_provider.configure(mock_config)

        assert "PCMDI_CONDA_EXE" in test_provider.env_vars
        assert test_provider.env_vars["PCMDI_CONDA_EXE"] == "/path/to/conda"
        assert "FI_PROVIDER" in test_provider.env_vars
        assert test_provider.env_vars["FI_PROVIDER"] == "tcp"
