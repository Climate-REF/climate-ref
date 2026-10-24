from pathlib import Path

import pooch
from climate_ref_pmp import (
    _DEFAULT_THREAD_LIMIT,
    _THREAD_LIMIT_VARS,
    PMPDiagnosticProvider,
    __version__,
    provider,
)

from climate_ref_core.data import LayeredResource, PackagedResource
from climate_ref_core.providers import DataIngestRequest
from climate_ref_core.source_types import SourceDatasetType


def configured_provider(mocker, tmp_path):
    """Build a provider and run `configure` against a mock config rooted at `tmp_path`."""
    test_provider = PMPDiagnosticProvider("PMP-Test", "1.0")
    mock_config = mocker.Mock()
    mock_config.paths.software = tmp_path / "software"
    mock_config.ignore_datasets_file = tmp_path / "ignore.yaml"
    mock_config.ignore_datasets_file.touch()
    mock_config.ignore_datasets_resource = LayeredResource(
        packaged=PackagedResource("climate_ref", "default_ignore_datasets.yaml"),
        override=mock_config.ignore_datasets_file,
    )
    test_provider.configure(mock_config)
    return test_provider, mock_config


def test_provider():
    assert provider.name == "PMP"
    assert provider.slug == "pmp"
    assert provider.version == __version__

    assert len(provider) >= 1


class TestPMPProviderHooks:
    """Tests for PMPDiagnosticProvider lifecycle hooks."""

    def test_get_data_path(self, monkeypatch):
        """Test that get_data_path returns the pooch cache path."""
        monkeypatch.delenv("REF_DATASET_CACHE_DIR", raising=False)

        data_path = provider.get_data_path()
        assert data_path is not None
        assert isinstance(data_path, Path)
        assert data_path == Path(pooch.os_cache("climate_ref")) / "pmp-climatology"

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

    def test_configure_sets_env_vars(self, mocker, tmp_path, monkeypatch):
        """Test that configure sets the required environment variables."""
        for name in _THREAD_LIMIT_VARS:
            monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv("DASK_SCHEDULER", raising=False)
        test_provider, mock_config = configured_provider(mocker, tmp_path)

        assert "PCMDI_CONDA_EXE" in test_provider.env_overrides
        # The path is recorded without installing anything, so `configure` stays offline.
        assert test_provider.env_overrides["PCMDI_CONDA_EXE"] == str(
            mock_config.paths.software / "conda/micromamba"
        )
        assert "FI_PROVIDER" in test_provider.env_overrides
        assert test_provider.env_overrides["FI_PROVIDER"] == "tcp"
        for name in _THREAD_LIMIT_VARS:
            assert test_provider.env_overrides[name] == _DEFAULT_THREAD_LIMIT
        assert test_provider.env_overrides["DASK_SCHEDULER"] == "synchronous"

    def test_configure_keeps_user_thread_limits(self, mocker, tmp_path, monkeypatch):
        """One thread limit in the environment suppresses the default for all of them."""
        monkeypatch.setenv("OMP_NUM_THREADS", "4")
        test_provider, _ = configured_provider(mocker, tmp_path)

        for name in _THREAD_LIMIT_VARS:
            assert name not in test_provider.env_overrides

    def test_configure_keeps_user_dask_scheduler(self, mocker, tmp_path, monkeypatch):
        monkeypatch.setenv("DASK_SCHEDULER", "threads")
        test_provider, _ = configured_provider(mocker, tmp_path)

        assert "DASK_SCHEDULER" not in test_provider.env_overrides

    def test_data_to_ingest_path_not_exists(self, mocker, caplog):
        """Test data_to_ingest logs warning when data path doesn't exist."""
        mocker.patch.object(provider, "get_data_path", return_value=Path("/nonexistent"))

        assert provider.data_to_ingest(mocker.Mock()) == []

        assert "does not exist" in caplog.text

    def test_data_to_ingest_climatology_path_not_exists(self, mocker, tmp_path, caplog):
        """Test data_to_ingest logs warning when climatology subdirectory doesn't exist."""
        mocker.patch.object(provider, "get_data_path", return_value=tmp_path)

        assert provider.data_to_ingest(mocker.Mock()) == []

        assert "PMP climatology data not found" in caplog.text

    def test_data_to_ingest(self, mocker, tmp_path):
        """Test data_to_ingest requests the climatology directory."""
        climatology_dir = tmp_path / "PMP_obs4MIPsClims"
        climatology_dir.mkdir()
        mocker.patch.object(provider, "get_data_path", return_value=tmp_path)

        assert provider.data_to_ingest(mocker.Mock()) == [
            DataIngestRequest(SourceDatasetType.PMPClimatology, climatology_dir)
        ]
