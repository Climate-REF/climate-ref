import importlib.resources
from pathlib import Path

import pooch
from climate_ref_ilamb import __version__, provider


def test_provider():
    assert provider.name == "ILAMB"
    assert provider.slug == "ilamb"
    assert provider.version == __version__

    counts = []
    for f in importlib.resources.files("climate_ref_ilamb.configure").iterdir():
        with open(f) as fin:
            counts.append(fin.read().count("sources"))
    assert len(provider) == sum(counts)


class TestILAMBProviderHooks:
    """Tests for ILAMBProvider lifecycle hooks."""

    def test_get_data_path(self):
        """Test that get_data_path returns the pooch cache path."""
        data_path = provider.get_data_path()
        assert data_path is not None
        assert isinstance(data_path, Path)
        assert data_path == Path(pooch.os_cache("climate_ref"))

    def test_fetch_data(self, mocker):
        """Test that fetch_data calls fetch_all_files for all registries."""
        mock_fetch = mocker.patch("climate_ref_ilamb.fetch_all_files")
        mock_config = mocker.Mock()

        provider.fetch_data(mock_config)

        # Should be called once for each registry (ilamb-test, ilamb, iomb)
        assert mock_fetch.call_count == 3
        registry_names = [call[0][1] for call in mock_fetch.call_args_list]
        assert "ilamb-test" in registry_names
        assert "ilamb" in registry_names
        assert "iomb" in registry_names

    def test_validate_setup_all_valid(self, mocker):
        """Test validate_setup returns True when all data is valid."""
        mock_config = mocker.Mock()
        mocker.patch(
            "climate_ref_ilamb.validate_registry_cache",
            return_value=[],
        )

        result = provider.validate_setup(mock_config)
        assert result is True

    def test_validate_setup_data_invalid(self, mocker):
        """Test validate_setup returns False when any registry has errors."""
        mock_config = mocker.Mock()
        # Return errors for one registry
        mocker.patch(
            "climate_ref_ilamb.validate_registry_cache",
            side_effect=[[], ["File missing: test.nc"], []],
        )

        result = provider.validate_setup(mock_config)
        assert result is False

    def test_validate_setup_multiple_errors(self, mocker):
        """Test validate_setup collects errors from all registries."""
        mock_config = mocker.Mock()
        mocker.patch(
            "climate_ref_ilamb.validate_registry_cache",
            side_effect=[
                ["Error 1"],
                ["Error 2", "Error 3"],
                [],
            ],
        )

        result = provider.validate_setup(mock_config)
        assert result is False
