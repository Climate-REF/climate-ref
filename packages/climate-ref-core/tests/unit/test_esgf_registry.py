"""Tests for climate_ref_core.esgf.registry module."""

from unittest.mock import MagicMock, patch

import pytest

from climate_ref_core.esgf import ESGFRequest, RegistryRequest
from climate_ref_core.esgf.registry import _matches_facets, _parse_obs4ref_key, _parse_pmp_climatology_key


class TestParsePMPClimatologyKey:
    """Tests for _parse_pmp_climatology_key function."""

    def test_parse_valid_key(self):
        """Test parsing a valid PMP climatology key."""
        key = (
            "PMP_obs4MIPsClims/psl/gr/v20250224/psl_mon_ERA-5_PCMDI_gr_198101-200412_AC_v20250224_2.5x2.5.nc"
        )
        result = _parse_pmp_climatology_key(key)

        assert result["variable_id"] == "psl"
        assert result["source_id"] == "ERA-5"
        assert result["institution_id"] == "PCMDI"
        assert result["grid_label"] == "gr"
        assert result["time_range"] == "198101-200412"
        assert result["time_start"] == "198101"
        assert result["time_end"] == "200412"
        assert result["version"] == "v20250224"
        assert result["resolution"] == "2.5x2.5"
        assert result["key"] == key

    def test_parse_gpcp_key(self):
        """Test parsing a GPCP climatology key."""
        key = (
            "PMP_obs4MIPsClims/pr/gr/v20250211/"
            "pr_mon_GPCP-Monthly-3-2_RSS_gr_198301-200412_AC_v20250211_interp_2.5x2.5.nc"
        )
        result = _parse_pmp_climatology_key(key)

        assert result["variable_id"] == "pr"
        assert result["source_id"] == "GPCP-Monthly-3-2"
        assert result["institution_id"] == "RSS"

    def test_parse_ceres_key(self):
        """Test parsing a CERES climatology key."""
        key = (
            "PMP_obs4MIPsClims/rlds/gr/v20250213/"
            "rlds_mon_CERES-EBAF-4-2_RSS_gr_200101-200412_AC_v20250213_2.5x2.5.nc"
        )
        result = _parse_pmp_climatology_key(key)

        assert result["variable_id"] == "rlds"
        assert result["source_id"] == "CERES-EBAF-4-2"

    def test_parse_invalid_key_wrong_parts(self):
        """Test parsing a key with wrong number of path parts."""
        key = "invalid/path/structure.nc"
        result = _parse_pmp_climatology_key(key)
        assert result == {}

    def test_parse_invalid_filename_format(self):
        """Test parsing a key with invalid filename format."""
        key = "PMP_obs4MIPsClims/psl/gr/v20250224/invalid_filename.nc"
        result = _parse_pmp_climatology_key(key)
        assert result == {}


class TestParseObs4refKey:
    """Tests for _parse_obs4ref_key function."""

    def test_parse_valid_hadisst_key(self):
        """Test parsing a valid HadISST key."""
        key = "obs4REF/MOHC/HadISST-1-1/mon/ts/gn/v20250415/ts_mon_HadISST-1-1_PCMDI_gn_187001-202501.nc"
        result = _parse_obs4ref_key(key)

        assert result["variable_id"] == "ts"
        assert result["source_id"] == "HadISST-1-1"
        assert result["institution_id"] == "MOHC"
        assert result["grid_label"] == "gn"
        assert result["version"] == "v20250415"
        assert result["time_range"] == "187001-202501"
        assert result["time_start"] == "187001"
        assert result["time_end"] == "202501"
        assert result["key"] == key

    def test_parse_gpcp_key(self):
        """Test parsing a GPCP climatology key."""
        key = "obs4REF/NOAA-NCEI/GPCP-2-3/mon/pr/gn/v20231205/pr_mon_GPCP-Monthly-3-2_RSS_gn_198301-202303.nc"
        result = _parse_obs4ref_key(key)

        assert result["variable_id"] == "pr"
        assert result["source_id"] == "GPCP-Monthly-3-2"
        assert result["institution_id"] == "NOAA-NCEI"

    def test_parse_tropflux_key(self):
        """Test parsing a TropFlux key."""
        key = (
            "obs4REF/ESSO/TropFlux-1-0/mon/tauu/gn/v20250415/tauu_mon_TropFlux-1-0_PCMDI_gn_197901-201707.nc"
        )
        result = _parse_obs4ref_key(key)

        assert result["variable_id"] == "tauu"
        assert result["source_id"] == "TropFlux-1-0"
        assert result["institution_id"] == "ESSO"

    def test_parse_ceres_key(self):
        """Test parsing a CERES key."""
        key = (
            "obs4REF/NASA-LaRC/CERES-EBAF-4-2/mon/rlds/gn/v20230209/"
            "rlds_mon_CERES-EBAF-4-2_RSS_gn_200003-202309.nc"
        )
        result = _parse_obs4ref_key(key)

        assert result["variable_id"] == "rlds"
        assert result["source_id"] == "CERES-EBAF-4-2"
        assert result["institution_id"] == "NASA-LaRC"

    def test_parse_invalid_key_wrong_parts(self):
        """Test parsing a key with wrong number of path parts."""
        key = "invalid/path/structure.nc"
        result = _parse_obs4ref_key(key)
        assert result == {}

    def test_parse_invalid_filename_format(self):
        """Test parsing a key with invalid filename format."""
        key = "obs4REF/MOHC/HadISST-1-1/mon/ts/gn/v20250415/invalid_filename.nc"
        result = _parse_obs4ref_key(key)
        assert result == {}


class TestMatchesFacets:
    """Tests for _matches_facets function."""

    def test_matches_single_string_facet(self):
        """Test matching with a single string facet."""
        metadata = {"variable_id": "psl", "source_id": "ERA-5"}
        facets = {"variable_id": "psl"}
        assert _matches_facets(metadata, facets) is True

    def test_matches_multiple_facets(self):
        """Test matching with multiple facets."""
        metadata = {"variable_id": "psl", "source_id": "ERA-5"}
        facets = {"variable_id": "psl", "source_id": "ERA-5"}
        assert _matches_facets(metadata, facets) is True

    def test_matches_tuple_facet(self):
        """Test matching with a tuple of allowed values."""
        metadata = {"variable_id": "psl", "source_id": "ERA-5"}
        facets = {"source_id": ("ERA-5", "GPCP-Monthly-3-2")}
        assert _matches_facets(metadata, facets) is True

    def test_no_match_wrong_value(self):
        """Test no match when value doesn't match."""
        metadata = {"variable_id": "psl", "source_id": "ERA-5"}
        facets = {"variable_id": "tas"}
        assert _matches_facets(metadata, facets) is False

    def test_no_match_missing_facet(self):
        """Test no match when metadata is missing required facet."""
        metadata = {"variable_id": "psl"}
        facets = {"source_id": "ERA-5"}
        assert _matches_facets(metadata, facets) is False

    def test_empty_facets_match_all(self):
        """Test that empty facets match any metadata."""
        metadata = {"variable_id": "psl", "source_id": "ERA-5"}
        assert _matches_facets(metadata, {}) is True


class TestRegistryRequest:
    """Tests for RegistryRequest class."""

    def test_init_basic(self):
        """Test basic initialization."""
        request = RegistryRequest(
            slug="test-request",
            registry_name="pmp-climatology",
            facets={"variable_id": "psl", "source_id": "ERA-5"},
        )
        assert request.slug == "test-request"
        assert request.registry_name == "pmp-climatology"
        assert request.facets == {"variable_id": "psl", "source_id": "ERA-5"}
        assert request.source_type == "PMPClimatology"
        assert request.time_span is None

    def test_init_with_time_span(self):
        """Test initialization with time span."""
        request = RegistryRequest(
            slug="test-request",
            registry_name="pmp-climatology",
            facets={"variable_id": "psl"},
            time_span=("2000-01", "2010-12"),
        )
        assert request.time_span == ("2000-01", "2010-12")

    def test_init_custom_source_type(self):
        """Test initialization with custom source type."""
        request = RegistryRequest(
            slug="test-request",
            registry_name="custom-registry",
            facets={},
            source_type="CustomType",
        )
        assert request.source_type == "CustomType"

    def test_protocol_compliance(self):
        """Test that RegistryRequest satisfies ESGFRequest protocol."""
        request = RegistryRequest(
            slug="test-request",
            registry_name="pmp-climatology",
            facets={},
        )
        assert isinstance(request, ESGFRequest)

    def test_fetch_datasets_unknown_registry(self):
        """Test that unknown registry raises ValueError."""
        request = RegistryRequest(
            slug="test-request",
            registry_name="nonexistent-registry",
            facets={},
        )
        with pytest.raises(ValueError, match="Registry 'nonexistent-registry' not found"):
            request.fetch_datasets()

    def test_fetch_datasets_with_mock_registry(self):
        """Test fetch_datasets with a mocked pmp-climatology registry."""
        mock_registry = MagicMock()
        mock_registry.registry.keys.return_value = [
            "PMP_obs4MIPsClims/psl/gr/v20250224/psl_mon_ERA-5_PCMDI_gr_198101-200412_AC_v20250224_2.5x2.5.nc",
            "PMP_obs4MIPsClims/tas/gr/v20250224/tas_mon_ERA-5_PCMDI_gr_198101-200412_AC_v20250224_2.5x2.5.nc",
        ]
        mock_registry.fetch.side_effect = lambda key: f"/path/to/{key}"

        mock_manager = MagicMock()
        mock_manager.__getitem__ = MagicMock(return_value=mock_registry)
        mock_manager.keys.return_value = ["pmp-climatology"]

        with patch(
            "climate_ref_core.esgf.registry.dataset_registry_manager",
            mock_manager,
        ):
            request = RegistryRequest(
                slug="test-request",
                registry_name="pmp-climatology",
                facets={"variable_id": "psl"},
            )
            result = request.fetch_datasets()

            assert len(result) == 1
            assert result.iloc[0]["variable_id"] == "psl"
            assert result.iloc[0]["source_id"] == "ERA-5"
            assert "path" in result.columns
            assert "files" in result.columns

    def test_fetch_datasets_no_matches(self):
        """Test fetch_datasets returns empty DataFrame when no matches."""
        mock_registry = MagicMock()
        mock_registry.registry.keys.return_value = [
            "PMP_obs4MIPsClims/psl/gr/v20250224/psl_mon_ERA-5_PCMDI_gr_198101-200412_AC_v20250224_2.5x2.5.nc",
        ]

        mock_manager = MagicMock()
        mock_manager.__getitem__ = MagicMock(return_value=mock_registry)
        mock_manager.keys.return_value = ["pmp-climatology"]

        with patch(
            "climate_ref_core.esgf.registry.dataset_registry_manager",
            mock_manager,
        ):
            request = RegistryRequest(
                slug="test-request",
                registry_name="pmp-climatology",
                facets={"variable_id": "nonexistent"},
            )
            result = request.fetch_datasets()

            assert result.empty

    def test_fetch_datasets_multiple_matches(self):
        """Test fetch_datasets with multiple matching files."""
        mock_registry = MagicMock()
        mock_registry.registry.keys.return_value = [
            "PMP_obs4MIPsClims/rlds/gr/v20250213/rlds_mon_CERES-EBAF-4-2_RSS_gr_200101-200412_AC_v20250213_2.5x2.5.nc",
            "PMP_obs4MIPsClims/rlus/gr/v20250213/rlus_mon_CERES-EBAF-4-2_RSS_gr_200101-200412_AC_v20250213_2.5x2.5.nc",
            "PMP_obs4MIPsClims/psl/gr/v20250224/psl_mon_ERA-5_PCMDI_gr_198101-200412_AC_v20250224_2.5x2.5.nc",
        ]
        mock_registry.fetch.side_effect = lambda key: f"/path/to/{key}"

        mock_manager = MagicMock()
        mock_manager.__getitem__ = MagicMock(return_value=mock_registry)
        mock_manager.keys.return_value = ["pmp-climatology"]

        with patch(
            "climate_ref_core.esgf.registry.dataset_registry_manager",
            mock_manager,
        ):
            request = RegistryRequest(
                slug="test-request",
                registry_name="pmp-climatology",
                facets={"source_id": "CERES-EBAF-4-2"},
            )
            result = request.fetch_datasets()

            assert len(result) == 2
            assert set(result["variable_id"].tolist()) == {"rlds", "rlus"}

    def test_fetch_datasets_obs4ref_registry(self):
        """Test fetch_datasets with an obs4ref registry."""
        mock_registry = MagicMock()
        mock_registry.registry.keys.return_value = [
            "obs4REF/MOHC/HadISST-1-1/mon/ts/gn/v20250415/ts_mon_HadISST-1-1_PCMDI_gn_187001-202501.nc",
            "obs4REF/ESSO/TropFlux-1-0/mon/tauu/gn/v20250415/tauu_mon_TropFlux-1-0_PCMDI_gn_197901-201707.nc",
            "obs4REF/NASA-LaRC/CERES-EBAF-4-2/mon/rlds/gn/v20230209/rlds_mon_CERES-EBAF-4-2_RSS_gn_200003-202309.nc",
        ]
        mock_registry.fetch.side_effect = lambda key: f"/path/to/{key}"

        mock_manager = MagicMock()
        mock_manager.__getitem__ = MagicMock(return_value=mock_registry)
        mock_manager.keys.return_value = ["obs4ref"]

        with patch(
            "climate_ref_core.esgf.registry.dataset_registry_manager",
            mock_manager,
        ):
            request = RegistryRequest(
                slug="test-request",
                registry_name="obs4ref",
                source_type="obs4MIPs",
                facets={"source_id": ("HadISST-1-1", "TropFlux-1-0")},
            )
            result = request.fetch_datasets()

            assert len(result) == 2
            assert set(result["source_id"].tolist()) == {"HadISST-1-1", "TropFlux-1-0"}
            assert set(result["variable_id"].tolist()) == {"ts", "tauu"}

    def test_fetch_datasets_filters_to_latest_version(self):
        """Test that fetch_datasets returns only the latest version when multiple versions exist."""
        mock_registry = MagicMock()
        # Include multiple versions of the same dataset
        mock_registry.registry.keys.return_value = [
            "obs4REF/ESSO/TropFlux-1-0/mon/hfls/gn/v20210727/hfls_mon_TropFlux-1-0_PCMDI_gn_197901-201707.nc",
            "obs4REF/ESSO/TropFlux-1-0/mon/hfls/gn/v20250415/hfls_mon_TropFlux-1-0_PCMDI_gn_197901-201707.nc",
            "obs4REF/MOHC/HadISST-1-1/mon/ts/gn/v20210727/ts_mon_HadISST-1-1_PCMDI_gn_187001-201907.nc",
            "obs4REF/MOHC/HadISST-1-1/mon/ts/gn/v20250415/ts_mon_HadISST-1-1_PCMDI_gn_187001-202501.nc",
        ]
        mock_registry.fetch.side_effect = lambda key: f"/path/to/{key}"

        mock_manager = MagicMock()
        mock_manager.__getitem__ = MagicMock(return_value=mock_registry)
        mock_manager.keys.return_value = ["obs4ref"]

        with patch(
            "climate_ref_core.esgf.registry.dataset_registry_manager",
            mock_manager,
        ):
            request = RegistryRequest(
                slug="test-request",
                registry_name="obs4ref",
                source_type="obs4MIPs",
                facets={"source_id": ("HadISST-1-1", "TropFlux-1-0")},
            )
            result = request.fetch_datasets()

            # Should only return 2 rows (latest version of each dataset)
            assert len(result) == 2
            # All returned rows should have the latest version
            assert all(result["version"] == "v20250415")
            assert set(result["source_id"].tolist()) == {"HadISST-1-1", "TropFlux-1-0"}
