"""Tests for climate_ref_core.esgf.cmip6 module."""

from unittest.mock import MagicMock

import pytest

from climate_ref_core.esgf import CMIP6Request, ESGFRequest
from climate_ref_core.esgf.cmip6 import prefix_to_filename


class TestCMIP6Request:
    """Tests for CMIP6Request class."""

    def test_init_basic(self):
        """Test basic initialization."""
        request = CMIP6Request(
            slug="test-request",
            facets={"source_id": "ACCESS-ESM1-5", "variable_id": "tas"},
        )
        assert request.slug == "test-request"
        assert request.facets == {"source_id": "ACCESS-ESM1-5", "variable_id": "tas"}
        assert request.remove_ensembles is False
        assert request.time_span is None
        assert request.source_type == "CMIP6"

    def test_init_with_time_span(self):
        """Test initialization with time span."""
        request = CMIP6Request(
            slug="test-request",
            facets={"source_id": "ACCESS-ESM1-5"},
            time_span=("2000-01", "2010-12"),
        )
        assert request.time_span == ("2000-01", "2010-12")

    def test_init_with_remove_ensembles(self):
        """Test initialization with remove_ensembles flag."""
        request = CMIP6Request(
            slug="test-request",
            facets={"source_id": "ACCESS-ESM1-5"},
            remove_ensembles=True,
        )
        assert request.remove_ensembles is True

    def test_repr(self):
        """Test string representation."""
        request = CMIP6Request(
            slug="test-request",
            facets={"source_id": "ACCESS-ESM1-5"},
        )
        repr_str = repr(request)
        assert "CMIP6Request" in repr_str
        assert "test-request" in repr_str
        assert "ACCESS-ESM1-5" in repr_str

    def test_protocol_compliance(self):
        """Test that CMIP6Request satisfies ESGFRequest protocol."""
        request = CMIP6Request(
            slug="test-request",
            facets={"source_id": "ACCESS-ESM1-5"},
        )
        assert isinstance(request, ESGFRequest)

    def test_cmip6_path_items_valid(self):
        """Test that all path items are in available facets."""
        for item in CMIP6Request.cmip6_path_items:
            assert item in CMIP6Request.available_facets

    def test_cmip6_filename_paths_valid(self):
        """Test that all filename path items are in available facets."""
        for item in CMIP6Request.cmip6_filename_paths:
            assert item in CMIP6Request.available_facets

    def test_init_invalid_path_item(self):
        """Test that invalid path items raise ValueError."""

        class BadCMIP6Request(CMIP6Request):
            cmip6_path_items = ("invalid_facet",)

        with pytest.raises(ValueError, match=r"Path item.*not in available facets"):
            BadCMIP6Request(slug="test", facets={})

    def test_init_invalid_filename_path(self):
        """Test that invalid filename path items raise ValueError."""

        class BadCMIP6Request(CMIP6Request):
            cmip6_filename_paths = ("invalid_facet",)

        with pytest.raises(ValueError, match=r"Filename path.*not in available facets"):
            BadCMIP6Request(slug="test", facets={})


class TestPrefixToFilename:
    """Tests for prefix_to_filename function."""

    def test_with_time_dimension(self):
        """Test filename generation with time dimension."""
        mock_ds = MagicMock()
        mock_ds.dims = {"time": 100, "lat": 180, "lon": 360}
        mock_time = MagicMock()
        mock_time.min.return_value.dt.strftime.return_value.item.return_value = "200001"
        mock_time.max.return_value.dt.strftime.return_value.item.return_value = "201012"
        mock_ds.time = mock_time

        result = prefix_to_filename(mock_ds, "tas_Amon_ACCESS-ESM1-5_historical_r1i1p1f1_gn")
        assert result == "tas_Amon_ACCESS-ESM1-5_historical_r1i1p1f1_gn_200001-201012.nc"

    def test_without_time_dimension(self):
        """Test filename generation without time dimension."""
        mock_ds = MagicMock()
        mock_ds.dims = {"lat": 180, "lon": 360}

        result = prefix_to_filename(mock_ds, "areacella_fx_ACCESS-ESM1-5")
        assert result == "areacella_fx_ACCESS-ESM1-5.nc"
