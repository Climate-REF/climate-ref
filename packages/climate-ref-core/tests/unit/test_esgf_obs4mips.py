"""Tests for climate_ref_core.esgf.obs4mips module."""

from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import pytest

from climate_ref_core.esgf import ESGFRequest, Obs4MIPsRequest


class TestObs4MIPsRequest:
    """Tests for Obs4MIPsRequest class."""

    def test_init_basic(self):
        """Test basic initialization."""
        request = Obs4MIPsRequest(
            slug="test-obs",
            facets={"source_id": "GPCP-SG", "variable_id": "pr"},
        )
        assert request.slug == "test-obs"
        assert request.facets == {"source_id": "GPCP-SG", "variable_id": "pr"}
        assert request.remove_ensembles is False
        assert request.time_span is None
        assert request.source_type == "obs4MIPs"

    def test_init_with_time_span(self):
        """Test initialization with time span."""
        request = Obs4MIPsRequest(
            slug="test-obs",
            facets={"source_id": "GPCP-SG"},
            time_span=("1990-01", "2020-12"),
        )
        assert request.time_span == ("1990-01", "2020-12")

    def test_repr(self):
        """Test string representation."""
        request = Obs4MIPsRequest(
            slug="test-obs",
            facets={"source_id": "GPCP-SG"},
        )
        repr_str = repr(request)
        assert "Obs4MIPsRequest" in repr_str
        assert "test-obs" in repr_str
        assert "GPCP-SG" in repr_str

    def test_protocol_compliance(self):
        """Test that Obs4MIPsRequest satisfies ESGFRequest protocol."""
        request = Obs4MIPsRequest(
            slug="test-obs",
            facets={"source_id": "GPCP-SG"},
        )
        assert isinstance(request, ESGFRequest)

    def test_obs4mips_path_items_valid(self):
        """Test that all path items are in available facets."""
        for item in Obs4MIPsRequest.obs4mips_path_items:
            assert item in Obs4MIPsRequest.avail_facets

    def test_obs4mips_filename_paths_valid(self):
        """Test that all filename path items are in available facets."""
        for item in Obs4MIPsRequest.obs4mips_filename_paths:
            assert item in Obs4MIPsRequest.avail_facets

    def test_init_invalid_path_item(self):
        """Test that invalid path items raise ValueError."""

        class BadObs4MIPsRequest(Obs4MIPsRequest):
            obs4mips_path_items = ("invalid_facet",)

        with pytest.raises(ValueError, match=r"Path item.*not in available facets"):
            BadObs4MIPsRequest(slug="test", facets={})

    def test_init_invalid_filename_path(self):
        """Test that invalid filename path items raise ValueError."""

        class BadObs4MIPsRequest(Obs4MIPsRequest):
            obs4mips_filename_paths = ("invalid_facet",)

        with pytest.raises(ValueError, match=r"Filename path.*not in available facets"):
            BadObs4MIPsRequest(slug="test", facets={})


class TestObs4MIPsRequestGenerateOutputPath:
    """Tests for Obs4MIPsRequest.generate_output_path method."""

    def test_generate_output_path_matching_variable(self):
        """Test output path when filename variable matches dataset variable_id."""
        request = Obs4MIPsRequest(
            slug="test-obs4mips",
            facets={"source_id": "GPCP-SG", "variable_id": "pr"},
        )

        metadata = pd.Series(
            {
                "activity_id": "obs4MIPs",
                "institution_id": "NASA-GSFC",
                "source_id": "GPCP-SG",
                "variable_id": "pr",
                "grid_label": "gn",
                "version": "20200101",
            }
        )

        mock_ds = MagicMock()
        mock_ds.dims = {"time": 100}
        mock_ds.variable_id = "pr"
        mock_time = MagicMock()
        mock_time.min.return_value.dt.strftime.return_value.item.return_value = "199701"
        mock_time.max.return_value.dt.strftime.return_value.item.return_value = "202012"
        mock_ds.time = mock_time

        result = request.generate_output_path(metadata, mock_ds, Path("pr_GPCP-SG_gn_19970101-20201231.nc"))

        assert "obs4MIPs" in str(result)
        assert "GPCP-SG" in str(result)
        assert "v20200101" in str(result)
        assert result.suffix == ".nc"

    def test_generate_output_path_different_variable(self):
        """Test output path when filename variable differs from dataset variable_id."""
        request = Obs4MIPsRequest(
            slug="test-obs4mips-alt",
            facets={"source_id": "GPCP-SG", "variable_id": "pr"},
        )

        metadata = pd.Series(
            {
                "activity_id": "obs4MIPs",
                "institution_id": "NASA-GSFC",
                "source_id": "GPCP-SG",
                "variable_id": "pr",
                "grid_label": "gn",
                "version": "20200101",
            }
        )

        mock_ds = MagicMock()
        mock_ds.dims = {"time": 100}
        mock_ds.variable_id = "pr"
        mock_time = MagicMock()
        mock_time.min.return_value.dt.strftime.return_value.item.return_value = "199701"
        mock_time.max.return_value.dt.strftime.return_value.item.return_value = "202012"
        mock_ds.time = mock_time

        # Filename starts with different variable
        result = request.generate_output_path(metadata, mock_ds, Path("precip_GPCP-SG_gn.nc"))

        assert "obs4MIPs" in str(result)
        assert result.suffix == ".nc"
