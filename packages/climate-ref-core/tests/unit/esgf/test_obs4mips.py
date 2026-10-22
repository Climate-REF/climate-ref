"""Tests for climate_ref_core.esgf.obs4mips module."""

from unittest.mock import MagicMock, patch

import intake_esgf
import pandas as pd
import pytest

from climate_ref_core.esgf import CMIP6Request, ESGFRequest, Obs4MIPsRequest


def _catalog_recording_ceda(seen):
    def make_catalog():
        seen.append(intake_esgf.conf["solr_indices"]["esgf.ceda.ac.uk"])
        cat = MagicMock()
        cat.df = pd.DataFrame({"key": ["ds1"]})
        cat.to_path_dict.return_value = {"ds1": ["/ds1.nc"]}
        return cat

    return make_catalog


@pytest.fixture
def ceda_disabled(monkeypatch):
    monkeypatch.setitem(intake_esgf.conf["solr_indices"], "esgf.ceda.ac.uk", False)


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
        assert request.project == "obs4MIPs"
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

    @pytest.mark.usefixtures("ceda_disabled")
    def test_fetch_searches_ceda_only_while_fetching(self):
        request = Obs4MIPsRequest(slug="test-obs", facets={"source_id": "C3S-GTO-ECV-9-0"})
        seen = []

        with patch("climate_ref_core.esgf.base.ESGFCatalog", side_effect=_catalog_recording_ceda(seen)):
            request.fetch_datasets()

        assert seen == [True]
        assert intake_esgf.conf["solr_indices"]["esgf.ceda.ac.uk"] is False

    @pytest.mark.usefixtures("ceda_disabled")
    def test_cmip6_fetch_does_not_search_ceda(self):
        request = CMIP6Request(slug="test", facets={"source_id": "ACCESS-ESM1-5"})
        seen = []

        with patch("climate_ref_core.esgf.base.ESGFCatalog", side_effect=_catalog_recording_ceda(seen)):
            request.fetch_datasets()

        assert seen == [False]
