"""Tests for climate_ref_core.esgf.base module."""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from climate_ref_core.esgf import CMIP6Request, ESGFRequest
from climate_ref_core.esgf.base import _deduplicate_datasets


class TestESGFRequestProtocol:
    """Tests for the ESGFRequest protocol."""

    def test_protocol_is_runtime_checkable(self):
        """Test that the protocol can be used for isinstance checks."""
        request = CMIP6Request(slug="test", facets={})
        assert isinstance(request, ESGFRequest)

    def test_non_compliant_object(self):
        """Test that non-compliant objects don't satisfy the protocol."""

        class NotARequest:
            pass

        obj = NotARequest()
        assert not isinstance(obj, ESGFRequest)

    def test_required_attributes(self):
        """Test that protocol requires expected attributes."""
        request = CMIP6Request(slug="test", facets={})

        # These should all exist
        assert hasattr(request, "slug")
        assert hasattr(request, "source_type")
        assert hasattr(request, "time_span")
        assert hasattr(request, "fetch_datasets")
        assert hasattr(request, "generate_output_path")


class TestDeduplicateDatasets:
    """Tests for _deduplicate_datasets function."""

    def test_deduplicate_single_row(self):
        """Test deduplication with a single row returns it unchanged."""
        df = pd.DataFrame(
            {
                "key": ["dataset1"],
                "value": ["data1"],
                "time_start": ["2000-01"],
                "time_end": ["2010-12"],
            }
        )
        result = _deduplicate_datasets(df)
        assert len(result) == 1
        assert result.iloc[0]["key"] == "dataset1"
        assert result.iloc[0]["time_start"] == "2000-01"
        assert result.iloc[0]["time_end"] == "2010-12"

    def test_deduplicate_multiple_same_key(self):
        """Test deduplication merges rows with same key and expands time range."""
        df = pd.DataFrame(
            {
                "key": ["dataset1", "dataset1", "dataset1"],
                "value": ["v1", "v2", "v3"],
                "time_start": ["2000-01", "2005-01", "2010-01"],
                "time_end": ["2004-12", "2009-12", "2015-12"],
            }
        )
        result = _deduplicate_datasets(df)
        assert len(result) == 1
        assert result.iloc[0]["key"] == "dataset1"
        # Takes first row's value
        assert result.iloc[0]["value"] == "v1"
        # Time range expanded to min/max
        assert result.iloc[0]["time_start"] == "2000-01"
        assert result.iloc[0]["time_end"] == "2015-12"

    def test_deduplicate_multiple_different_keys(self):
        """Test deduplication keeps all unique keys."""
        df = pd.DataFrame(
            {
                "key": ["ds1", "ds2", "ds1"],
                "value": ["a", "b", "c"],
                "time_start": ["2000-01", "2005-01", "2010-01"],
                "time_end": ["2009-12", "2015-12", "2019-12"],
            }
        )
        result = _deduplicate_datasets(df)
        assert len(result) == 2
        keys = result["key"].tolist()
        assert "ds1" in keys
        assert "ds2" in keys

    def test_deduplicate_without_time_columns(self):
        """Test deduplication works when time columns are missing."""
        df = pd.DataFrame(
            {
                "key": ["ds1", "ds1"],
                "value": ["a", "b"],
            }
        )
        result = _deduplicate_datasets(df)
        assert len(result) == 1
        assert result.iloc[0]["key"] == "ds1"

    def test_deduplicate_with_only_time_start(self):
        """Test deduplication works with only time_start column."""
        df = pd.DataFrame(
            {
                "key": ["ds1", "ds1"],
                "value": ["a", "b"],
                "time_start": ["2000-01", "2010-01"],
            }
        )
        result = _deduplicate_datasets(df)
        assert len(result) == 1
        assert result.iloc[0]["time_start"] == "2000-01"


class TestIntakeESGFMixin:
    """Tests for IntakeESGFMixin.fetch_datasets method."""

    def test_fetch_datasets_basic(self):
        """Test fetch_datasets with basic facets."""
        request = CMIP6Request(
            slug="test",
            facets={"source_id": "ACCESS-ESM1-5", "variable_id": "tas"},
        )

        mock_cat = MagicMock()
        mock_cat.df = pd.DataFrame(
            {
                "key": ["ds1"],
                "source_id": ["ACCESS-ESM1-5"],
            }
        )
        mock_cat.to_path_dict.return_value = {"ds1": ["/path/to/file.nc"]}

        with patch("climate_ref_core.esgf.base.ESGFCatalog", return_value=mock_cat):
            result = request.fetch_datasets()

        mock_cat.search.assert_called_once_with(source_id="ACCESS-ESM1-5", variable_id="tas")
        assert "key" in result.columns
        assert "files" in result.columns

    def test_fetch_datasets_with_time_span(self):
        """Test fetch_datasets with time span filter."""
        request = CMIP6Request(
            slug="test",
            facets={"source_id": "ACCESS-ESM1-5"},
            time_span=("2000-01", "2010-12"),
        )

        mock_cat = MagicMock()
        mock_cat.df = pd.DataFrame(
            {
                "key": ["ds1"],
                "source_id": ["ACCESS-ESM1-5"],
            }
        )
        mock_cat.to_path_dict.return_value = {"ds1": ["/path/to/file.nc"]}

        with patch("climate_ref_core.esgf.base.ESGFCatalog", return_value=mock_cat):
            result = request.fetch_datasets()

        # Check that time span was passed to search
        call_kwargs = mock_cat.search.call_args.kwargs
        assert call_kwargs.get("file_start") == "2000-01"
        assert call_kwargs.get("file_end") == "2010-12"

        # Check time columns are in result
        assert "time_start" in result.columns
        assert "time_end" in result.columns

    def test_fetch_datasets_with_remove_ensembles(self):
        """Test fetch_datasets with remove_ensembles flag."""
        request = CMIP6Request(
            slug="test",
            facets={"source_id": "ACCESS-ESM1-5"},
            remove_ensembles=True,
        )

        mock_cat = MagicMock()
        mock_cat.df = pd.DataFrame(
            {
                "key": ["ds1"],
                "source_id": ["ACCESS-ESM1-5"],
            }
        )
        mock_cat.to_path_dict.return_value = {"ds1": ["/path/to/file.nc"]}

        with patch("climate_ref_core.esgf.base.ESGFCatalog", return_value=mock_cat):
            request.fetch_datasets()

        mock_cat.remove_ensembles.assert_called_once()

    def test_fetch_datasets_empty_result(self):
        """Test fetch_datasets raises error when no datasets found."""
        request = CMIP6Request(
            slug="test",
            facets={"source_id": "NonExistent"},
        )

        mock_cat = MagicMock()
        mock_cat.df = None
        mock_cat.to_path_dict.return_value = {}

        with patch("climate_ref_core.esgf.base.ESGFCatalog", return_value=mock_cat):
            with pytest.raises(ValueError, match="No datasets found"):
                request.fetch_datasets()

    def test_fetch_datasets_empty_dataframe(self):
        """Test fetch_datasets raises error when df is empty."""
        request = CMIP6Request(
            slug="test",
            facets={"source_id": "NonExistent"},
        )

        mock_cat = MagicMock()
        mock_cat.df = pd.DataFrame()
        mock_cat.to_path_dict.return_value = {}

        with patch("climate_ref_core.esgf.base.ESGFCatalog", return_value=mock_cat):
            with pytest.raises(ValueError, match="No datasets found"):
                request.fetch_datasets()
