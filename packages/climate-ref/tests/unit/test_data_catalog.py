from unittest import mock

import pandas as pd
import pytest

from climate_ref.data_catalog import DataCatalog
from climate_ref.datasets.base import DatasetAdapter
from climate_ref.datasets.mixins import FinaliseableDatasetAdapterMixin


@pytest.fixture
def mock_adapter():
    """A mock adapter that does NOT implement FinaliseableDatasetAdapterMixin."""
    adapter = mock.MagicMock(spec=DatasetAdapter)
    adapter.load_catalog.return_value = pd.DataFrame(
        {"variable_id": ["tas", "pr"], "finalised": [True, True]}
    )
    return adapter


@pytest.fixture
def mock_finaliseable_adapter():
    """A mock adapter that implements FinaliseableDatasetAdapterMixin."""

    class _MockAdapter(FinaliseableDatasetAdapterMixin, DatasetAdapter):
        pass

    adapter = mock.MagicMock(spec=_MockAdapter)
    # Make isinstance checks work for FinaliseableDatasetAdapterMixin
    adapter.__class__ = _MockAdapter
    adapter.load_catalog.return_value = pd.DataFrame(
        {
            "variable_id": ["tas", "pr"],
            "finalised": [False, True],
        }
    )
    return adapter


@pytest.fixture
def mock_db():
    return mock.MagicMock()


class TestDataCatalogToFrame:
    def test_lazy_loads_on_first_access(self, mock_db, mock_adapter):
        catalog = DataCatalog(database=mock_db, adapter=mock_adapter)

        result = catalog.to_frame()

        mock_adapter.load_catalog.assert_called_once_with(mock_db)
        assert len(result) == 2

    def test_caches_after_first_load(self, mock_db, mock_adapter):
        catalog = DataCatalog(database=mock_db, adapter=mock_adapter)

        first = catalog.to_frame()
        second = catalog.to_frame()

        # Only one call to load_catalog despite two to_frame() calls
        mock_adapter.load_catalog.assert_called_once()
        assert first is second


class TestDataCatalogFinalise:
    def test_returns_subset_unchanged_for_non_finaliseable_adapter(self, mock_db, mock_adapter):
        catalog = DataCatalog(database=mock_db, adapter=mock_adapter)
        subset = pd.DataFrame({"variable_id": ["tas"], "finalised": [False]})

        result = catalog.finalise(subset)

        pd.testing.assert_frame_equal(result, subset)

    def test_returns_subset_unchanged_when_all_finalised(self, mock_db, mock_finaliseable_adapter):
        catalog = DataCatalog(database=mock_db, adapter=mock_finaliseable_adapter)
        subset = pd.DataFrame({"variable_id": ["tas"], "finalised": [True]})

        result = catalog.finalise(subset)

        pd.testing.assert_frame_equal(result, subset)
        mock_finaliseable_adapter.finalise_datasets.assert_not_called()

    def test_returns_subset_unchanged_when_no_finalised_column(self, mock_db, mock_finaliseable_adapter):
        catalog = DataCatalog(database=mock_db, adapter=mock_finaliseable_adapter)
        subset = pd.DataFrame({"variable_id": ["tas"]})

        result = catalog.finalise(subset)

        pd.testing.assert_frame_equal(result, subset)
        mock_finaliseable_adapter.finalise_datasets.assert_not_called()

    def test_calls_finalise_datasets_with_unfinalised_data(self, mock_db, mock_finaliseable_adapter):
        catalog = DataCatalog(database=mock_db, adapter=mock_finaliseable_adapter)
        subset = pd.DataFrame({"variable_id": ["tas", "pr"], "finalised": [False, True]})

        finalised_result = subset.copy()
        finalised_result["finalised"] = True
        mock_finaliseable_adapter.finalise_datasets.return_value = finalised_result

        result = catalog.finalise(subset)

        mock_finaliseable_adapter.finalise_datasets.assert_called_once_with(mock_db, subset)
        assert result["finalised"].all()

    def test_updates_cached_dataframe(self, mock_db, mock_finaliseable_adapter):
        catalog = DataCatalog(database=mock_db, adapter=mock_finaliseable_adapter)

        # First load to populate cache
        cached_df = catalog.to_frame()
        assert not cached_df["finalised"].all()

        # Now finalise a subset
        subset = cached_df[cached_df["finalised"] == False].copy()  # noqa: E712
        finalised_result = subset.copy()
        finalised_result["finalised"] = True
        mock_finaliseable_adapter.finalise_datasets.return_value = finalised_result

        # After finalise_datasets writes to DB, subsequent load_catalog
        # calls should return the updated state
        mock_finaliseable_adapter.load_catalog.return_value = pd.DataFrame(
            {"variable_id": ["tas", "pr"], "finalised": [True, True]}
        )

        catalog.finalise(subset)

        # The cache was invalidated, so to_frame() reloads from DB
        # which now reflects the finalised state
        updated_cache = catalog.to_frame()
        assert updated_cache["finalised"].all()
