"""
Integration tests verifying lazy vs complete solver parity.

Uses the large pre-generated parquet catalogs from ``tests/test-data/esgf-catalog/``
to verify that solving with a lazy DataCatalog produces the same executions
as solving with a fully materialised catalog.
"""

from collections.abc import Mapping
from typing import cast
from unittest import mock

import pandas as pd
import pytest
from climate_ref_esmvaltool import provider as esmvaltool_provider
from climate_ref_example import provider as example_provider
from climate_ref_ilamb import provider as ilamb_provider
from climate_ref_pmp import provider as pmp_provider

from climate_ref.data_catalog import DataCatalog
from climate_ref.datasets.base import DatasetAdapter
from climate_ref.datasets.cmip6 import CMIP6DatasetAdapter
from climate_ref.datasets.mixins import FinaliseableDatasetAdapterMixin
from climate_ref.solver import solve_executions
from climate_ref_core.datasets import SourceDatasetType
from climate_ref_core.exceptions import InvalidDiagnosticException
from climate_ref_core.providers import DiagnosticProvider

ALL_PROVIDERS = [example_provider, pmp_provider, esmvaltool_provider, ilamb_provider]


def _to_tuples(details: list[dict]) -> list[tuple[str, str, str]]:
    """Extract sorted (provider, diagnostic, dataset_key) tuples from execution details."""
    return [(r["provider"], r["diagnostic"], r["dataset_key"]) for r in details]


def _collect_execution_details(
    data_catalog: Mapping[SourceDatasetType, pd.DataFrame | DataCatalog],
    providers: list[DiagnosticProvider],
) -> list[dict]:
    """
    Solve and return detailed execution info including dataset instance_ids.

    Sorted by (provider, diagnostic, dataset_key) for stable comparison.
    """
    results = []
    for provider in providers:
        for diagnostic in provider.diagnostics():
            try:
                for execution in solve_executions(data_catalog, diagnostic, provider):
                    datasets = {}
                    for source_type, ds_collection in execution.datasets.items():
                        instance_ids = sorted(ds_collection.instance_id.unique().tolist())
                        datasets[str(source_type.value)] = instance_ids

                    results.append(
                        {
                            "provider": execution.provider.slug,
                            "diagnostic": execution.diagnostic.slug,
                            "dataset_key": execution.dataset_key,
                            "datasets": datasets,
                        }
                    )
            except InvalidDiagnosticException:
                continue

    results.sort(key=lambda r: (r["provider"], r["diagnostic"], r["dataset_key"]))
    return results


class _DRSSimulatedAdapter(FinaliseableDatasetAdapterMixin):
    """
    Mock adapter that simulates DRS-only parsing followed by finalisation.

    The unfinalised DataFrame has file-open columns NaN'd out, as a real DRS
    parser would produce (no netCDF I/O).  ``finalise_datasets()`` restores
    those values from the original fully-parsed DataFrame, mirroring what
    ``CMIP6DatasetAdapter.finalise_datasets()`` does when the solver requests
    finalisation for a group.
    """

    columns_requiring_finalisation = CMIP6DatasetAdapter.columns_requiring_finalisation
    """
    Columns that require opening the netCDF file and are absent from DRS-only parsing.

    These are NaN'd in the DRS simulation and restored by the mock finaliser.
    """

    def __init__(self, df_full: pd.DataFrame, df_drs: pd.DataFrame):
        self._df_full = df_full
        self._df_drs = df_drs

    def load_catalog(self, db, **kwargs):
        """Return the DRS-parsed (unfinalised) catalog."""
        return self._df_drs.copy()

    def finalise_datasets(self, db, datasets):
        """Restore file-open columns from the original full DataFrame."""
        result = self._df_full.loc[datasets.index].copy()
        result["finalised"] = True
        return result


def _build_drs_catalogs(
    esgf_data_catalog: dict[SourceDatasetType, pd.DataFrame],
) -> dict[SourceDatasetType, DataCatalog]:
    """
    Create DataCatalog instances that simulate DRS-only parsing.

    File-open columns are NaN'd to reflect what a DRS parser produces without
    opening any netCDF files.  The ``_DRSSimulatedAdapter`` restores full
    metadata during ``finalise_datasets()``, mimicking what the real
    ``CMIP6DatasetAdapter`` does when the solver requests finalisation.
    """
    mock_db = mock.MagicMock()
    catalogs = {}
    for source_type, df in esgf_data_catalog.items():
        df_drs = df.copy()
        df_drs["finalised"] = False
        cols_to_nan = _DRSSimulatedAdapter.columns_requiring_finalisation & set(df_drs.columns)
        df_drs[list(cols_to_nan)] = pd.NA

        adapter = _DRSSimulatedAdapter(df_full=df, df_drs=df_drs)
        catalogs[source_type] = DataCatalog(database=mock_db, adapter=cast("DatasetAdapter", adapter))

    return catalogs


@pytest.fixture(scope="session")
def complete_executions(esgf_data_catalog_trimmed):
    """Baseline: solve with plain DataFrames (fully finalised data)."""
    return _collect_execution_details(esgf_data_catalog_trimmed, ALL_PROVIDERS)


@pytest.fixture(scope="session")
def lazy_executions(esgf_data_catalog_trimmed):
    """Solve with DRS-simulated catalogs (file-open columns NaN'd, restored on finalise)."""
    return _collect_execution_details(_build_drs_catalogs(esgf_data_catalog_trimmed), ALL_PROVIDERS)


class TestDRSSimulatedParity:
    """
    Verify that DRS-simulated unfinalised data produces
    the same solve results after mock finalisation as fully finalised data.

    This exercises the full DataCatalog.finalise() code path: the solver encounters
    unfinalised rows, calls finalise_datasets() per group, invalidates the cache,
    and the resulting executions must match the baseline.
    """

    def test_execution_keys_identical(self, complete_executions, lazy_executions):
        """The exact set of execution tuples must match after finalisation."""
        raw = _to_tuples(complete_executions)
        drs = _to_tuples(lazy_executions)
        assert len(raw) > 0
        assert drs == raw, (
            f"Baseline found {len(raw)} executions "
            f"but DRS-simulated path found {len(drs)}.\n"
            f"Only in baseline: {set(raw) - set(drs)}\n"
            f"Only in DRS: {set(drs) - set(raw)}"
        )

    @pytest.mark.parametrize("provider", ALL_PROVIDERS, ids=lambda p: p.slug)
    def test_parity_per_provider(self, complete_executions, lazy_executions, provider):
        """Per-provider check that DRS-simulated -> finalised path matches baseline."""
        baseline = [t for t in _to_tuples(complete_executions) if t[0] == provider.slug]
        drs = [t for t in _to_tuples(lazy_executions) if t[0] == provider.slug]

        assert baseline == drs, (
            f"[{provider.slug}] Baseline found {len(baseline)} "
            f"but DRS-simulated found {len(drs)}.\n"
            f"Only in baseline: {set(baseline) - set(drs)}\n"
            f"Only in DRS: {set(drs) - set(baseline)}"
        )

    def test_drs_dataset_assignments_identical(self, complete_executions, lazy_executions):
        """Dataset assignments must also match when going through the DRS-simulated path."""
        assert len(complete_executions) == len(lazy_executions)

        mismatches = []
        for raw, drs in zip(complete_executions, lazy_executions):
            assert raw["provider"] == drs["provider"]
            assert raw["diagnostic"] == drs["diagnostic"]
            assert raw["dataset_key"] == drs["dataset_key"]

            if raw["datasets"] != drs["datasets"]:
                mismatches.append(
                    f"  {raw['provider']}/{raw['diagnostic']}/{raw['dataset_key']}:\n"
                    f"    raw: {raw['datasets']}\n"
                    f"    drs: {drs['datasets']}"
                )

        assert not mismatches, (
            f"Dataset assignment mismatches found in {len(mismatches)} executions:\n" + "\n".join(mismatches)
        )
