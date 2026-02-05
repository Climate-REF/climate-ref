"""
Regression tests for the solver using parquet catalogs.

These tests use pre-generated parquet catalogs containing ESGF dataset metadata
(without actual data files) to verify that the solver produces consistent results.

Tests are skipped if no parquet catalog is available.
Generate catalogs with::

    python scripts/generate_esgf_catalog.py \\
        --cmip6-dir /path/to/CMIP6 \\
        --obs4mips-dir /path/to/obs4MIPs \\
        --output-dir tests/test-data/esgf-catalog/
"""

from __future__ import annotations

import pytest
from climate_ref_esmvaltool import provider as esmvaltool_provider
from climate_ref_example import provider as example_provider
from climate_ref_ilamb import provider as ilamb_provider
from climate_ref_pmp import provider as pmp_provider

from climate_ref.solve_helpers import solve_results_for_regression, solve_to_results


@pytest.fixture
def solve_catalog_or_skip(esgf_solve_catalog):
    """Wrap esgf_solve_catalog, skipping if no catalog is present."""
    if esgf_solve_catalog is None:
        pytest.skip("No ESGF parquet catalog available (run scripts/generate_esgf_catalog.py)")
    return esgf_solve_catalog


class TestSolveRegressionExample:
    """Regression test using the example provider."""

    def test_example_provider(self, solve_catalog_or_skip, data_regression):
        results = solve_to_results(solve_catalog_or_skip, providers=[example_provider])
        regression = solve_results_for_regression(results)
        data_regression.check(regression)


class TestSolveRegressionAllProviders:
    """Regression test using all available providers."""

    def test_all_providers(self, solve_catalog_or_skip, data_regression):
        results = solve_to_results(
            solve_catalog_or_skip,
            providers=[pmp_provider, esmvaltool_provider, ilamb_provider],
        )
        regression = solve_results_for_regression(results)
        data_regression.check(regression)


@pytest.mark.parametrize(
    "provider_slug",
    ["pmp", "esmvaltool", "ilamb"],
)
def test_solve_regression_per_provider(solve_catalog_or_skip, data_regression, provider_slug):
    """Test each provider independently for finer-grained regression tracking."""
    provider_map = {
        "pmp": pmp_provider,
        "esmvaltool": esmvaltool_provider,
        "ilamb": ilamb_provider,
    }

    provider = provider_map[provider_slug]
    results = solve_to_results(solve_catalog_or_skip, providers=[provider])
    regression = solve_results_for_regression(results)
    data_regression.check(regression)
