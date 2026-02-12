"""
Regression tests for the solver using parquet catalogs.

These tests use pre-generated parquet catalogs containing ESGF dataset metadata
(without actual data files) to verify that the solver produces consistent results.

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


class TestSolveRegressionExample:
    """Regression test using the example provider."""

    def test_example_provider(self, esgf_data_catalog, data_regression):
        results = solve_to_results(esgf_data_catalog, providers=[example_provider])
        regression = solve_results_for_regression(results)
        data_regression.check(regression)


@pytest.mark.parametrize(
    "provider_slug",
    ["pmp", "esmvaltool", "ilamb"],
)
def test_solve_regression_per_provider(esgf_data_catalog, data_regression, provider_slug):
    """Test each provider independently for finer-grained regression tracking."""
    provider_map = {
        "pmp": pmp_provider,
        "esmvaltool": esmvaltool_provider,
        "ilamb": ilamb_provider,
    }

    provider = provider_map[provider_slug]
    results = solve_to_results(esgf_data_catalog, providers=[provider])
    regression = solve_results_for_regression(results)
    data_regression.check(regression)
