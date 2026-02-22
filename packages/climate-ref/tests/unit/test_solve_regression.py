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
from climate_ref_example import provider as example_provider

from climate_ref.solve_helpers import solve_results_for_regression, solve_to_results


@pytest.fixture(scope="module")
def example_results(esgf_data_catalog):
    return solve_to_results(esgf_data_catalog, providers=[example_provider])


@pytest.mark.parametrize(
    "diagnostic_slug",
    [d.slug for d in example_provider.diagnostics()],
)
def test_solve_regression(example_results, data_regression, diagnostic_slug):
    filtered = [r for r in example_results if r["diagnostic"] == diagnostic_slug]
    regression = solve_results_for_regression(filtered)
    data_regression.check(regression)
