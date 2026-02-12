"""
Regression test for PMP solver output.

Uses pre-generated parquet catalogs to verify solver produces consistent results.
"""

from __future__ import annotations

import pytest
from climate_ref_pmp import provider as pmp_provider

from climate_ref.solve_helpers import solve_results_for_regression, solve_to_results


@pytest.fixture(scope="module")
def pmp_results(esgf_data_catalog):
    return solve_to_results(esgf_data_catalog, providers=[pmp_provider])


@pytest.mark.parametrize(
    "diagnostic_slug",
    [d.slug for d in pmp_provider.diagnostics()],
)
def test_solve_regression(pmp_results, data_regression, diagnostic_slug):
    filtered = [r for r in pmp_results if r["diagnostic"] == diagnostic_slug]
    regression = solve_results_for_regression(filtered)
    data_regression.check(regression)
