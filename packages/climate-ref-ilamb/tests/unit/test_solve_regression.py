"""
Regression test for ILAMB solver output.

Uses pre-generated parquet catalogs to verify solver produces consistent results.
"""

from __future__ import annotations

import pytest
from climate_ref_ilamb import provider as ilamb_provider

from climate_ref.config import Config
from climate_ref.solve_helpers import solve_results_for_regression, solve_to_results


@pytest.fixture(scope="module")
def ilamb_results(esgf_data_catalog):
    ilamb_provider.configure(Config.default())
    return solve_to_results(esgf_data_catalog, providers=[ilamb_provider])


@pytest.mark.parametrize(
    "diagnostic_slug",
    [d.slug for d in ilamb_provider.diagnostics()],
)
def test_solve_regression(ilamb_results, data_regression, diagnostic_slug):
    filtered = [r for r in ilamb_results if r["diagnostic"] == diagnostic_slug]
    regression = solve_results_for_regression(filtered)
    data_regression.check(regression)
