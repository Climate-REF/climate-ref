"""
Regression test for ESMValTool solver output.

Uses pre-generated parquet catalogs to verify solver produces consistent results.
"""

from __future__ import annotations

import pytest
from climate_ref_esmvaltool import provider as esmvaltool_provider

from climate_ref.config import Config
from climate_ref.solve_helpers import solve_results_for_regression, solve_to_results


@pytest.fixture(scope="module")
def esmvaltool_results(esgf_data_catalog):
    esmvaltool_provider.configure(Config.default())
    return solve_to_results(esgf_data_catalog, providers=[esmvaltool_provider])


@pytest.mark.parametrize(
    "diagnostic_slug",
    [d.slug for d in esmvaltool_provider.diagnostics()],
)
def test_solve_regression(esmvaltool_results, data_regression, diagnostic_slug):
    filtered = [r for r in esmvaltool_results if r["diagnostic"] == diagnostic_slug]
    regression = solve_results_for_regression(filtered)
    data_regression.check(regression)
