"""
Regression test for ILAMB solver output.

Uses pre-generated parquet catalogs to verify solver produces consistent results.
"""

from __future__ import annotations

import pytest
from climate_ref_ilamb import provider as ilamb_provider

from climate_ref.solve_helpers import solve_results_for_regression, solve_to_results

# Diagnostics that legitimately solve to nothing because their reference data is
# intentionally absent from the solve catalog fixture (tests/test-data/esgf-catalog/).
# Every other diagnostic must match at least one execution group, otherwise an empty
# snapshot would pass while asserting no solver output.
KNOWN_NO_MATCH = frozenset({"emp-gleamgpcp2.3"})


@pytest.fixture(scope="module")
def ilamb_results(esgf_data_catalog, solve_config):
    ilamb_provider.configure(solve_config)
    return solve_to_results(esgf_data_catalog, providers=[ilamb_provider])


@pytest.mark.parametrize(
    "diagnostic_slug",
    [d.slug for d in ilamb_provider.diagnostics()],
)
def test_solve_regression(ilamb_results, data_regression, diagnostic_slug):
    filtered = [r for r in ilamb_results if r["diagnostic"] == diagnostic_slug]
    regression = solve_results_for_regression(filtered)
    if diagnostic_slug not in KNOWN_NO_MATCH:
        assert regression, (
            f"{diagnostic_slug} solved to no execution groups. Its reference data is "
            "likely missing from the solve catalog fixture (tests/test-data/esgf-catalog/). "
            "Add the reference dataset, or add the slug to KNOWN_NO_MATCH if this is intended."
        )
    data_regression.check(regression)
