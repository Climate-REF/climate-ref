"""
Contract tests over the committed ESMValTool ``series.json`` regression baselines.

These assert two invariants hold across every committed baseline:

- the role of each series is carried by the ``kind`` flag, derived from the presence of
  ``reference_source_id`` in the series dimensions;
- the presentation-attribute key set is uniform across all series, modulo a small set of
  legitimately-absent fields (a ``calendar`` only exists for a temporal index, a
  ``standard_name`` only when the source variable declares one, and so on).
"""

from pathlib import Path

import pytest

from climate_ref_core.metric_values import SeriesMetricValue

TEST_DATA = Path(__file__).resolve().parents[1] / "test-data"

# Always present on every committed series, regardless of the underlying variable or index.
REQUIRED_ATTRS = frozenset({"caption", "index_name", "value_long_name", "value_units", "values_name"})
# Present only when the underlying data carries them (a temporal index has a calendar; a
# spatial index has units; some variables declare a CF ``standard_name``, others do not).
OPTIONAL_ATTRS = frozenset(
    {"calendar", "index_long_name", "index_standard_name", "index_units", "value_standard_name"}
)

SERIES_FILES = sorted(TEST_DATA.glob("**/regression/series.json"))


def _case_id(path: Path) -> str:
    return str(path.relative_to(TEST_DATA).parent.parent)


@pytest.mark.parametrize("series_file", SERIES_FILES, ids=[_case_id(p) for p in SERIES_FILES])
def test_committed_series_role_and_attrs(series_file):
    """Every committed series carries the correct ``kind`` and a uniform attribute key set."""
    # An empty series.json is legitimate: some cases emit only scalars and plots, no series.
    for s in SeriesMetricValue.load_from_json(series_file):
        is_reference = "reference_source_id" in s.dimensions
        expected_kind = "reference" if is_reference else "model"
        assert s.kind == expected_kind, (
            f"{_case_id(series_file)}: series {s.dimensions} has kind={s.kind!r}, expected {expected_kind!r}"
        )

        attr_keys = frozenset(s.attributes or {})
        missing = REQUIRED_ATTRS - attr_keys
        assert not missing, f"{_case_id(series_file)}: series {s.dimensions} missing attrs {sorted(missing)}"
        unexpected = attr_keys - REQUIRED_ATTRS - OPTIONAL_ATTRS
        assert not unexpected, (
            f"{_case_id(series_file)}: series {s.dimensions} has unexpected attrs {sorted(unexpected)}"
        )


def test_some_reference_series_exist():
    """Guard the guard: at least one committed baseline exercises the reference branch."""
    seen_reference = any(
        s.kind == "reference"
        for series_file in SERIES_FILES
        for s in SeriesMetricValue.load_from_json(series_file)
    )
    assert seen_reference, "no reference series found in any committed baseline; the kind test is vacuous"
