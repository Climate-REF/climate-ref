"""
Regression tests for Climate-REF/climate-ref#839.

`json.dumps` writes the bare tokens `NaN`, `Infinity` and `-Infinity` by default.
Python reads them back, so SQLite never complains,
but PostgreSQL validates the value on insert and rejects the whole row.
The execution then ends up with 0 metric values despite the diagnostic having succeeded.
"""

import json

import numpy as np
import pytest

from climate_ref.models import SeriesIndex, SeriesMetricValue
from climate_ref.models.base import replace_non_finite

NAN = float("nan")
INF = float("inf")


class TestReplaceNonFinite:
    @pytest.mark.parametrize("value", [NAN, INF, -INF])
    def test_non_finite_becomes_none(self, value):
        assert replace_non_finite(value) is None

    @pytest.mark.parametrize("value", [0.0, 1.5, -2.5, 1e300])
    def test_finite_floats_are_untouched(self, value):
        assert replace_non_finite(value) == value

    @pytest.mark.parametrize("value", [None, True, 3, "NaN", "a string"])
    def test_other_types_are_untouched(self, value):
        assert replace_non_finite(value) is value

    @pytest.mark.parametrize("dtype", ["float16", "float32", "float64"])
    def test_numpy_non_finite_becomes_none(self, dtype):
        assert replace_non_finite(np.dtype(dtype).type(NAN)) is None
        assert replace_non_finite(np.dtype(dtype).type(INF)) is None
        assert replace_non_finite(np.dtype(dtype).type(1.5)) == 1.5

    def test_recurses_into_lists(self):
        assert replace_non_finite([1.0, NAN, [INF, 2.0]]) == [1.0, None, [None, 2.0]]

    def test_recurses_into_dicts(self):
        assert replace_non_finite({"a": NAN, "b": {"c": -INF}}) == {"a": None, "b": {"c": None}}

    def test_series_keeps_its_length(self):
        """A shorter series would silently misalign against its index axis."""
        values = [1.0, NAN, 3.0, INF, 5.0]

        assert len(replace_non_finite(values)) == len(values)

    def test_result_is_strict_json(self):
        def reject(name):
            raise AssertionError(f"bare {name} survived")

        body = json.dumps(replace_non_finite([NAN, INF, -INF]))

        assert json.loads(body, parse_constant=reject) == [None, None, None]


class TestNonFiniteReachesTheDatabaseAsNull:
    def test_series_values_with_nan_are_stored_as_null(self, db_seeded):
        session = db_seeded.session

        axis = SeriesIndex.get_or_create(session, "time", [0, 1, 2])
        session.flush()

        value = SeriesMetricValue.build(
            execution_id=1,
            values=[1.0, NAN, 3.0],
            index_axis=axis,
            dimensions={},
            attributes=None,
        )
        session.add(value)
        session.flush()
        session.expire(value)

        assert value.values == [1.0, None, 3.0]

    def test_infinity_in_attributes_is_stored_as_null(self, db_seeded):
        session = db_seeded.session

        axis = SeriesIndex.get_or_create(session, "time", [0, 1])
        session.flush()

        value = SeriesMetricValue.build(
            execution_id=1,
            values=[1.0, 2.0],
            index_axis=axis,
            dimensions={},
            attributes={"threshold": INF, "nested": {"floor": -INF}},
        )
        session.add(value)
        session.flush()
        session.expire(value)

        assert value.attributes == {"threshold": None, "nested": {"floor": None}}

    def test_none_attributes_stay_a_json_null(self, db_seeded):
        """
        `attributes` is not nullable,
        and relies on JSON storing a Python None as the JSON literal null rather than SQL NULL.
        A TypeDecorator does not inherit that behaviour on its own.
        """
        session = db_seeded.session

        axis = SeriesIndex.get_or_create(session, "time", [0, 1])
        session.flush()

        value = SeriesMetricValue.build(
            execution_id=1,
            values=[1.0, 2.0],
            index_axis=axis,
            dimensions={},
            attributes=None,
        )
        session.add(value)
        session.flush()
        session.expire(value)

        assert value.attributes is None

    def test_index_axis_values_with_nan_are_stored_as_null(self, db_seeded):
        session = db_seeded.session

        axis = SeriesIndex.get_or_create(session, "depth", [0.0, NAN, 2.0])
        session.flush()
        session.expire(axis)

        assert axis.values == [0.0, None, 2.0]

    def test_hash_still_distinguishes_nan_from_none(self, db_seeded):
        """
        The content hash is deliberately left alone.

        Its serialisation is relied on by the series-index migration backfill, so it must stay stable.
        Two axes that differ only in NaN versus None
        therefore remain separate rows that happen to store the same values.
        """
        assert SeriesIndex.compute_hash("t", [1.0, NAN]) != SeriesIndex.compute_hash("t", [1.0, None])

    def test_a_wholly_non_finite_series_still_stores(self, db_seeded):
        """The failing case in #839 was a series whose first element was NaN."""
        session = db_seeded.session

        axis = SeriesIndex.get_or_create(session, "time", [0, 1, 2])
        session.flush()

        value = SeriesMetricValue.build(
            execution_id=1,
            values=[NAN, NAN, NAN],
            index_axis=axis,
            dimensions={},
            attributes=None,
        )
        session.add(value)
        session.flush()
        session.expire(value)

        assert value.values == [None, None, None]
