"""Unit tests for ``SeriesIndex.bulk_get_or_create``.

Covers:
1. Happy-path roundtrip: a mix of pre-existing and missing axes resolves to the correct ids.
2. Idempotency: a second call over already-resolved axes inserts nothing.
3. Mismatched-key rejection: a key that is not ``compute_hash(name, values)`` raises ``ValueError``.
"""

import pytest
from sqlalchemy import func, select

from climate_ref.models import SeriesIndex


def _axis_entry(name, values):
    """Build the ``{hash: (name, values)}`` entry ``bulk_get_or_create`` expects."""
    return SeriesIndex.compute_hash(name, values), (name, values)


class TestBulkGetOrCreate:
    def test_empty_input_returns_empty(self, db_seeded):
        assert SeriesIndex.bulk_get_or_create(db_seeded.session, {}) == {}

    def test_roundtrip_mix_of_existing_and_missing(self, db_seeded):
        session = db_seeded.session

        # Pre-create one axis via the single-row path so the bulk call must reuse it.
        existing = SeriesIndex.get_or_create(session, "time", [0, 1, 2])
        session.flush()

        h_existing, e_existing = _axis_entry("time", [0, 1, 2])
        h_missing_a, e_missing_a = _axis_entry("depth", [10, 20])
        h_missing_b, e_missing_b = _axis_entry(None, [1.5, 2.5, 3.5])
        axes_by_hash = {h_existing: e_existing, h_missing_a: e_missing_a, h_missing_b: e_missing_b}

        result = SeriesIndex.bulk_get_or_create(session, axes_by_hash)

        assert set(result) == {h_existing, h_missing_a, h_missing_b}
        # The pre-existing axis keeps its original id (reused, not duplicated).
        assert result[h_existing] == existing.id

        # Every returned id points at a row whose stored hash matches its key.
        for digest, axis_id in result.items():
            axis = session.get(SeriesIndex, axis_id)
            assert axis.hash == digest

        # The two missing axes were inserted with the right content.
        depth = session.get(SeriesIndex, result[h_missing_a])
        assert depth.name == "depth"
        assert depth.values == [10, 20]
        assert depth.length == 2

    def test_idempotent_second_call_inserts_nothing(self, db_seeded):
        session = db_seeded.session

        axes_by_hash = dict([_axis_entry("time", [0, 1, 2]), _axis_entry("depth", [10, 20])])

        first = SeriesIndex.bulk_get_or_create(session, axes_by_hash)
        session.flush()
        count_after_first = session.execute(select(func.count()).select_from(SeriesIndex)).scalar_one()

        second = SeriesIndex.bulk_get_or_create(session, axes_by_hash)
        session.flush()
        count_after_second = session.execute(select(func.count()).select_from(SeriesIndex)).scalar_one()

        assert second == first
        assert count_after_second == count_after_first

    def test_mismatched_key_is_rejected(self, db_seeded):
        session = db_seeded.session

        # A key that does not equal compute_hash(name, values) for its axis.
        bad = {"not-the-real-hash": ("time", [0, 1, 2])}

        with pytest.raises(ValueError, match="does not match compute_hash"):
            SeriesIndex.bulk_get_or_create(session, bad)

        # Nothing was inserted for the rejected batch.
        count = session.execute(select(func.count()).select_from(SeriesIndex)).scalar_one()
        assert count == 0
