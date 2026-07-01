"""Unit tests for the metric-value data access layer (spike)."""

import math

import pytest

from climate_ref.models import Execution, ExecutionGroup
from climate_ref.models.diagnostic import Diagnostic
from climate_ref.models.metric_value import ScalarMetricValue, SeriesIndex, SeriesMetricValue
from climate_ref.results import (
    MetricValueFilter,
    OutlierPolicy,
    Reader,
)
from climate_ref.results._query import latest_execution_for_group


@pytest.fixture
def dal_db(db_seeded):
    """A seeded DB with one execution group, execution, and scalar + series values."""
    session = db_seeded.session

    with session.begin():
        diagnostic = session.query(Diagnostic).first()
        assert diagnostic is not None, "example provider should seed at least one diagnostic"
        group = ExecutionGroup(key="dal-key", diagnostic_id=diagnostic.id, selectors={})
        session.add(group)
        session.flush()
        execution = Execution(
            execution_group_id=group.id,
            output_fragment="frag",
            dataset_hash="hash-dal",
            successful=True,
        )
        session.add(execution)
        session.flush()

        # A well-behaved cluster of five models, one wild outlier, a NaN, and a Reference value.
        base_sources = {
            "ACCESS-CM2": 10.0,
            "CESM3": 10.5,
            "MIROC6": 9.5,
            "MPI-ESM": 10.2,
            "NorESM": 9.8,
        }
        for source_id, value in base_sources.items():
            session.add(
                ScalarMetricValue.build(
                    execution_id=execution.id,
                    value=value,
                    attributes=None,
                    dimensions={"statistic": "mean", "metric": "tas", "source_id": source_id},
                )
            )
        session.add(
            ScalarMetricValue.build(
                execution_id=execution.id,
                value=1000.0,  # wild outlier
                attributes=None,
                dimensions={"statistic": "mean", "metric": "tas", "source_id": "WILD"},
            )
        )
        session.add(
            ScalarMetricValue.build(
                execution_id=execution.id,
                value=math.nan,  # non-finite -> always flagged (own group so it can't poison the tas IQR)
                attributes=None,
                dimensions={"statistic": "mean", "metric": "broken", "source_id": "BROKEN"},
            )
        )
        session.add(
            ScalarMetricValue.build(
                execution_id=execution.id,
                value=99999.0,  # Reference -> never flagged
                attributes=None,
                dimensions={"statistic": "mean", "metric": "tas", "source_id": "Reference"},
            )
        )

        axis = SeriesIndex.get_or_create(session, "time", [0, 1, 2])
        session.add(
            SeriesMetricValue.build(
                execution_id=execution.id,
                values=[1.0, 2.0, 3.0],
                index_axis=axis,
                dimensions={"source_id": "ACCESS-CM2", "metric": "tas"},
                attributes=None,
            )
        )
        ref_series = SeriesMetricValue.build(
            execution_id=execution.id,
            values=[1.1, 2.1, 3.1],
            index_axis=axis,
            dimensions={"source_id": "Reference", "metric": "tas"},
            attributes=None,
        )
        ref_series.reference_id = "ref-hash-1"
        session.add(ref_series)
        session.flush()
        group_id = group.id  # capture inside the txn; reading it post-commit would autobegin

    db_seeded.execution_group_id = group_id
    return db_seeded


class TestScalarValues:
    def test_returns_all_without_outlier_detection(self, dal_db):
        ref = Reader(dal_db)
        coll = ref.values.scalar_values(MetricValueFilter(execution_group_ids=[dal_db.execution_group_id]))
        assert coll.total_count == 8  # 5 base + wild + nan + reference
        assert len(coll) == 8
        assert coll.had_outliers is False
        assert coll.outlier_count == 0
        # execution_group_id is projected onto every row
        assert all(v.execution_group_id == dal_db.execution_group_id for v in coll)

    def test_outlier_detection_removes_flagged(self, dal_db):
        ref = Reader(dal_db)
        coll = ref.values.scalar_values(
            MetricValueFilter(execution_group_ids=[dal_db.execution_group_id]),
            outliers=OutlierPolicy(),
            include_unverified=False,
        )
        # WILD and BROKEN (NaN) removed; Reference retained.
        assert coll.had_outliers is True
        assert coll.outlier_count >= 2
        assert coll.total_count == 8 - coll.outlier_count
        source_ids = {v.dimensions["source_id"] for v in coll}
        assert "WILD" not in source_ids
        assert "BROKEN" not in source_ids
        assert "Reference" in source_ids

    def test_include_unverified_keeps_flagged_with_annotation(self, dal_db):
        ref = Reader(dal_db)
        coll = ref.values.scalar_values(
            MetricValueFilter(execution_group_ids=[dal_db.execution_group_id]),
            outliers=OutlierPolicy(),
            include_unverified=True,
        )
        assert coll.total_count == 8
        flagged = {v.dimensions["source_id"] for v in coll if v.is_outlier}
        assert "WILD" in flagged
        assert "BROKEN" in flagged
        # verification_status mirrors the flag
        assert all((v.verification_status == "unverified") == bool(v.is_outlier) for v in coll)

    def test_to_pandas_columns(self, dal_db):
        ref = Reader(dal_db)
        coll = ref.values.scalar_values(MetricValueFilter(execution_group_ids=[dal_db.execution_group_id]))
        df = coll.to_pandas()
        expected_cols = (
            "source_id",
            "statistic",
            "metric",
            "value",
            "id",
            "execution_id",
            "execution_group_id",
            "kind",
        )
        for col in expected_cols:
            assert col in df.columns
        assert len(df) == 8

    def test_facets(self, dal_db):
        ref = Reader(dal_db)
        coll = ref.values.scalar_values(MetricValueFilter(execution_group_ids=[dal_db.execution_group_id]))
        facets = coll.facets_dict()
        assert "source_id" in facets
        assert "WILD" in facets["source_id"]
        assert "metric" in facets

    def test_pagination(self, dal_db):
        ref = Reader(dal_db)
        coll = ref.values.scalar_values(
            MetricValueFilter(execution_group_ids=[dal_db.execution_group_id]),
            limit=3,
            offset=0,
        )
        assert len(coll) == 3
        assert coll.total_count == 8

    def test_dimension_filter(self, dal_db):
        ref = Reader(dal_db)
        coll = ref.values.scalar_values(
            MetricValueFilter(
                execution_group_ids=[dal_db.execution_group_id],
                dimensions={"source_id": ["ACCESS-CM2", "CESM3"]},
            )
        )
        assert coll.total_count == 2
        assert {v.dimensions["source_id"] for v in coll} == {"ACCESS-CM2", "CESM3"}

    def test_unknown_dimension_raises(self, dal_db):
        ref = Reader(dal_db)
        with pytest.raises(KeyError, match="not_a_dim"):
            ref.values.scalar_values(MetricValueFilter(dimensions={"not_a_dim": "x"}))

    def test_include_context_adds_slugs(self, dal_db):
        ref = Reader(dal_db)
        coll = ref.values.scalar_values(
            MetricValueFilter(execution_group_ids=[dal_db.execution_group_id]),
            include_context=True,
            limit=1,
        )
        v = coll.items[0]
        assert v.diagnostic_slug is not None
        assert v.provider_slug is not None

    def test_detached_after_session_expunge(self, dal_db):
        ref = Reader(dal_db)
        coll = ref.values.scalar_values(MetricValueFilter(execution_group_ids=[dal_db.execution_group_id]))
        dal_db.session.expunge_all()
        # DTOs are detached; building a frame must not touch the ORM/session.
        df = coll.to_pandas()
        assert len(df) == 8


class TestSeriesValues:
    def test_returns_series_with_resolved_index(self, dal_db):
        ref = Reader(dal_db)
        coll = ref.values.series_values(MetricValueFilter(execution_group_ids=[dal_db.execution_group_id]))
        assert coll.total_count == 2
        v = coll.items[0]
        assert v.index == [0, 1, 2]
        assert v.index_name == "time"
        assert v.execution_group_id == dal_db.execution_group_id

    def test_reference_only_filter(self, dal_db):
        ref = Reader(dal_db)
        refs = ref.values.series_values(
            MetricValueFilter(execution_group_ids=[dal_db.execution_group_id], reference_only=True)
        )
        assert refs.total_count == 1
        assert refs.items[0].is_reference
        assert refs.items[0].reference_id == "ref-hash-1"

        models = ref.values.series_values(
            MetricValueFilter(execution_group_ids=[dal_db.execution_group_id], reference_only=False)
        )
        assert models.total_count == 1
        assert not models.items[0].is_reference

    def test_to_pandas_long_form(self, dal_db):
        ref = Reader(dal_db)
        coll = ref.values.series_values(MetricValueFilter(execution_group_ids=[dal_db.execution_group_id]))
        df = coll.to_pandas(explode=True)
        # 2 series x 3 points
        assert len(df) == 6
        for col in ("value", "index", "index_name", "source_id", "reference_id"):
            assert col in df.columns


class TestLatestExecutionForGroup:
    def test_returns_most_recent(self, dal_db):
        session = dal_db.session
        group_id = dal_db.execution_group_id
        with session.begin():
            newer = Execution(
                execution_group_id=group_id,
                output_fragment="frag2",
                dataset_hash="hash-dal-2",
                successful=True,
            )
            session.add(newer)
            session.flush()
            newer_id = newer.id
        latest = latest_execution_for_group(session, group_id)
        assert latest is not None
        assert latest.id == newer_id
