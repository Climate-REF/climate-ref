"""Unit tests for the metric-value data access layer (spike)."""

import datetime
import math

import attrs
import pytest
from pandas.testing import assert_frame_equal as pd_assert_frame_equal
from sqlalchemy import distinct

from climate_ref.models import Execution, ExecutionGroup
from climate_ref.models import execution as models_execution
from climate_ref.models.diagnostic import Diagnostic
from climate_ref.models.metric_value import MetricValueType, ScalarMetricValue, SeriesIndex, SeriesMetricValue
from climate_ref.results import (
    MetricValueFilter,
    OutlierPolicy,
    Reader,
)
from climate_ref.results._query import (
    latest_execution_for_group,
    select_latest_executions,
    select_scalar_values,
    select_series_values,
)
from climate_ref.results.frames import collect_facets, scalar_values_to_frame, series_values_to_frame
from climate_ref.results.outliers import detect_scalar_outliers


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

    def test_pagination_deterministic_across_tied_timestamps(self, dal_db):
        # Every scalar row here shares one execution (one created_at), so ordering must fall back to
        # the value id. Two adjacent pages must partition the result with no overlap and no gap.
        f = MetricValueFilter(execution_group_ids=[dal_db.execution_group_id])
        ref = Reader(dal_db)
        page_1 = ref.values.scalar_values(f, offset=0, limit=4)
        page_2 = ref.values.scalar_values(f, offset=4, limit=4)
        ids_1 = [v.id for v in page_1]
        ids_2 = [v.id for v in page_2]
        assert ids_1 == sorted(ids_1)
        assert ids_2 == sorted(ids_2)
        assert set(ids_1).isdisjoint(ids_2)  # no overlap
        assert ids_1 + ids_2 == sorted(v.id for v in ref.values.scalar_values(f))  # no gap

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


class TestWithCount:
    def test_scalar_without_count(self, dal_db, count_queries):
        values = Reader(dal_db).values
        filters = MetricValueFilter(execution_group_ids=[dal_db.execution_group_id])
        with count_queries(dal_db) as counted:
            full = values.scalar_values(filters, limit=3, with_facets=False)
        with count_queries(dal_db) as uncounted:
            page = values.scalar_values(filters, limit=3, with_facets=False, with_count=False)

        assert page.total_count is None
        assert page.items == full.items
        assert len(uncounted) == len(counted) - 1

    def test_scalar_outliers_without_count(self, dal_db):
        values = Reader(dal_db).values
        filters = MetricValueFilter(execution_group_ids=[dal_db.execution_group_id])
        policy = OutlierPolicy(method="iqr")
        full = values.scalar_values(filters, outliers=policy)
        page = values.scalar_values(filters, outliers=policy, with_count=False)

        assert page.total_count is None
        assert page.items == full.items

    def test_series_without_count(self, dal_db, count_queries):
        values = Reader(dal_db).values
        filters = MetricValueFilter(execution_group_ids=[dal_db.execution_group_id])
        with count_queries(dal_db) as counted:
            full = values.series_values(filters, limit=1, with_facets=False)
        with count_queries(dal_db) as uncounted:
            page = values.series_values(filters, limit=1, with_facets=False, with_count=False)

        assert page.total_count is None
        assert page.items == full.items
        assert len(uncounted) == len(counted) - 1


class TestFrameParity:
    """Both frame doors -- the frames.py builders and the collections' ``to_pandas()`` -- must emit
    identical columns and rows for the same DTOs, so no consumer sees a different scalar/series
    frame shape depending on which door it came through."""

    def test_scalar_columns_identical(self, dal_db):
        coll = Reader(dal_db).values.scalar_values(
            MetricValueFilter(execution_group_ids=[dal_db.execution_group_id])
        )
        via_collection = coll.to_pandas()
        via_helper = scalar_values_to_frame(coll.items)
        assert list(via_collection.columns) == list(via_helper.columns)
        assert "kind" in via_helper.columns  # kind promoted out of the dimension columns
        assert "type" not in via_helper.columns  # no legacy `type` marker column
        pd_assert_frame_equal(via_collection, via_helper)

    def test_series_columns_identical(self, dal_db):
        coll = Reader(dal_db).values.series_values(
            MetricValueFilter(execution_group_ids=[dal_db.execution_group_id])
        )
        for explode in (True, False):
            via_collection = coll.to_pandas(explode=explode)
            via_helper = series_values_to_frame(coll.items, explode=explode)
            assert list(via_collection.columns) == list(via_helper.columns)
            assert "kind" in via_helper.columns
            assert "execution_group_id" in via_helper.columns
            pd_assert_frame_equal(via_collection, via_helper)


class TestKindSeparation:
    def test_kind_promoted_out_of_dimensions(self, db_seeded):
        # kind is a registered CV dimension, so it has its own column; the read layer promotes it
        # to the dedicated `kind` field and must not leave it duplicated inside `dimensions`.
        session = db_seeded.session
        with session.begin():
            diagnostic = session.query(Diagnostic).first()
            group = ExecutionGroup(key="kind-key", diagnostic_id=diagnostic.id, selectors={})
            session.add(group)
            session.flush()
            execution = Execution(
                execution_group_id=group.id,
                output_fragment="frag",
                dataset_hash="hash-kind",
                successful=True,
            )
            session.add(execution)
            session.flush()
            session.add(
                ScalarMetricValue.build(
                    execution_id=execution.id,
                    value=1.0,
                    attributes=None,
                    dimensions={"metric": "tas", "source_id": "ACCESS-CM2", "kind": "reference"},
                )
            )
            group_id = group.id
        db_seeded.execution_group_id = group_id

        coll = Reader(db_seeded).values.scalar_values(MetricValueFilter(execution_group_ids=[group_id]))
        value = coll.items[0]
        assert value.kind == "reference"
        assert "kind" not in value.dimensions
        assert value.dimensions["source_id"] == "ACCESS-CM2"
        # kind is still surfaced as a frame column (promoted explicitly, not via dimensions).
        assert "kind" in coll.to_pandas().columns


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


def _add_execution(session, group_id, fragment, created_at):
    execution = Execution(
        execution_group_id=group_id,
        output_fragment=fragment,
        dataset_hash=f"hash-{fragment}",
        successful=True,
        created_at=created_at,
    )
    session.add(execution)
    session.flush()
    return execution.id


class TestSelectLatestExecutions:
    def test_newest_created_at_wins_over_higher_id(self, dal_db):
        session = dal_db.session
        group_id = dal_db.execution_group_id
        with session.begin():
            newest = _add_execution(session, group_id, "newest", datetime.datetime(2030, 1, 1))
            _add_execution(session, group_id, "older", datetime.datetime(2029, 1, 1))
        rows = session.execute(select_latest_executions()).all()
        assert dict(rows)[group_id] == newest

    def test_tied_created_at_picks_highest_id(self, dal_db):
        session = dal_db.session
        group_id = dal_db.execution_group_id
        tied_at = datetime.datetime(2030, 1, 1)
        with session.begin():
            ids = [_add_execution(session, group_id, f"tie-{i}", tied_at) for i in range(2)]
        rows = session.execute(select_latest_executions()).all()
        assert [r.execution_id for r in rows if r.execution_group_id == group_id] == [max(ids)]

    def test_one_row_per_group(self, dal_db):
        session = dal_db.session
        rows = session.execute(select_latest_executions()).all()
        group_ids = [r.execution_group_id for r in rows]
        assert len(group_ids) == len(set(group_ids))
        assert dal_db.execution_group_id in group_ids

    def test_among_executions_picks_newest_matching(self, dal_db):
        session = dal_db.session
        group_id = dal_db.execution_group_id
        with session.begin():
            older_success = _add_execution(session, group_id, "older-ok", datetime.datetime(2030, 1, 1))
            newer = _add_execution(session, group_id, "newer-failed", datetime.datetime(2031, 1, 1))
            session.get(Execution, newer).successful = False
        all_rows = dict(session.execute(select_latest_executions()).all())
        successful_rows = dict(
            session.execute(select_latest_executions([Execution.successful.is_(True)])).all()
        )
        assert all_rows[group_id] == newer
        assert successful_rows[group_id] == older_success

    def test_each_group_and_version_gets_its_own_latest(self, dal_db):
        session = dal_db.session
        with session.begin():
            diagnostic_id = session.get(ExecutionGroup, dal_db.execution_group_id).diagnostic_id
            groups = {}
            for key, version in [("multi", 1), ("multi", 2), ("other", 1)]:
                group = ExecutionGroup(
                    key=key, diagnostic_id=diagnostic_id, diagnostic_version=version, selectors={}
                )
                session.add(group)
                session.flush()
                groups[key, version] = group.id
            # Interleaved so the newest execution overall belongs to one group only.
            expected = {}
            for (key, version), group_id in groups.items():
                for year in (2030, 2032, 2031):
                    execution_id = _add_execution(
                        session,
                        group_id,
                        f"{key}-v{version}-{year}",
                        datetime.datetime(year, 1, 1, version),
                    )
                    if year == 2032:
                        expected[group_id] = execution_id
        rows = dict(session.execute(select_latest_executions()).all())
        assert {group_id: rows[group_id] for group_id in groups.values()} == expected

    def test_latest_only_follows_the_promoted_version(self, dal_db):
        session = dal_db.session
        with session.begin():
            diagnostic = session.get(ExecutionGroup, dal_db.execution_group_id).diagnostic
            latest = {}
            for version in (1, 2):
                group = ExecutionGroup(
                    key="versioned", diagnostic_id=diagnostic.id, diagnostic_version=version, selectors={}
                )
                session.add(group)
                session.flush()
                for year in (2030, 2031):
                    execution_id = _add_execution(
                        session, group.id, f"versioned-v{version}-{year}", datetime.datetime(year, 1, 1)
                    )
                    session.add(
                        ScalarMetricValue.build(
                            execution_id=execution_id,
                            value=float(version),
                            attributes=None,
                            dimensions={"statistic": "mean", "metric": "tas", "source_id": "ACCESS-CM2"},
                        )
                    )
                latest[version] = execution_id
            diagnostic.promoted_version = 2

        values = Reader(dal_db).values
        promoted = values.scalar_values(MetricValueFilter(latest_only=True))
        assert {v.execution_id for v in promoted.items} == {latest[2]}
        every_version = values.scalar_values(MetricValueFilter(latest_only=True, include_superseded=True))
        assert {latest[1], latest[2]} <= {v.execution_id for v in every_version.items}

    def test_among_executions_drops_groups_without_a_match(self, dal_db):
        session = dal_db.session
        rows = session.execute(select_latest_executions([Execution.output_fragment == "no-such"])).all()
        assert rows == []

    def test_results_re_exports_the_models_builder(self):
        assert select_latest_executions is models_execution.select_latest_executions


class TestLatestOnly:
    def test_drops_values_from_superseded_execution(self, dal_db):
        session = dal_db.session
        group_id = dal_db.execution_group_id
        with session.begin():
            newer = _add_execution(session, group_id, "newer", datetime.datetime(2030, 1, 1))
            session.add(
                ScalarMetricValue.build(
                    execution_id=newer,
                    value=1.0,
                    attributes=None,
                    dimensions={"statistic": "mean", "metric": "tas", "source_id": "ACCESS-CM2"},
                )
            )
        reader = Reader(dal_db)
        latest = reader.values.scalar_values(MetricValueFilter(latest_only=True))
        assert {v.execution_id for v in latest.items} == {newer}
        assert len(latest) == 1
        everything = reader.values.scalar_values(MetricValueFilter())
        assert len(everything) == 9

    def test_keeps_all_values_of_sole_execution(self, dal_db):
        reader = Reader(dal_db)
        assert len(reader.values.scalar_values(MetricValueFilter(latest_only=True))) == len(
            reader.values.scalar_values(MetricValueFilter())
        )
        assert len(reader.values.series_values(MetricValueFilter(latest_only=True))) == 2


class TestRetracted:
    @pytest.fixture
    def retracted_latest(self, dal_db):
        session = dal_db.session
        with session.begin():
            execution_id = _add_execution(
                session, dal_db.execution_group_id, "retracted", datetime.datetime(2030, 1, 1)
            )
            session.get(Execution, execution_id).retracted = True
            session.add(
                ScalarMetricValue.build(
                    execution_id=execution_id,
                    value=1.0,
                    attributes=None,
                    dimensions={"statistic": "mean", "metric": "tas", "source_id": "ACCESS-CM2"},
                )
            )
        return execution_id

    def test_excluded_by_default(self, dal_db, retracted_latest):
        values = Reader(dal_db).values.scalar_values(MetricValueFilter())
        assert retracted_latest not in {v.execution_id for v in values.items}
        assert len(values) == 8

    def test_include_retracted(self, dal_db, retracted_latest):
        values = Reader(dal_db).values.scalar_values(MetricValueFilter(include_retracted=True))
        assert len(values) == 9

    def test_retracted_latest_execution_still_wins_its_group(self, dal_db, retracted_latest):
        reader = Reader(dal_db).values
        assert len(reader.scalar_values(MetricValueFilter(latest_only=True))) == 0
        latest = reader.scalar_values(MetricValueFilter(latest_only=True, include_retracted=True))
        assert {v.execution_id for v in latest.items} == {retracted_latest}


class TestPromotedOnly:
    @pytest.fixture
    def superseded_execution(self, dal_db):
        """A group at version 2 holds one value, while the diagnostic stays promoted at version 1."""
        session = dal_db.session
        with session.begin():
            original = session.get(ExecutionGroup, dal_db.execution_group_id)
            group = ExecutionGroup(
                key=original.key, diagnostic_id=original.diagnostic_id, selectors={}, diagnostic_version=2
            )
            session.add(group)
            session.flush()
            execution_id = _add_execution(session, group.id, "superseded", datetime.datetime(2030, 1, 1))
            session.add(
                ScalarMetricValue.build(
                    execution_id=execution_id,
                    value=1.0,
                    attributes=None,
                    dimensions={"statistic": "mean", "metric": "tas", "source_id": "ACCESS-CM2"},
                )
            )
        return execution_id

    def test_default_drops_unpromoted_versions(self, dal_db, superseded_execution):
        values = Reader(dal_db).values.scalar_values(MetricValueFilter())
        assert superseded_execution not in {v.execution_id for v in values.items}
        assert len(values) == 8

    def test_include_superseded_keeps_every_version(self, dal_db, superseded_execution):
        values = Reader(dal_db).values.scalar_values(MetricValueFilter(include_superseded=True))
        assert len(values) == 9

    def test_latest_only_is_per_group_across_versions(self, dal_db, superseded_execution):
        reader = Reader(dal_db).values
        promoted = reader.scalar_values(MetricValueFilter(latest_only=True))
        assert superseded_execution not in {v.execution_id for v in promoted.items}
        assert len(promoted) == 8
        every_version = reader.scalar_values(MetricValueFilter(latest_only=True, include_superseded=True))
        assert len(every_version) == 9


class TestPromotedOnlyAlias:
    def test_default_excludes_superseded(self):
        assert not MetricValueFilter().include_superseded
        assert MetricValueFilter() == MetricValueFilter(include_superseded=False)

    @pytest.mark.parametrize(("promoted_only", "include_superseded"), [(True, False), (False, True)])
    def test_maps_onto_include_superseded(self, promoted_only, include_superseded):
        with pytest.warns(DeprecationWarning, match="include_superseded"):
            f = MetricValueFilter(promoted_only=promoted_only)
        assert f.include_superseded is include_superseded
        assert f == MetricValueFilter(include_superseded=include_superseded)

    def test_disagreeing_values_raise(self):
        with pytest.raises(ValueError, match="disagree"):
            MetricValueFilter(promoted_only=True, include_superseded=True)

    def test_agreeing_values_are_accepted(self):
        with pytest.warns(DeprecationWarning):
            assert MetricValueFilter(promoted_only=False, include_superseded=True).include_superseded

    @pytest.mark.parametrize(("promoted_only", "include_superseded"), [(True, False), (False, True)])
    def test_evolve_applies_alias(self, promoted_only, include_superseded):
        with pytest.warns(DeprecationWarning):
            f = attrs.evolve(MetricValueFilter(), promoted_only=promoted_only)
        assert bool(f.include_superseded) is include_superseded

    def test_evolve_keeps_resolved_value(self):
        f = attrs.evolve(MetricValueFilter(include_superseded=True), latest_only=True)
        assert f.include_superseded is True


class _FakeScalar:
    """Minimal stand-in exposing the attributes ``detect_scalar_outliers`` reads."""

    def __init__(self, id_, value, dimensions):
        self.id = id_
        self.value = value
        self.dimensions = dimensions


def _flags(scalars, **policy_kwargs):
    """Run detection and return a ``{id: is_outlier}`` mapping."""
    annotated, _ = detect_scalar_outliers(scalars, OutlierPolicy(**policy_kwargs))
    return {a.value.id: a.is_outlier for a in annotated}


class TestDetectScalarOutliers:
    def test_outlier_policy_rejects_min_n_below_two(self):
        with pytest.raises(ValueError):
            OutlierPolicy(min_n=1)
        with pytest.raises(ValueError):
            OutlierPolicy(min_n=0)
        # sanity: the safe boundary and default still construct
        assert OutlierPolicy(min_n=2).min_n == 2
        assert OutlierPolicy().min_n == 4

    def test_fallback_path_nan_does_not_disable_detection(self):
        # No source_id -> fallback IQR over raw values. A single NaN must not poison the quantiles:
        # the wild outlier among the finite values is still flagged, and the NaN is flagged non-finite.
        scalars = [
            *[_FakeScalar(i, v, {"metric": "tas"}) for i, v in enumerate([10.0, 10.5, 9.5, 10.2, 9.8])],
            _FakeScalar(100, 1000.0, {"metric": "tas"}),  # wild outlier
            _FakeScalar(200, math.nan, {"metric": "tas"}),  # non-finite
        ]
        flags = _flags(scalars, min_n=4)
        assert flags[100] is True
        assert flags[200] is True
        assert all(flags[i] is False for i in range(5))

    def test_per_source_all_nan_source_does_not_poison_bounds(self):
        # A source whose values are all NaN yields a NaN mean; it must be dropped before the
        # quantiles so the well-behaved sources are not all flagged, while the wild source is.
        scalars = [
            _FakeScalar(i, v, {"metric": "tas", "source_id": sid})
            for i, (sid, v) in enumerate(
                [("A", 10.0), ("B", 10.5), ("C", 9.5), ("D", 10.2), ("WILD", 1000.0)]
            )
        ]
        scalars += [
            _FakeScalar(500, math.nan, {"metric": "tas", "source_id": "BAD"}),
            _FakeScalar(501, math.nan, {"metric": "tas", "source_id": "BAD"}),
        ]
        flags = _flags(scalars, min_n=4)
        assert flags[4] is True  # WILD source
        assert flags[500] is True and flags[501] is True  # non-finite always flagged
        assert all(flags[i] is False for i in range(4))  # A-D not poisoned

    def test_zero_iqr_flags_nothing_finite(self):
        # Identical per-source means -> zero spread -> collapsed bounds must flag nothing, even for a
        # value that differs by an epsilon. Genuinely non-finite values are still flagged.
        scalars = [
            _FakeScalar(i, 10.0, {"metric": "tas", "source_id": sid})
            for i, sid in enumerate(["A", "B", "C", "D"])
        ]
        scalars.append(_FakeScalar(10, 10.0 + 1e-9, {"metric": "tas", "source_id": "E"}))
        scalars.append(_FakeScalar(11, math.nan, {"metric": "tas", "source_id": "F"}))
        flags = _flags(scalars, min_n=4)
        assert flags[10] is False  # epsilon deviation not flagged under zero spread
        assert flags[11] is True  # non-finite still flagged
        assert all(flags[i] is False for i in range(4))

    def test_min_n_counts_distinct_sources_not_rows(self):
        # Many rows but only 3 distinct source_ids; with min_n=4 detection must not run, so the wild
        # value is left unflagged despite the row count exceeding min_n.
        scalars = [
            _FakeScalar(i, v, {"metric": "tas", "source_id": sid})
            for i, (sid, v) in enumerate(
                [("A", 10.0), ("A", 10.1), ("B", 10.5), ("B", 9.9), ("C", 1000.0), ("C", 9.5)]
            )
        ]
        flags = _flags(scalars, min_n=4)
        assert all(v is False for v in flags.values())


def _shared_session_is_off_limits(database):
    raise AssertionError("the shared session must not be used when an explicit one was given")


class TestReaderSession:
    def test_defaults_to_the_database_session(self, dal_db):
        reader = Reader(dal_db)
        assert reader.session is dal_db.session
        assert reader.values.session is dal_db.session
        assert reader.executions.session is dal_db.session
        assert reader.datasets.session is dal_db.session
        assert reader.diagnostics.session is dal_db.session
        assert reader.resources.session is dal_db.session

    def test_an_explicit_session_is_used_by_every_sub_reader(self, dal_db):
        with dal_db.session_scope() as session:
            reader = Reader(dal_db, session=session)
            assert reader.session is session
            assert reader.values.session is session
            assert reader.executions.session is session
            assert reader.datasets.session is session
            assert reader.diagnostics.session is session
            assert reader.resources.session is session

    def test_a_read_goes_through_the_explicit_session(self, dal_db, mocker):
        expected = Reader(dal_db).values.scalar_values()

        with dal_db.session_scope() as session:
            # Reaching for the shared session now fails, so the read can only use the scope's one.
            mocker.patch.object(type(dal_db), "session", property(_shared_session_is_off_limits), create=True)
            values = Reader(dal_db, session=session).values.scalar_values()

        assert len(values.items) == len(expected.items) > 0


def _per_dimension_facets(session, stmt, entity):
    """The previous implementation: one DISTINCT per dimension."""
    facets = {}
    for key in entity._cv_dimensions:
        sub = stmt.with_only_columns(distinct(getattr(entity, key))).order_by(None)
        values = [v for (v,) in session.execute(sub) if v is not None]
        if values:
            facets[key] = sorted(values)
    return facets


class TestFacets:
    @pytest.mark.parametrize("value_type", [MetricValueType.SCALAR, MetricValueType.SERIES])
    @pytest.mark.parametrize("filtered", [False, True])
    def test_matches_collection_facets(self, dal_db, value_type, filtered):
        values = Reader(dal_db).values
        filters = MetricValueFilter(execution_group_ids=[dal_db.execution_group_id]) if filtered else None
        if value_type is MetricValueType.SCALAR:
            expected = values.scalar_values(filters, with_facets=True).facets
        else:
            expected = values.series_values(filters, with_facets=True).facets

        assert values.facets(filters, value_type=value_type) == expected
        assert expected

    @pytest.mark.parametrize(
        ("value_type", "entity", "select_fn"),
        [
            (MetricValueType.SCALAR, ScalarMetricValue, select_scalar_values),
            (MetricValueType.SERIES, SeriesMetricValue, select_series_values),
        ],
    )
    def test_matches_per_dimension_distinct(self, dal_db, value_type, entity, select_fn):
        stmt = select_fn(MetricValueFilter())
        assert collect_facets(dal_db.session, stmt, entity) == _per_dimension_facets(
            dal_db.session, stmt, entity
        )

    def test_filter_narrows_facets(self, dal_db):
        values = Reader(dal_db).values
        wild = values.facets(MetricValueFilter(dimensions={"source_id": "WILD"}))
        assert {f.key: f.values for f in wild}["source_id"] == ("WILD",)

    def test_single_query(self, dal_db, count_queries):
        stmt = select_scalar_values(MetricValueFilter())
        with count_queries(dal_db) as statements:
            collect_facets(dal_db.session, stmt, ScalarMetricValue)
        assert len(statements) == 1


class TestWithFacetsDeprecation:
    def test_unset_warns_and_computes(self, dal_db):
        values = Reader(dal_db).values
        with pytest.warns(FutureWarning, match=r"ValuesReader\.facets\(\)") as record:
            scalar = values.scalar_values()
        with pytest.warns(FutureWarning):
            series = values.series_values()
        assert scalar.facets
        assert series.facets
        assert record[0].filename == __file__

    @pytest.mark.parametrize("flag", [True, False])
    def test_explicit_does_not_warn(self, dal_db, flag, recwarn):
        values = Reader(dal_db).values
        values.scalar_values(with_facets=flag)
        values.series_values(with_facets=flag)
        assert not [w for w in recwarn if issubclass(w.category, FutureWarning)]
