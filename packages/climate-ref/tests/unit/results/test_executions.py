"""Unit tests for the execution-group / execution read layer."""

import datetime

import pytest
from climate_ref_esmvaltool import provider as esmvaltool_provider
from climate_ref_pmp import provider as pmp_provider

from climate_ref.models import Execution, ExecutionGroup
from climate_ref.models.diagnostic import Diagnostic
from climate_ref.models.execution import ExecutionOutput, ResultOutputType
from climate_ref.provider_registry import _register_provider
from climate_ref.results import ExecutionGroupFilter, MetricValueFilter, OutlierPolicy, Reader
from climate_ref.results.executions import select_execution_outputs, select_execution_statistics


def test_import_smoke() -> None:
    """Public surface imports cleanly (guards the `values`<->`artifacts`<->`executions` wiring)."""
    assert ExecutionGroupFilter is not None
    assert MetricValueFilter is not None
    assert OutlierPolicy is not None
    assert Reader is not None


@pytest.fixture
def db_with_groups(db_seeded):
    """A seeded DB with execution groups spanning two providers, dirty/successful/facet variety."""
    with db_seeded.session.begin():
        _register_provider(db_seeded, pmp_provider)
        _register_provider(db_seeded, esmvaltool_provider)

        # Diagnostic 1, Provider 1 (pmp), Facets: source_id=GFDL-ESM4, variable_id=tas
        diag_1 = db_seeded.session.query(Diagnostic).filter_by(slug="enso_tel").first()
        eg1 = ExecutionGroup(
            key="key1",
            diagnostic_id=diag_1.id,
            selectors={"cmip6": [["source_id", "GFDL-ESM4"], ["variable_id", "tas"]]},
        )
        db_seeded.session.add(eg1)

        # Diagnostic 2, Provider 1 (pmp), Facets: source_id=ACCESS-ESM1-5, variable_id=pr
        diag_2 = (
            db_seeded.session.query(Diagnostic)
            .filter_by(slug="extratropical-modes-of-variability-nao")
            .first()
        )
        eg2 = ExecutionGroup(
            key="key2",
            diagnostic_id=diag_2.id,
            selectors={"cmip6": [["source_id", "ACCESS-ESM1-5"], ["variable_id", "pr"]]},
        )
        db_seeded.session.add(eg2)

        # Diagnostic 3, Provider 2 (esmvaltool), Facets: source_id=CNRM-CM6-1, variable_id=tas
        diag_3 = db_seeded.session.query(Diagnostic).filter_by(slug="enso-characteristics").first()
        eg3 = ExecutionGroup(
            key="key3",
            diagnostic_id=diag_3.id,
            selectors={"cmip6": [["source_id", "CNRM-CM6-1"], ["variable_id", "tas"]]},
        )
        db_seeded.session.add(eg3)

        # Diagnostic 4, Provider 2 (esmvaltool)
        diag_4 = db_seeded.session.query(Diagnostic).filter_by(slug="sea-ice-area-basic").first()
        eg4 = ExecutionGroup(
            key="key4", diagnostic_id=diag_4.id, selectors={"cmip6": [["experiment_id", "historical"]]}
        )
        db_seeded.session.add(eg4)

        db_seeded.session.flush()
        db_seeded.session.add(
            Execution(
                execution_group_id=eg1.id, successful=True, output_fragment="out1", dataset_hash="hash1"
            )
        )
        db_seeded.session.add(
            Execution(
                execution_group_id=eg2.id, successful=True, output_fragment="out2", dataset_hash="hash2"
            )
        )
        db_seeded.session.add(
            Execution(
                execution_group_id=eg3.id, successful=False, output_fragment="out3", dataset_hash="hash3"
            )
        )
        db_seeded.session.add(
            Execution(
                execution_group_id=eg4.id, successful=True, output_fragment="out4", dataset_hash="hash4"
            )
        )

        # A dirty execution group (successful=True latest, but flagged dirty)
        eg5 = ExecutionGroup(
            key="key5",
            diagnostic_id=diag_4.id,
            selectors={"cmip6": [["experiment_id", "historical"]]},
            dirty=True,
        )
        db_seeded.session.add(eg5)
        db_seeded.session.flush()
        db_seeded.session.add(
            Execution(
                execution_group_id=eg5.id, successful=True, output_fragment="out5", dataset_hash="hash5"
            )
        )

        # An execution group with no executions at all (not-started)
        eg6 = ExecutionGroup(
            key="key6", diagnostic_id=diag_4.id, selectors={"cmip6": [["experiment_id", "ssp126"]]}
        )
        db_seeded.session.add(eg6)

        # A running execution group (successful=None)
        eg7 = ExecutionGroup(
            key="key7", diagnostic_id=diag_4.id, selectors={"cmip6": [["experiment_id", "ssp245"]]}
        )
        db_seeded.session.add(eg7)
        db_seeded.session.flush()
        db_seeded.session.add(
            Execution(
                execution_group_id=eg7.id, successful=None, output_fragment="out7", dataset_hash="hash7"
            )
        )
    db_seeded.session.commit()

    ids = {
        "key1": eg1.id,
        "key2": eg2.id,
        "key3": eg3.id,
        "key4": eg4.id,
        "key5": eg5.id,
        "key6": eg6.id,
        "key7": eg7.id,
    }
    db_seeded.group_ids = ids
    return db_seeded


class TestGroupsFilter:
    def test_diagnostic_contains(self, db_with_groups):
        reader = Reader(db_with_groups)
        coll = reader.executions.groups(ExecutionGroupFilter(diagnostic_contains=["enso"]))
        keys = {g.key for g in coll}
        assert "key1" in keys  # enso_tel
        assert "key3" in keys  # enso-characteristics
        assert "key2" not in keys
        assert "key4" not in keys

    def test_provider_contains(self, db_with_groups):
        reader = Reader(db_with_groups)
        coll = reader.executions.groups(ExecutionGroupFilter(provider_contains=["pmp"]))
        keys = {g.key for g in coll}
        assert "key1" in keys
        assert "key2" in keys
        assert "key3" not in keys

    def test_dirty(self, db_with_groups):
        reader = Reader(db_with_groups)
        coll = reader.executions.groups(ExecutionGroupFilter(dirty=True))
        keys = {g.key for g in coll}
        assert keys == {"key5"}

        coll = reader.executions.groups(ExecutionGroupFilter(dirty=False))
        keys = {g.key for g in coll}
        assert "key5" not in keys

    def test_successful_true_only_true(self, db_with_groups):
        reader = Reader(db_with_groups)
        coll = reader.executions.groups(ExecutionGroupFilter(successful=True))
        keys = {g.key for g in coll}
        # key1, key2, key4, key5 have successful=True latest executions
        assert keys == {"key1", "key2", "key4", "key5"}

    def test_successful_false_includes_failed_running_and_no_exec(self, db_with_groups):
        reader = Reader(db_with_groups)
        coll = reader.executions.groups(ExecutionGroupFilter(successful=False))
        keys = {g.key for g in coll}
        # key3 (failed), key6 (no exec), key7 (running) must all be included.
        assert {"key3", "key6", "key7"} <= keys
        # while truly successful groups must be excluded.
        assert not ({"key1", "key2", "key4", "key5"} & keys)

    def test_facet_bare_key(self, db_with_groups):
        reader = Reader(db_with_groups)
        coll = reader.executions.groups(ExecutionGroupFilter(facets={"source_id": ["GFDL-ESM4"]}))
        keys = {g.key for g in coll}
        assert keys == {"key1"}

    def test_facet_dataset_type_scoped_key(self, db_with_groups):
        reader = Reader(db_with_groups)
        coll = reader.executions.groups(ExecutionGroupFilter(facets={"cmip6.source_id": ["GFDL-ESM4"]}))
        keys = {g.key for g in coll}
        assert keys == {"key1"}

    def test_bare_string_rejected_diagnostic_contains(self):
        with pytest.raises(TypeError):
            ExecutionGroupFilter(diagnostic_contains="enso")

    def test_bare_string_rejected_provider_contains(self):
        with pytest.raises(TypeError):
            ExecutionGroupFilter(provider_contains="pmp")


class TestGroupsPromotedVersion:
    def test_promoted_version_default_excludes_superseded(self, db_with_groups):
        # Bump the diagnostic's promoted_version so eg1's diagnostic_version=1 becomes superseded.
        session = db_with_groups.session
        diag = session.query(ExecutionGroup).filter_by(key="key1").first().diagnostic
        with session.begin_nested() if session.in_transaction() else session.begin():
            diag.promoted_version = 2

        reader = Reader(db_with_groups)
        coll = reader.executions.groups(ExecutionGroupFilter(diagnostic_contains=["enso_tel"]))
        assert coll.total_count == 0

    def test_include_superseded_true_sees_it(self, db_with_groups):
        session = db_with_groups.session
        diag = session.query(ExecutionGroup).filter_by(key="key1").first().diagnostic
        with session.begin_nested() if session.in_transaction() else session.begin():
            diag.promoted_version = 2

        reader = Reader(db_with_groups)
        coll = reader.executions.groups(
            ExecutionGroupFilter(diagnostic_contains=["enso_tel"], include_superseded=True)
        )
        assert coll.total_count == 1


class TestGroupsPagination:
    def test_total_count_vs_page_with_offset_limit(self, db_with_groups):
        reader = Reader(db_with_groups)
        full = reader.executions.groups()
        assert full.total_count == 7

        page = reader.executions.groups(offset=2, limit=2)
        assert page.total_count == 7
        assert len(page) == 2
        assert page.offset == 2
        assert page.limit == 2


class TestGroupsToPandas:
    def test_to_pandas_columns(self, db_with_groups):
        reader = Reader(db_with_groups)
        coll = reader.executions.groups()
        df = coll.to_pandas()
        expected_cols = [
            "id",
            "key",
            "provider",
            "diagnostic",
            "dirty",
            "successful",
            "created_at",
            "updated_at",
            "selectors",
        ]
        assert list(df.columns) == expected_cols
        assert len(df) == 7

    def test_to_pandas_columns_when_empty(self, db_with_groups):
        reader = Reader(db_with_groups)
        coll = reader.executions.groups(ExecutionGroupFilter(diagnostic_contains=["nonexistent"]))
        df = coll.to_pandas()
        expected_cols = [
            "id",
            "key",
            "provider",
            "diagnostic",
            "dirty",
            "successful",
            "created_at",
            "updated_at",
            "selectors",
        ]
        assert list(df.columns) == expected_cols
        assert len(df) == 0

    def test_detached_survival(self, db_with_groups):
        reader = Reader(db_with_groups)
        coll = reader.executions.groups()
        db_with_groups.session.expunge_all()
        df = coll.to_pandas()
        assert len(df) == 7
        selectors = [item.selectors for item in coll]
        assert all(isinstance(s, dict) for s in selectors)
        # Ensure it's a plain dict, not the ORM-attached mapping.
        assert type(next(s for s in selectors if s)) is dict


class TestGroupLatestMapping:
    def test_latest_none_when_no_executions(self, db_with_groups):
        reader = Reader(db_with_groups)
        coll = reader.executions.groups(ExecutionGroupFilter(facets={"experiment_id": ["ssp126"]}))
        assert len(coll) == 1
        assert coll.items[0].latest is None
        assert coll.items[0].successful is None

    def test_latest_successful(self, db_with_groups):
        reader = Reader(db_with_groups)
        coll = reader.executions.groups(ExecutionGroupFilter(diagnostic_contains=["enso_tel"]))
        assert len(coll) == 1
        view = coll.items[0]
        assert view.latest is not None
        assert view.latest.successful is True
        assert view.successful is True

    def test_latest_failed(self, db_with_groups):
        reader = Reader(db_with_groups)
        coll = reader.executions.groups(ExecutionGroupFilter(diagnostic_contains=["enso-characteristics"]))
        assert len(coll) == 1
        view = coll.items[0]
        assert view.latest is not None
        assert view.latest.successful is False

    def test_latest_running(self, db_with_groups):
        reader = Reader(db_with_groups)
        coll = reader.executions.groups(ExecutionGroupFilter(facets={"experiment_id": ["ssp245"]}))
        assert len(coll) == 1
        view = coll.items[0]
        assert view.latest is not None
        assert view.latest.successful is None
        assert view.successful is None


class TestSelectExecutionStatistics:
    def test_raw_rows(self, db_with_groups):
        stmt = select_execution_statistics()
        rows = db_with_groups.session.execute(stmt).all()
        by_diag = {row.diagnostic: row for row in rows}
        assert "enso_tel" in by_diag
        row = by_diag["enso_tel"]
        assert row.total == 1
        assert row.successful == 1
        assert row.failed == 0
        assert row.running == 0
        assert row.not_started == 0

    def test_raw_rows_provider_filter(self, db_with_groups):
        stmt = select_execution_statistics(provider_contains=["pmp"])
        rows = db_with_groups.session.execute(stmt).all()
        providers = {row.provider for row in rows}
        assert providers == {"pmp"}

    def test_raw_rows_diagnostic_filter(self, db_with_groups):
        stmt = select_execution_statistics(diagnostic_contains=["enso"])
        rows = db_with_groups.session.execute(stmt).all()
        diagnostics = {row.diagnostic for row in rows}
        assert diagnostics == {"enso_tel", "enso-characteristics"}


class TestStatistics:
    def test_status_counts(self, db_with_groups):
        reader = Reader(db_with_groups)
        stats = reader.executions.statistics()
        by_diag = {s.diagnostic: s for s in stats}

        # sea-ice-area-basic has key4 (successful), key5 (successful, dirty),
        # key6 (not started), key7 (running).
        sea_ice = by_diag["sea-ice-area-basic"]
        assert sea_ice.total == 4
        assert sea_ice.successful == 2
        assert sea_ice.not_started == 1
        assert sea_ice.running == 1
        assert sea_ice.dirty == 1
        assert sea_ice.failed == 0

    def test_provider_substring_filter(self, db_with_groups):
        reader = Reader(db_with_groups)
        stats = reader.executions.statistics(provider_contains=["esmvaltool"])
        providers = {s.provider for s in stats}
        assert providers == {"esmvaltool"}

    def test_diagnostic_substring_filter(self, db_with_groups):
        reader = Reader(db_with_groups)
        stats = reader.executions.statistics(diagnostic_contains=["enso"])
        diagnostics = {s.diagnostic for s in stats}
        assert diagnostics == {"enso_tel", "enso-characteristics"}


class TestLatestExecution:
    def test_returns_latest_by_created_at(self, db_with_groups):
        session = db_with_groups.session
        reader = Reader(db_with_groups)
        group_id = db_with_groups.group_ids["key1"]

        older_id = session.query(Execution).filter_by(execution_group_id=group_id).one().id

        with session.begin_nested() if session.in_transaction() else session.begin():
            newer = Execution(
                execution_group_id=group_id,
                output_fragment="frag-newer",
                dataset_hash="hash-newer",
                successful=True,
                created_at=datetime.datetime.now(tz=datetime.UTC) + datetime.timedelta(hours=1),
            )
            session.add(newer)
            session.flush()
            newer_id = newer.id

        latest = reader.executions.latest_execution(group_id)
        assert latest is not None
        assert latest.id == newer_id
        assert latest.id != older_id

    def test_none_for_empty_group(self, db_with_groups):
        reader = Reader(db_with_groups)
        group_id = db_with_groups.group_ids["key6"]
        assert reader.executions.latest_execution(group_id) is None


@pytest.fixture
def execution_with_outputs(db_with_groups):
    """The `key1` execution, with a mix of `ExecutionOutput` rows attached."""
    session = db_with_groups.session
    group_id = db_with_groups.group_ids["key1"]
    execution = session.query(Execution).filter_by(execution_group_id=group_id).one()

    with session.begin_nested() if session.in_transaction() else session.begin():
        session.add(
            ExecutionOutput(
                execution_id=execution.id,
                output_type=ResultOutputType.Data,
                filename="data.csv",
                short_name="data-short",
                long_name="Data long name",
                description="A data output",
            )
        )
        session.add(
            ExecutionOutput(
                execution_id=execution.id,
                output_type=ResultOutputType.Plot,
                filename="plot.png",
                short_name="plot-short",
                long_name="Plot long name",
            )
        )
        # A metadata-only output with no filename (e.g. an HTML summary row without a file).
        session.add(
            ExecutionOutput(
                execution_id=execution.id,
                output_type=ResultOutputType.HTML,
                filename=None,
                short_name="html-short",
            )
        )

    return db_with_groups, execution.id


class TestOutputs:
    def test_dto_fields_and_order(self, execution_with_outputs):
        db, execution_id = execution_with_outputs
        reader = Reader(db)

        outs = reader.executions.outputs(execution_id)

        assert len(outs) == 3
        # Ordered by (output_type, id): HTML < Data < Plot alphabetically by enum value.
        assert [o.output_type for o in outs] == ["data", "html", "plot"]

        data_out = next(o for o in outs if o.output_type == "data")
        assert data_out.execution_id == execution_id
        assert data_out.filename == "data.csv"
        assert data_out.short_name == "data-short"
        assert data_out.long_name == "Data long name"
        assert data_out.description == "A data output"
        assert isinstance(data_out.dimensions, dict)

        html_out = next(o for o in outs if o.output_type == "html")
        assert html_out.filename is None
        assert html_out.long_name is None

    def test_empty_for_execution_with_no_outputs(self, db_with_groups):
        reader = Reader(db_with_groups)
        group_id = db_with_groups.group_ids["key2"]
        execution = db_with_groups.session.query(Execution).filter_by(execution_group_id=group_id).one()

        assert reader.executions.outputs(execution.id) == ()

    def test_detached_survival(self, execution_with_outputs):
        db, execution_id = execution_with_outputs
        reader = Reader(db)

        outs = reader.executions.outputs(execution_id)
        db.session.expunge_all()

        assert len(outs) == 3
        for out in outs:
            assert type(out.dimensions) is dict


class TestSelectExecutionOutputs:
    def test_raw_rows_count_and_ordering(self, execution_with_outputs):
        db, execution_id = execution_with_outputs
        stmt = select_execution_outputs(execution_id)
        rows = db.session.execute(stmt).scalars().all()

        assert len(rows) == 3
        output_types = [row.output_type.value for row in rows]
        assert output_types == sorted(output_types)

    def test_raw_rows_scoped_to_execution(self, db_with_groups):
        group_id = db_with_groups.group_ids["key2"]
        execution = db_with_groups.session.query(Execution).filter_by(execution_group_id=group_id).one()

        stmt = select_execution_outputs(execution.id)
        rows = db_with_groups.session.execute(stmt).scalars().all()
        assert rows == []


class TestReaderArtifactsGating:
    def test_raises_without_paths(self, db_with_groups):
        reader = Reader(db_with_groups)
        with pytest.raises(ValueError):
            reader.artifacts

    def test_works_with_paths(self, db_with_groups, tmp_path):
        reader = Reader(db_with_groups, results=tmp_path)
        assert reader.artifacts is not None
