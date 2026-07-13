"""Unit tests for the diagnostic read layer."""

import pytest
from climate_ref_esmvaltool import provider as esmvaltool_provider
from climate_ref_pmp import provider as pmp_provider

from climate_ref.models.diagnostic import Diagnostic
from climate_ref.models.execution import Execution, ExecutionGroup
from climate_ref.provider_registry import _register_provider
from climate_ref.results import DiagnosticFilter, Reader
from climate_ref.results.diagnostics import select_diagnostics


def test_import_smoke() -> None:
    """Public surface imports cleanly (guards the `values`<->`diagnostics` wiring)."""
    assert DiagnosticFilter is not None
    assert Reader is not None


@pytest.fixture
def db_with_diagnostics(db_seeded):
    """A seeded DB with two providers and execution groups spread unevenly across diagnostics."""
    with db_seeded.session.begin():
        _register_provider(db_seeded, pmp_provider)
        _register_provider(db_seeded, esmvaltool_provider)

        diag_1 = db_seeded.session.query(Diagnostic).filter_by(slug="enso_tel").first()
        diag_2 = (
            db_seeded.session.query(Diagnostic)
            .filter_by(slug="extratropical-modes-of-variability-nao")
            .first()
        )
        # diag_1 gets two execution groups, diag_2 gets one, enso-characteristics gets none.
        db_seeded.session.add(ExecutionGroup(key="key1", diagnostic_id=diag_1.id, selectors={}))
        db_seeded.session.add(ExecutionGroup(key="key2", diagnostic_id=diag_1.id, selectors={}))
        db_seeded.session.add(ExecutionGroup(key="key3", diagnostic_id=diag_2.id, selectors={}))

    db_seeded.session.commit()

    return db_seeded


class TestSelectDiagnostics:
    def test_raw_rows_include_all_diagnostics(self, db_with_diagnostics):
        stmt = select_diagnostics()
        rows = db_with_diagnostics.session.execute(stmt).all()
        slugs = {row.slug for row in rows}
        assert "enso_tel" in slugs
        assert "enso-characteristics" in slugs

    def test_execution_group_counts(self, db_with_diagnostics):
        stmt = select_diagnostics()
        rows = db_with_diagnostics.session.execute(stmt).all()
        by_slug = {row.slug: row for row in rows}

        assert by_slug["enso_tel"].execution_group_count == 2
        assert by_slug["extratropical-modes-of-variability-nao"].execution_group_count == 1
        # A diagnostic with zero execution groups still appears, with a count of 0.
        assert by_slug["enso-characteristics"].execution_group_count == 0

    def test_provider_contains_filter(self, db_with_diagnostics):
        stmt = select_diagnostics(DiagnosticFilter(provider_contains=["pmp"]))
        rows = db_with_diagnostics.session.execute(stmt).all()
        providers = {row.provider_slug for row in rows}
        assert providers == {"pmp"}

    def test_diagnostic_contains_filter(self, db_with_diagnostics):
        stmt = select_diagnostics(DiagnosticFilter(diagnostic_contains=["enso_tel"]))
        rows = db_with_diagnostics.session.execute(stmt).all()
        slugs = {row.slug for row in rows}
        assert slugs == {"enso_tel"}

    def test_ordered_by_provider_then_diagnostic(self, db_with_diagnostics):
        stmt = select_diagnostics()
        rows = db_with_diagnostics.session.execute(stmt).all()
        pairs = [(row.provider_slug, row.slug) for row in rows]
        assert pairs == sorted(pairs)

    def test_bare_string_rejected_diagnostic_contains(self):
        with pytest.raises(TypeError):
            DiagnosticFilter(diagnostic_contains="enso")

    def test_bare_string_rejected_provider_contains(self):
        with pytest.raises(TypeError):
            DiagnosticFilter(provider_contains="pmp")


class TestDiagnosticsReaderList:
    def test_dto_mapping(self, db_with_diagnostics):
        reader = Reader(db_with_diagnostics)
        coll = reader.diagnostics.list(DiagnosticFilter(diagnostic_contains=["enso_tel"]))
        assert len(coll) == 1
        view = coll.items[0]
        assert view.provider_slug == "pmp"
        assert view.slug == "enso_tel"
        assert view.name
        assert view.promoted_version == 1
        assert view.execution_group_count == 2
        # Both groups are at the promoted version with no executions yet.
        assert view.total == 2
        assert view.successful == 0
        assert view.inflight == 0

    def test_promoted_version_status_counts(self, db_with_diagnostics):
        # enso_tel has two promoted-version groups (key1, key2); give key1 a successful latest
        # execution and key2 a running (successful IS NULL) one.
        with db_with_diagnostics.session.begin():
            group_1 = db_with_diagnostics.session.query(ExecutionGroup).filter_by(key="key1").one()
            group_2 = db_with_diagnostics.session.query(ExecutionGroup).filter_by(key="key2").one()
            db_with_diagnostics.session.add(
                Execution(
                    execution_group_id=group_1.id,
                    output_fragment="frag1",
                    dataset_hash="hash1",
                    successful=True,
                )
            )
            db_with_diagnostics.session.add(
                Execution(
                    execution_group_id=group_2.id,
                    output_fragment="frag2",
                    dataset_hash="hash2",
                    successful=None,
                )
            )

        reader = Reader(db_with_diagnostics)
        view = reader.diagnostics.list(DiagnosticFilter(diagnostic_contains=["enso_tel"])).items[0]
        assert view.successful == 1
        assert view.inflight == 1
        assert view.total == 2

    def test_filter_by_provider(self, db_with_diagnostics):
        reader = Reader(db_with_diagnostics)
        coll = reader.diagnostics.list(DiagnosticFilter(provider_contains=["esmvaltool"]))
        providers = {d.provider_slug for d in coll}
        assert providers == {"esmvaltool"}

    def test_filters_keyword(self, db_with_diagnostics):
        # The list-style filter parameter is named `filters`, matching the rest of the contract.
        reader = Reader(db_with_diagnostics)
        coll = reader.diagnostics.list(filters=DiagnosticFilter(diagnostic_contains=["enso_tel"]))
        assert {d.slug for d in coll} == {"enso_tel"}

    def test_pagination_deterministic(self, db_with_diagnostics):
        # Full listing is ordered by (provider, diagnostic, id); adjacent pages must partition it.
        reader = Reader(db_with_diagnostics)
        full = [(d.provider_slug, d.slug) for d in reader.diagnostics.list()]
        page_1 = [(d.provider_slug, d.slug) for d in reader.diagnostics.list(offset=0, limit=2)]
        page_2 = [(d.provider_slug, d.slug) for d in reader.diagnostics.list(offset=2, limit=2)]
        assert set(page_1).isdisjoint(page_2)
        assert page_1 + page_2 == full[:4]

    def test_pagination_offset_limit(self, db_with_diagnostics):
        reader = Reader(db_with_diagnostics)
        full = reader.diagnostics.list()
        total = full.total_count
        assert total > 2

        page = reader.diagnostics.list(offset=1, limit=1)
        assert page.total_count == total
        assert len(page) == 1
        assert page.offset == 1
        assert page.limit == 1

    def test_detached_survival(self, db_with_diagnostics):
        reader = Reader(db_with_diagnostics)
        coll = reader.diagnostics.list()
        db_with_diagnostics.session.expunge_all()
        df = coll.to_pandas()
        assert len(df) == coll.total_count

    def test_to_pandas_columns(self, db_with_diagnostics):
        reader = Reader(db_with_diagnostics)
        coll = reader.diagnostics.list()
        df = coll.to_pandas()
        assert list(df.columns) == [
            "provider",
            "diagnostic",
            "name",
            "promoted_version",
            "execution_group_count",
            "successful",
            "inflight",
            "total",
        ]

    def test_to_pandas_columns_when_empty(self, db_with_diagnostics):
        reader = Reader(db_with_diagnostics)
        coll = reader.diagnostics.list(DiagnosticFilter(diagnostic_contains=["nonexistent"]))
        df = coll.to_pandas()
        assert list(df.columns) == [
            "provider",
            "diagnostic",
            "name",
            "promoted_version",
            "execution_group_count",
            "successful",
            "inflight",
            "total",
        ]
        assert len(df) == 0


class TestDiagnosticsReaderStats:
    def test_delegates_to_execution_statistics(self, db_with_diagnostics):
        reader = Reader(db_with_diagnostics)
        stats = reader.diagnostics.stats()
        by_diag = {s.diagnostic: s for s in stats}

        assert "enso_tel" in by_diag
        assert by_diag["enso_tel"].total == 2
        assert by_diag["enso_tel"].not_started == 2

    def test_matches_executions_statistics(self, db_with_diagnostics):
        reader = Reader(db_with_diagnostics)
        via_diagnostics = reader.diagnostics.stats()
        via_executions = reader.executions.statistics()
        assert via_diagnostics == via_executions

    def test_provider_filter(self, db_with_diagnostics):
        reader = Reader(db_with_diagnostics)
        stats = reader.diagnostics.stats(provider_contains=["pmp"])
        providers = {s.provider for s in stats}
        assert providers == {"pmp"}

    def test_diagnostic_filter(self, db_with_diagnostics):
        reader = Reader(db_with_diagnostics)
        stats = reader.diagnostics.stats(diagnostic_contains=["enso_tel"])
        diagnostics = {s.diagnostic for s in stats}
        assert diagnostics == {"enso_tel"}
