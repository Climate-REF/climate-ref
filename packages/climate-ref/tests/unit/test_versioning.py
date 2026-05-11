"""Tests for diagnostic versioning read-path foundations.

Covers:
1. Migration up/down + backfill
2. Helper version filtering (include_superseded)
3. recompute_promoted_version helper
4. stats() aggregation filter (promoted_version only)
5. mark_failed_running operational invariant (version-agnostic)
6. Dormancy regression (solver does NOT branch on Diagnostic.version)
"""

import importlib.resources
import re

import pytest
from alembic import command
from alembic.config import Config as AlembicConfig
from climate_ref_example import provider as example_provider
from climate_ref_pmp import provider as pmp_provider
from sqlalchemy import inspect

from climate_ref.database import Database
from climate_ref.models.diagnostic import Diagnostic, recompute_promoted_version
from climate_ref.models.execution import Execution, ExecutionGroup, get_execution_group_and_latest_filtered
from climate_ref.provider_registry import ProviderRegistry, _register_provider
from climate_ref.solver import ExecutionSolver, solve_required_executions

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_columns(engine, table_name: str) -> set[str]:
    insp = inspect(engine)
    return {c["name"] for c in insp.get_columns(table_name)}


def _get_unique_constraints(engine, table_name: str) -> list[dict]:
    insp = inspect(engine)
    return insp.get_unique_constraints(table_name)


# ---------------------------------------------------------------------------
# Test case 1: migration up/down + backfill
# ---------------------------------------------------------------------------


class TestMigrationUpDown:
    """Verify the diagnostic-versioning migration applies and reverts cleanly."""

    def test_new_columns_present_after_migrate(self, db: Database) -> None:
        """After migration the three new columns exist."""
        cols_eg = _get_columns(db._engine, "execution_group")
        cols_diag = _get_columns(db._engine, "diagnostic")
        cols_exec = _get_columns(db._engine, "execution")

        assert "diagnostic_version" in cols_eg
        assert "promoted_version" in cols_diag
        assert "provider_version" in cols_exec

    def test_unique_constraint_is_three_columns(self, db: Database) -> None:
        """The unique constraint on execution_group must include diagnostic_version."""
        constraints = _get_unique_constraints(db._engine, "execution_group")
        named = {c["name"]: set(c["column_names"]) for c in constraints}
        # The constraint is renamed in place; look for execution_ident or any that
        # contains all three columns.
        three_col = any(
            {"diagnostic_id", "key", "diagnostic_version"} == set(c["column_names"]) for c in constraints
        )
        assert three_col, f"Expected 3-column unique constraint, got: {named}"

    def test_existing_rows_backfilled_to_version_1(self, db_seeded: Database) -> None:
        """All existing execution_group rows must have diagnostic_version=1."""
        with db_seeded.session.begin():
            eg = ExecutionGroup(key="backfill-test", diagnostic_id=1)
            db_seeded.session.add(eg)
        db_seeded.session.commit()

        eg_reloaded = db_seeded.session.query(ExecutionGroup).filter_by(key="backfill-test").one()
        assert eg_reloaded.diagnostic_version == 1

    def test_diagnostic_promoted_version_default_1(self, db_seeded: Database) -> None:
        """All existing diagnostic rows must have promoted_version=1."""
        diags = db_seeded.session.query(Diagnostic).all()
        assert diags, "db_seeded must have at least one diagnostic"
        for d in diags:
            assert d.promoted_version == 1, f"Diagnostic {d.slug} has promoted_version={d.promoted_version}"

    def test_downgrade_removes_columns(self, config) -> None:
        """Migration downgrade removes the three columns and restores 2-column constraint."""
        alembic_cfg = AlembicConfig()
        alembic_cfg.set_main_option(
            "script_location",
            str(importlib.resources.files("climate_ref").parent / "climate_ref" / "migrations"),
        )
        alembic_cfg.set_main_option("sqlalchemy.url", config.db.database_url)

        # Start from the migrated state (db fixture already migrated)
        db = Database.from_config(config, run_migrations=True)

        # Downgrade to the previous revision
        command.downgrade(alembic_cfg, "a3b4c5d6e7f8")

        cols_eg = _get_columns(db._engine, "execution_group")
        cols_diag = _get_columns(db._engine, "diagnostic")
        cols_exec = _get_columns(db._engine, "execution")

        assert "diagnostic_version" not in cols_eg
        assert "promoted_version" not in cols_diag
        assert "provider_version" not in cols_exec

        # Two-column constraint must be back
        two_col = any(
            {"diagnostic_id", "key"} == set(c["column_names"])
            and "diagnostic_version" not in c["column_names"]
            for c in _get_unique_constraints(db._engine, "execution_group")
        )
        assert two_col, "Expected 2-column constraint after downgrade"

        # Re-apply so the fixture teardown doesn't fail
        command.upgrade(alembic_cfg, "head")
        db.close()


# ---------------------------------------------------------------------------
# Test case 2: helper version filtering
# ---------------------------------------------------------------------------


class TestHelperVersionFiltering:
    """get_execution_group_and_latest_filtered respects include_superseded."""

    @pytest.fixture
    def db_two_versions(self, db_seeded: Database) -> Database:
        """Insert v1 and v2 groups for the same diagnostic; set promoted_version=2."""
        diag = db_seeded.session.query(Diagnostic).first()
        diag.promoted_version = 2

        with db_seeded.session.begin_nested():
            eg_v1 = ExecutionGroup(
                key="version-filter-key",
                diagnostic_id=diag.id,
                diagnostic_version=1,
            )
            eg_v2 = ExecutionGroup(
                key="version-filter-key",
                diagnostic_id=diag.id,
                diagnostic_version=2,
            )
            db_seeded.session.add_all([eg_v1, eg_v2])
        db_seeded.session.commit()
        return db_seeded

    def test_default_returns_only_promoted_version(self, db_two_versions: Database) -> None:
        results = get_execution_group_and_latest_filtered(db_two_versions.session)
        returned_versions = {eg.diagnostic_version for eg, _ in results}
        # Only v2 (the promoted version) should be visible
        assert 2 in returned_versions
        assert 1 not in returned_versions

    def test_include_superseded_returns_both_versions(self, db_two_versions: Database) -> None:
        results = get_execution_group_and_latest_filtered(db_two_versions.session, include_superseded=True)
        returned_versions = {eg.diagnostic_version for eg, _ in results}
        assert 1 in returned_versions
        assert 2 in returned_versions


# ---------------------------------------------------------------------------
# Test case 3: recompute_promoted_version helper
# ---------------------------------------------------------------------------


class TestRecomputePromotedVersion:
    """recompute_promoted_version correctly tracks max(diagnostic_version)."""

    def test_single_v1_group_promotes_to_1(self, db_seeded: Database) -> None:
        diag = db_seeded.session.query(Diagnostic).first()
        with db_seeded.session.begin_nested():
            eg = ExecutionGroup(
                key="recompute-test-1",
                diagnostic_id=diag.id,
                diagnostic_version=1,
            )
            db_seeded.session.add(eg)
        db_seeded.session.flush()

        result = recompute_promoted_version(diag.id, db_seeded.session)
        assert result == 1
        db_seeded.session.refresh(diag)
        assert diag.promoted_version == 1

    def test_v2_group_promotes_to_2(self, db_seeded: Database) -> None:
        # Use a fresh diagnostic with no pre-existing groups to avoid interference
        # from the seeded data which already has v1 groups.
        seed_diag = db_seeded.session.query(Diagnostic).first()
        fresh_diag = Diagnostic(
            slug="recompute-test-v2-diag",
            name="Recompute Test V2",
            provider_id=seed_diag.provider_id,
        )
        db_seeded.session.add(fresh_diag)
        db_seeded.session.commit()

        db_seeded.session.add(
            ExecutionGroup(
                key="recompute-test-2a",
                diagnostic_id=fresh_diag.id,
                diagnostic_version=1,
            )
        )
        db_seeded.session.add(
            ExecutionGroup(
                key="recompute-test-2b",
                diagnostic_id=fresh_diag.id,
                diagnostic_version=2,
            )
        )
        db_seeded.session.commit()

        result = recompute_promoted_version(fresh_diag.id, db_seeded.session)
        assert result == 2
        # Flush so the DB row is updated, then reload to confirm the write landed.
        db_seeded.session.flush()
        db_seeded.session.refresh(fresh_diag)
        assert fresh_diag.promoted_version == 2

    def test_no_groups_leaves_promoted_version_at_1(self, db_seeded: Database) -> None:
        diag = db_seeded.session.query(Diagnostic).first()
        # Ensure no groups for this diagnostic (use a fresh diagnostic)
        with db_seeded.session.begin_nested():
            fresh_diag = Diagnostic(
                slug="fresh-diag-recompute",
                name="Fresh Diagnostic",
                provider_id=diag.provider_id,
            )
            db_seeded.session.add(fresh_diag)
        db_seeded.session.flush()

        result = recompute_promoted_version(fresh_diag.id, db_seeded.session)
        # No groups → stays at default 1
        assert result == 1
        db_seeded.session.refresh(fresh_diag)
        assert fresh_diag.promoted_version == 1


# ---------------------------------------------------------------------------
# Test case 4: stats() aggregation filter
# ---------------------------------------------------------------------------


class TestStatsPromotedVersionFilter:
    """stats() CLI command only counts groups at the promoted version."""

    @pytest.fixture
    def db_stats_versions(self, db_seeded: Database) -> Database:
        """v1 and v2 groups for one diagnostic; promoted_version=2."""
        with db_seeded.session.begin():
            _register_provider(db_seeded, pmp_provider)

        diag = db_seeded.session.query(Diagnostic).filter_by(slug="enso_tel").first()
        diag.promoted_version = 2

        with db_seeded.session.begin_nested():
            eg_v1 = ExecutionGroup(
                key="stats-filter-key",
                diagnostic_id=diag.id,
                diagnostic_version=1,
                selectors={"cmip6": [["source_id", "MODEL-A"]]},
            )
            eg_v2 = ExecutionGroup(
                key="stats-filter-key",
                diagnostic_id=diag.id,
                diagnostic_version=2,
                selectors={"cmip6": [["source_id", "MODEL-A"]]},
            )
            db_seeded.session.add_all([eg_v1, eg_v2])
        db_seeded.session.flush()
        db_seeded.session.add(
            Execution(
                execution_group_id=eg_v1.id,
                successful=True,
                output_fragment="out-v1",
                dataset_hash="hash-v1",
            )
        )
        db_seeded.session.add(
            Execution(
                execution_group_id=eg_v2.id,
                successful=True,
                output_fragment="out-v2",
                dataset_hash="hash-v2",
            )
        )
        db_seeded.session.commit()
        return db_seeded

    def test_stats_shows_only_promoted_version_totals(self, db_stats_versions: Database, invoke_cli) -> None:
        """stats() must count only the promoted-version group, not both versions.

        Regression guard: if the promoted_version filter in stats() is removed,
        both the v1 and v2 groups would be counted and total would be 2, not 1.
        """
        result = invoke_cli(["executions", "stats", "--diagnostic", "enso_tel"])
        assert result.exit_code == 0
        assert "enso_tel" in result.stdout

        # Find the enso_tel row and extract all integers from it.
        # The rich table row looks like: "│ enso_tel │ 0 │ 0 │ 1 │ 0 │ 0 │ 1 │"
        # The last integer is the `total` column. It must be 1, not 2.
        enso_line = next((line for line in result.stdout.splitlines() if "enso_tel" in line), None)
        assert enso_line is not None, "enso_tel row missing from stats output"
        numbers = [int(m) for m in re.findall(r"\b(\d+)\b", enso_line)]
        assert numbers, "No numeric values found in enso_tel stats row"
        total = numbers[-1]
        assert total == 1, (
            f"stats() total for enso_tel is {total}, expected 1. "
            "The promoted_version filter may be missing — both v1 and v2 groups are visible."
        )


# ---------------------------------------------------------------------------
# Test case 5: mark_failed_running operational invariant
# ---------------------------------------------------------------------------


class TestMarkFailedRunningVersionAgnostic:
    """fail-running must surface in-flight executions regardless of diagnostic_version."""

    @pytest.fixture
    def db_with_v2_running(self, db_seeded: Database) -> Database:
        """In-flight execution on a v2 group while promoted_version=1."""
        with db_seeded.session.begin():
            _register_provider(db_seeded, pmp_provider)

        diag = db_seeded.session.query(Diagnostic).filter_by(slug="enso_tel").first()
        # promoted_version stays at 1 (default); group is at v2
        diag.promoted_version = 1

        with db_seeded.session.begin_nested():
            eg_v2 = ExecutionGroup(
                key="fail-running-v2",
                diagnostic_id=diag.id,
                diagnostic_version=2,
                selectors={"cmip6": [["source_id", "MODEL-X"]]},
            )
            db_seeded.session.add(eg_v2)
        db_seeded.session.flush()
        db_seeded.session.add(
            Execution(
                execution_group_id=eg_v2.id,
                successful=None,  # in-flight
                output_fragment="out-v2-running",
                dataset_hash="hash-v2-r",
            )
        )
        db_seeded.session.commit()
        return db_seeded

    def test_fail_running_catches_v2_in_flight(self, db_with_v2_running: Database, invoke_cli) -> None:
        result = invoke_cli(["executions", "fail-running", "--force"])
        assert result.exit_code == 0
        # At least the v2 in-flight execution was marked failed
        assert "Successfully marked" in result.stdout

        session = db_with_v2_running.session
        remaining_running = session.query(Execution).filter(Execution.successful.is_(None)).count()
        assert remaining_running == 0, "All in-flight executions must be marked failed"


# ---------------------------------------------------------------------------
# Test case 6: dormancy regression
# ---------------------------------------------------------------------------


class TestDormancyRegression:
    """The solver must not branch on ``Diagnostic.version``.

    The solver's get_or_create still uses only (diagnostic_id, key) as lookup
    keys, so newly created groups must always have diagnostic_version=1
    regardless of the Python-side Diagnostic.version class attribute.
    """

    def test_solver_creates_group_at_version_1_even_when_diagnostic_version_is_2(
        self, db_seeded: Database, config, monkeypatch
    ) -> None:
        # Monkeypatch the version attribute on the example diagnostic class to 2.
        example_diag_cls = type(example_provider.diagnostics()[0])
        monkeypatch.setattr(example_diag_cls, "version", 2, raising=False)

        # Build a solver with the example provider (same as test_solver.py `solver` fixture).
        local_solver = ExecutionSolver.build_from_db(config, db_seeded)
        local_solver.provider_registry = ProviderRegistry(providers=[example_provider])

        # dry_run=True so we get the group creation side effects without running diagnostics.
        solve_required_executions(db_seeded, config=config, solver=local_solver, dry_run=True)

        # All newly created groups must have diagnostic_version=1 (write-path not activated)
        groups = db_seeded.session.query(ExecutionGroup).all()
        assert groups, "Solver should create at least one ExecutionGroup"
        for eg in groups:
            assert eg.diagnostic_version == 1, (
                f"ExecutionGroup {eg.key!r} has diagnostic_version={eg.diagnostic_version}; "
                "solver must not branch on Diagnostic.version"
            )

        # promoted_version on all diagnostics must also remain at 1
        diags = db_seeded.session.query(Diagnostic).all()
        for d in diags:
            assert d.promoted_version == 1, (
                f"Diagnostic {d.slug!r} has promoted_version={d.promoted_version}; "
                "recompute_promoted_version must not promote beyond 1 when all groups are v1"
            )
