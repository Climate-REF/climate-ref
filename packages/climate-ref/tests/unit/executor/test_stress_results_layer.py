"""
Stress / performance harness for the results ingest layer.

These tests quantify the database round-trips made while ingesting metric values,
by counting the SQL statements issued against the engine. They exist to guard the
axis-resolution path inR
[ingest_series_values][climate_ref.executor.result_handling.ingest_series_values]
against regressing into an N+1 over distinct index axes: the statement count must
stay a small constant regardless of how many distinct axes a batch contains.
"""

import contextlib
import pathlib
from collections.abc import Iterator

import pytest
from sqlalchemy import event, func, select

from climate_ref.database import Database
from climate_ref.executor.result_handling import ingest_series_values
from climate_ref.models import SeriesIndex, SeriesMetricValue
from climate_ref.models.diagnostic import Diagnostic as DiagnosticModel
from climate_ref.models.execution import Execution, ExecutionGroup
from climate_ref.models.provider import Provider as ProviderModel
from climate_ref_core.metric_values import SeriesMetricValue as TSeries
from climate_ref_core.pycmec.controlled_vocabulary import CV


class _Result:
    """Minimal stand-in for an ``ExecutionResult`` exposing only what ingest needs."""

    def __init__(self, scratch_dir: pathlib.Path) -> None:
        self.series_filename = pathlib.Path("series.json")
        self._scratch_dir = scratch_dir

    def to_output_path(self, filename: pathlib.Path | str | None) -> pathlib.Path:
        return self._scratch_dir / filename if filename else self._scratch_dir


@contextlib.contextmanager
def count_statements(database: Database) -> Iterator[dict[str, int]]:
    """Count SQL statements issued against the database engine within the block."""
    counter = {"total": 0, "select": 0, "insert": 0}

    def _after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
        counter["total"] += 1
        head = statement.lstrip().split(" ", 1)[0].lower()
        if head == "select":
            counter["select"] += 1
        elif head == "insert":
            counter["insert"] += 1

    event.listen(database._engine, "after_cursor_execute", _after_cursor_execute)
    try:
        yield counter
    finally:
        event.remove(database._engine, "after_cursor_execute", _after_cursor_execute)


def _make_execution(database: Database, key: str) -> Execution:
    """Create a provider/diagnostic/group/execution chain and return the execution."""
    with database.session.begin():
        provider = ProviderModel(name="stress", slug=f"stress-{key}", version="v0.1.0")
        database.session.add(provider)
        database.session.flush()

        diagnostic = DiagnosticModel(name="stress", slug=f"stress-{key}", provider_id=provider.id)
        database.session.add(diagnostic)
        database.session.flush()

        group = ExecutionGroup(
            key=key,
            diagnostic_id=diagnostic.id,
            selectors={},
            dirty=False,
        )
        database.session.add(group)
        database.session.flush()

        execution = Execution(
            execution_group_id=group.id,
            successful=True,
            output_fragment=f"stress/{key}",
            dataset_hash=f"hash-{key}",
        )
        database.session.add(execution)
        database.session.flush()

        return execution


def _distinct_axis_series(n: int) -> list[TSeries]:
    """``n`` single-point series, each with a distinct index (so ``n`` distinct axes)."""
    return [
        TSeries(dimensions={"source_id": "m"}, values=[float(i)], index=[i], index_name="time")
        for i in range(n)
    ]


def _shared_axis_series(n: int) -> list[TSeries]:
    """``n`` series that all share a single common index (so one distinct axis)."""
    return [
        TSeries(
            dimensions={"source_id": "m"},
            values=[float(i), float(i) + 1, float(i) + 2],
            index=[0, 1, 2],
            index_name="time",
        )
        for i in range(n)
    ]


def _write_and_ingest(
    database: Database,
    config,
    execution: Execution,
    series: list[TSeries],
    scratch_root: pathlib.Path,
) -> dict[str, int]:
    """Write ``series`` to a scratch file and ingest it, counting SQL statements."""
    # Touch the (post-commit expired) execution outside the counted block so its refresh
    # SELECT is not attributed to ingest.
    scratch_dir = scratch_root / execution.output_fragment
    scratch_dir.mkdir(parents=True, exist_ok=True)
    TSeries.dump_to_json(scratch_dir / "series.json", series)

    cv = CV.load_from_file(config.paths.dimensions_cv)
    result = _Result(scratch_dir)

    with count_statements(database) as counter:
        ingest_series_values(database=database, result=result, execution=execution, cv=cv)
        database.session.commit()
    return counter


# Statement counts are a small constant: one SELECT for existing axes, one bulk INSERT
# for the missing axes, one SELECT to read their ids, and one bulk INSERT for the series
# rows. A little head-room is allowed for incidental statements.
_MAX_INGEST_STATEMENTS = 8


class TestAxisDedupHotspot:
    """Guard the batched axis resolution against regressing to a per-axis N+1."""

    @pytest.mark.filterwarnings("ignore:Unknown dimension values.*:UserWarning")
    def test_distinct_axes_do_not_scale_statements(self, db, config, tmp_path):
        """3000 distinct-index series must not issue O(n) SQL statements."""
        n = 3000
        execution = _make_execution(db, "distinct")

        counter = _write_and_ingest(db, config, execution, _distinct_axis_series(n), tmp_path)

        # The N+1 version issued ~2n+1 statements (a SELECT + INSERT per axis plus the
        # series insert); the batched version collapses that to a small constant.
        assert counter["total"] <= _MAX_INGEST_STATEMENTS, counter

        # Every series and every distinct axis is still persisted exactly once.
        assert db.session.scalar(select(func.count()).select_from(SeriesMetricValue)) == n
        assert db.session.scalar(select(func.count()).select_from(SeriesIndex)) == n

    @pytest.mark.filterwarnings("ignore:Unknown dimension values.*:UserWarning")
    def test_statement_count_is_independent_of_batch_size(self, db, config, tmp_path):
        """Doubling the number of distinct axes must not change the statement count."""
        small = _write_and_ingest(
            db, config, _make_execution(db, "small"), _distinct_axis_series(500), tmp_path
        )
        large = _write_and_ingest(
            db, config, _make_execution(db, "large"), _distinct_axis_series(1000), tmp_path
        )

        assert small["total"] == large["total"], (small, large)
        assert large["total"] <= _MAX_INGEST_STATEMENTS, large

    @pytest.mark.filterwarnings("ignore:Unknown dimension values.*:UserWarning")
    def test_shared_axis_matches_distinct_axis_cost(self, db, config, tmp_path):
        """A batch sharing one axis costs the same handful of statements as many axes."""
        shared = _write_and_ingest(
            db, config, _make_execution(db, "shared"), _shared_axis_series(3000), tmp_path
        )
        distinct = _write_and_ingest(
            db, config, _make_execution(db, "distinct"), _distinct_axis_series(3000), tmp_path
        )

        assert shared["total"] <= _MAX_INGEST_STATEMENTS, shared
        assert distinct["total"] <= _MAX_INGEST_STATEMENTS, distinct
        # One shared axis is stored once; 3000 distinct axes stored once each.
        assert db.session.scalar(select(func.count()).select_from(SeriesIndex)) == 3001

    @pytest.mark.filterwarnings("ignore:Unknown dimension values.*:UserWarning")
    def test_existing_axes_are_reused_without_reinsert(self, db, config, tmp_path):
        """A re-solve over identical axes reuses them: no new index_axis rows, no axis INSERT."""
        first = _make_execution(db, "first")
        _write_and_ingest(db, config, first, _distinct_axis_series(500), tmp_path)
        axes_after_first = db.session.scalar(select(func.count()).select_from(SeriesIndex))
        db.session.commit()  # close the read transaction before starting the next execution

        second = _make_execution(db, "second")
        counter = _write_and_ingest(db, config, second, _distinct_axis_series(500), tmp_path)

        # No axes were duplicated across executions...
        assert db.session.scalar(select(func.count()).select_from(SeriesIndex)) == axes_after_first
        # ...and the only INSERT this time was the series rows themselves (axes were reused).
        assert counter["insert"] == 1, counter
        assert counter["total"] <= _MAX_INGEST_STATEMENTS, counter
