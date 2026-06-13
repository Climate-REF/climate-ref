"""
End-to-end failure-mode coverage for ``LocalExecutor`` plus the stale-execution
sweep performed at the start of ``solve_required_executions``.

These tests build real ``Execution`` rows in a real (sqlite) database and drive
``LocalExecutor.join`` with hand-crafted futures so we can assert on database
state for each failure mode.
"""

from __future__ import annotations

import datetime
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import pytest
from sqlalchemy import update

from climate_ref.executor.local import ExecutionFuture, LocalExecutor
from climate_ref.models import Diagnostic as DiagnosticModel
from climate_ref.models import Execution, ExecutionGroup
from climate_ref.models import Provider as ProviderModel
from climate_ref.provider_registry import _register_provider
from climate_ref.solver import fail_stale_in_progress_executions
from climate_ref_core.diagnostics import ExecutionResult
from climate_ref_core.exceptions import ExecutionError
from climate_ref_core.logging import EXECUTION_LOG_FILENAME


@pytest.fixture
def db_with_provider(db, provider):
    """A database with the mock provider and its diagnostics persisted."""
    with db.session.begin():
        _register_provider(db, provider)
    return db


@pytest.fixture
def thread_pool():
    pool = ThreadPoolExecutor(max_workers=1)
    yield pool
    pool.shutdown(wait=False)


def _seed_execution(
    db,
    provider,
    diagnostic_slug: str,
    config,
    *,
    key: str,
    dataset_hash: str = "hash",
) -> tuple[int, int, Path]:
    """Persist a pending Execution and stage its scratch directory + log file.

    handle_execution_result expects to find ``out.log`` under
    ``config.paths.scratch / execution.output_fragment``; staging it lets the
    retryable / non-retryable branches run rather than the missing-log
    early-return.
    """
    output_fragment = f"mock/{key}/{dataset_hash}"
    scratch_dir = config.paths.scratch / output_fragment
    scratch_dir.mkdir(parents=True, exist_ok=True)
    (scratch_dir / EXECUTION_LOG_FILENAME).write_text("")

    with db.session.begin():
        diagnostic_row = (
            db.session.query(DiagnosticModel)
            .join(DiagnosticModel.provider)
            .filter(
                ProviderModel.slug == provider.slug,
                DiagnosticModel.slug == diagnostic_slug,
            )
            .one()
        )
        execution_group = ExecutionGroup(
            key=key,
            diagnostic_id=diagnostic_row.id,
            dirty=True,
            selectors={},
        )
        db.session.add(execution_group)
        db.session.flush()
        execution = Execution(
            execution_group=execution_group,
            dataset_hash=dataset_hash,
            output_fragment=output_fragment,
        )
        db.session.add(execution)
        db.session.flush()
        return execution.id, execution_group.id, scratch_dir


def _attach_future(
    executor: LocalExecutor,
    definition,
    execution_id: int,
    future: Future,
    *,
    submitted_at: float | None = None,
) -> None:
    executor._results.append(
        ExecutionFuture(
            future=future,
            definition=definition,
            execution_id=execution_id,
            submitted_at=submitted_at if submitted_at is not None else time.time(),
        )
    )


def _build_executor(db, config, thread_pool, *, task_timeout: float = 0.0) -> LocalExecutor:
    return LocalExecutor(
        database=db,
        config=config,
        pool=thread_pool,
        task_timeout=task_timeout,
    )


class TestLocalExecutorFailureModes:
    """LocalExecutor.join must always reach a terminal DB state for every row."""

    def test_diagnostic_logic_failure_clears_dirty(
        self, db_with_provider, config, provider, definition_factory, mock_diagnostic, thread_pool
    ):
        executor = _build_executor(db_with_provider, config, thread_pool)
        definition = definition_factory(diagnostic=mock_diagnostic)
        execution_id, eg_id, _ = _seed_execution(
            db_with_provider, provider, "mock", config, key="logic-failure"
        )
        future: Future = Future()
        future.set_result(ExecutionResult.build_from_failure(definition, retryable=False))
        _attach_future(executor, definition, execution_id, future)

        executor.join(timeout=10)

        with db_with_provider.session.begin():
            execution = db_with_provider.session.get(Execution, execution_id)
            execution_group = db_with_provider.session.get(ExecutionGroup, eg_id)
        assert execution.successful is False
        # Logic errors are not retried automatically; group is marked clean.
        assert execution_group.dirty is False

    def test_system_failure_keeps_dirty_for_retry(
        self, db_with_provider, config, provider, definition_factory, mock_diagnostic, thread_pool
    ):
        executor = _build_executor(db_with_provider, config, thread_pool)
        definition = definition_factory(diagnostic=mock_diagnostic)
        execution_id, eg_id, _ = _seed_execution(
            db_with_provider, provider, "mock", config, key="system-failure"
        )
        future: Future = Future()
        future.set_result(ExecutionResult.build_from_failure(definition, retryable=True))
        _attach_future(executor, definition, execution_id, future)

        executor.join(timeout=10)

        with db_with_provider.session.begin():
            execution = db_with_provider.session.get(Execution, execution_id)
            execution_group = db_with_provider.session.get(ExecutionGroup, eg_id)
        assert execution.successful is False
        # System-level failure: leave dirty=True so the next solve picks it up.
        assert execution_group.dirty is True

    def test_per_task_timeout_marks_failed_retryable(
        self, db_with_provider, config, provider, definition_factory, mock_diagnostic, thread_pool
    ):
        executor = _build_executor(db_with_provider, config, thread_pool, task_timeout=0.05)
        definition = definition_factory(diagnostic=mock_diagnostic)
        execution_id, eg_id, _ = _seed_execution(
            db_with_provider, provider, "mock", config, key="per-task-timeout"
        )
        # Submit-time pre-dated so the very first join iteration triggers the
        # per-task timeout branch without depending on real wall clock.
        future: Future = Future()
        _attach_future(
            executor,
            definition,
            execution_id,
            future,
            submitted_at=time.time() - 60,
        )

        executor.join(timeout=10)

        assert future.cancelled() or future.done()
        assert executor._results == []
        with db_with_provider.session.begin():
            execution = db_with_provider.session.get(Execution, execution_id)
            execution_group = db_with_provider.session.get(ExecutionGroup, eg_id)
        assert execution.successful is False
        assert execution_group.dirty is True

    def test_overall_join_timeout_fails_outstanding(
        self, db_with_provider, config, provider, definition_factory, mock_diagnostic, thread_pool
    ):
        # task_timeout=0 disables per-task timeout so we hit the join-level branch.
        executor = _build_executor(db_with_provider, config, thread_pool, task_timeout=0)
        definition = definition_factory(diagnostic=mock_diagnostic)
        execution_id, eg_id, _ = _seed_execution(
            db_with_provider, provider, "mock", config, key="overall-timeout"
        )
        future: Future = Future()
        _attach_future(executor, definition, execution_id, future)

        with pytest.raises(TimeoutError):
            executor.join(timeout=0.2)

        assert executor._results == []
        with db_with_provider.session.begin():
            execution = db_with_provider.session.get(Execution, execution_id)
            execution_group = db_with_provider.session.get(ExecutionGroup, eg_id)
        assert execution.successful is False
        # Overall timeout is treated as a system-level failure -> retryable.
        assert execution_group.dirty is True

    def test_future_exception_marks_failed_then_raises(
        self, db_with_provider, config, provider, definition_factory, mock_diagnostic, thread_pool
    ):
        executor = _build_executor(db_with_provider, config, thread_pool)
        definition = definition_factory(diagnostic=mock_diagnostic)
        execution_id, eg_id, _ = _seed_execution(
            db_with_provider, provider, "mock", config, key="future-exception"
        )
        future: Future = Future()
        future.set_exception(RuntimeError("worker exploded"))
        _attach_future(executor, definition, execution_id, future)

        with pytest.raises(ExecutionError):
            executor.join(timeout=5)

        assert executor._results == []
        with db_with_provider.session.begin():
            execution = db_with_provider.session.get(Execution, execution_id)
            execution_group = db_with_provider.session.get(ExecutionGroup, eg_id)
        assert execution.successful is False
        # Worker explosions are infrastructure failures -> retryable.
        assert execution_group.dirty is True

    def test_missing_log_file_keeps_dirty_for_retry(
        self, db_with_provider, config, provider, definition_factory, mock_diagnostic, thread_pool
    ):
        # Worker crashed before writing out.log -- handle_execution_result must
        # mark the execution failed but leave dirty=True so the next solve retries.
        executor = _build_executor(db_with_provider, config, thread_pool)
        definition = definition_factory(diagnostic=mock_diagnostic)
        execution_id, eg_id, scratch_dir = _seed_execution(
            db_with_provider, provider, "mock", config, key="missing-log"
        )
        # Remove the log file we staged in the seeder to simulate a crash.
        (scratch_dir / EXECUTION_LOG_FILENAME).unlink()
        future: Future = Future()
        future.set_result(ExecutionResult.build_from_failure(definition, retryable=False))
        _attach_future(executor, definition, execution_id, future)

        executor.join(timeout=10)

        with db_with_provider.session.begin():
            execution = db_with_provider.session.get(Execution, execution_id)
            execution_group = db_with_provider.session.get(ExecutionGroup, eg_id)
        assert execution.successful is False
        # Even though the result claimed retryable=False, the missing-log path
        # treats this as a system failure so the row is retried next solve.
        assert execution_group.dirty is True


class TestStaleExecutionSweep:
    """``fail_stale_in_progress_executions`` reaps abandoned in-progress rows."""

    def _backdate(self, db, execution_id: int, hours: int) -> None:
        # TODO: Using a naive UTC datetime
        old = datetime.datetime.now(datetime.timezone.utc).replace(tzinfo=None) - datetime.timedelta(
            hours=hours
        )
        with db.session.begin():
            db.session.execute(update(Execution).where(Execution.id == execution_id).values(created_at=old))

    def test_marks_old_in_progress_execution_failed(self, db_with_provider, provider, config):
        execution_id, eg_id, _ = _seed_execution(db_with_provider, provider, "mock", config, key="stale-old")
        self._backdate(db_with_provider, execution_id, hours=12)

        marked = fail_stale_in_progress_executions(db_with_provider)

        assert marked == 1
        with db_with_provider.session.begin():
            execution = db_with_provider.session.get(Execution, execution_id)
            execution_group = db_with_provider.session.get(ExecutionGroup, eg_id)
        assert execution.successful is False
        # Sweep deliberately does not flip dirty; ExecutionGroup.should_run uses
        # the existing dirty=True flag (set during seeding) to retry.
        assert execution_group.dirty is True

    def test_leaves_recent_in_progress_execution_alone(self, db_with_provider, provider, config):
        execution_id, _, _ = _seed_execution(db_with_provider, provider, "mock", config, key="stale-recent")
        # No backdate: execution was just created.

        marked = fail_stale_in_progress_executions(db_with_provider)

        assert marked == 0
        with db_with_provider.session.begin():
            execution = db_with_provider.session.get(Execution, execution_id)
        assert execution.successful is None

    def test_respects_custom_threshold(self, db_with_provider, provider, config):
        execution_id, _, _ = _seed_execution(db_with_provider, provider, "mock", config, key="stale-custom")
        self._backdate(db_with_provider, execution_id, hours=2)

        # 1h cutoff -> the 2h-old row is stale and gets reaped.
        marked = fail_stale_in_progress_executions(db_with_provider, stale_after_seconds=3600)

        assert marked == 1
        with db_with_provider.session.begin():
            execution = db_with_provider.session.get(Execution, execution_id)
        assert execution.successful is False

    def test_does_not_touch_already_completed_executions(self, db_with_provider, provider, config):
        execution_id, _, _ = _seed_execution(db_with_provider, provider, "mock", config, key="stale-done")
        with db_with_provider.session.begin():
            execution = db_with_provider.session.get(Execution, execution_id)
            execution.successful = True
        self._backdate(db_with_provider, execution_id, hours=24)

        marked = fail_stale_in_progress_executions(db_with_provider)

        assert marked == 0
        with db_with_provider.session.begin():
            execution = db_with_provider.session.get(Execution, execution_id)
        assert execution.successful is True
