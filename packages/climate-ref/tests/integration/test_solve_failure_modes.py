"""
Solve and execute failure modes that leave the database in a wrong state.

Each test asserts the correct terminal state for one failure path.
"""

from __future__ import annotations

import contextlib
import time
from collections.abc import Iterator
from concurrent.futures import Future
from unittest import mock

import pytest

from climate_ref.config import ExecutorConfig
from climate_ref.executor import result_handling
from climate_ref.executor.hpc import HPCExecutor
from climate_ref.executor.local import LocalExecutor
from climate_ref.executor.result_handling import (
    ExecutionFuture,
    handle_execution_result,
    mark_execution_failed,
)
from climate_ref.models import Execution, ExecutionGroup
from climate_ref.provider_registry import ProviderRegistry
from climate_ref.solver import DiagnosticExecution, ExecutionSolver, solve_required_executions
from climate_ref_core.diagnostics import ExecutionResult
from climate_ref_core.logging import EXECUTION_LOG_FILENAME

from .test_executor_failure_modes import _attach_future, _build_executor, _seed_execution


def _state(db, execution_id: int, group_id: int) -> tuple[Execution, ExecutionGroup]:
    with db.session.begin():
        return db.session.get(Execution, execution_id), db.session.get(ExecutionGroup, group_id)


class _CollectingExecutor:
    """Collects submissions and fails them on ``join``, as a real executor reaps its queue."""

    name = "collecting"
    collects_results_on_join = True

    def __init__(self, db, config):
        self.db = db
        self.config = config
        self.submitted: list = []
        self.joined = False

    def run(self, definition, execution=None):
        self.submitted.append((definition, execution.id))

    def join(self, timeout):
        self.joined = True
        for definition, execution_id in self.submitted:
            mark_execution_failed(self.db, self.config, definition, execution_id, retryable=True)


@pytest.fixture
def potential_execution(db_with_provider, provider, definition_factory, mock_diagnostic):
    execution = mock.MagicMock(spec=DiagnosticExecution)
    execution.provider = provider
    execution.diagnostic = mock_diagnostic
    execution.selectors = {"cmip6": (("source_id", "Test"),)}
    datasets = mock.Mock(hash="123456", items=mock.Mock(return_value=[]))
    execution.build_execution_definition.return_value = definition_factory(
        diagnostic=mock_diagnostic, execution_dataset_collection=datasets
    )
    return execution


def test_broken_diagnostic_does_not_orphan_submitted_execution(
    mocker, db_with_provider, config, provider, potential_execution
):
    def fake_solve_executions(data_catalog, diagnostic, provider) -> Iterator:
        if diagnostic.slug == "failed":
            raise ValueError("constraint blew up")
        yield potential_execution

    mocker.patch("climate_ref.solver.solve_executions", side_effect=fake_solve_executions)
    executor = _CollectingExecutor(db_with_provider, config)
    mocker.patch.object(ExecutorConfig, "build", return_value=executor)
    solver = ExecutionSolver(provider_registry=ProviderRegistry(providers=[provider]), data_catalog={})

    with contextlib.suppress(ValueError):
        solve_required_executions(db_with_provider, solver=solver, config=config)

    assert len(executor.submitted) == 1
    execution_id = executor.submitted[0][1]
    with db_with_provider.session.begin():
        execution = db_with_provider.session.get(Execution, execution_id)
    assert execution.successful is not None


def test_failure_recording_survives_permission_error_on_copy(
    mocker, db_with_provider, config, provider, definition_factory, mock_diagnostic, thread_pool
):
    mocker.patch.object(result_handling, "copy_output_file", side_effect=PermissionError("read-only"))
    executor = LocalExecutor(database=db_with_provider, config=config, pool=thread_pool, task_timeout=0.0)
    definition = definition_factory(diagnostic=mock_diagnostic)
    execution_id, eg_id, _ = _seed_execution(db_with_provider, provider, "mock", config, key="perm-error")
    future: Future = Future()
    future.set_result(ExecutionResult.build_from_failure(definition, retryable=False))
    _attach_future(executor, definition, execution_id, future)

    with contextlib.suppress(PermissionError):
        executor.join(timeout=10)

    execution, _ = _state(db_with_provider, execution_id, eg_id)
    assert execution.successful is False


def test_hpc_join_keyboard_interrupt_fails_outstanding(
    tmp_path, db_with_provider, config, provider, definition_factory, mock_diagnostic
):
    executor = _hpc_executor(db_with_provider, config, tmp_path)
    definition = definition_factory(diagnostic=mock_diagnostic)
    execution_id, eg_id, _ = _seed_execution(db_with_provider, provider, "mock", config, key="hpc-interrupt")
    future = mock.MagicMock()
    future.running.side_effect = KeyboardInterrupt
    executor.parsl_results = [ExecutionFuture(future, definition=definition, execution_id=execution_id)]

    with pytest.raises(KeyboardInterrupt):
        executor.join(timeout=10)

    execution, _ = _state(db_with_provider, execution_id, eg_id)
    assert execution.successful is False
    assert "was not collected" in _results_log_text(config, "hpc-interrupt")


@pytest.mark.parametrize("failure", ["copy", "ingest"])
def test_failed_rerun_of_clean_group_is_marked_dirty(
    mocker, db_with_provider, config, provider, definition_factory, mock_diagnostic, failure
):
    execution_id, eg_id, _ = _seed_execution(
        db_with_provider, provider, "mock", config, key=f"clean-{failure}", dirty=False
    )
    definition = definition_factory(diagnostic=mock_diagnostic)
    result = ExecutionResult(
        definition=definition,
        successful=True,
        output_bundle_filename=None,
        metric_bundle_filename=definition.output_directory / "diagnostic.json",
    )
    if failure == "copy":
        mocker.patch.object(
            result_handling, "copy_execution_outputs", side_effect=FileNotFoundError("plot.png")
        )
    else:
        mocker.patch.object(result_handling, "copy_execution_outputs")
        mocker.patch.object(result_handling, "ingest_execution_result", side_effect=ValueError("bad bundle"))

    with db_with_provider.session.begin():
        execution = db_with_provider.session.get(Execution, execution_id)
        handle_execution_result(config, db_with_provider, execution, result)

    execution, group = _state(db_with_provider, execution_id, eg_id)
    assert execution.successful is False
    assert group.dirty is True


def _results_log_text(config, key: str) -> str:
    return (config.paths.results / "mock" / key / "hash" / EXECUTION_LOG_FILENAME).read_text()


def test_mark_execution_failed_writes_reason_to_results_log(
    db_with_provider, config, provider, definition_factory, mock_diagnostic
):
    definition = definition_factory(diagnostic=mock_diagnostic)
    execution_id, _, _ = _seed_execution(db_with_provider, provider, "mock", config, key="reason")

    mark_execution_failed(db_with_provider, config, definition, execution_id, retryable=True, detail="why")

    assert "): why" in _results_log_text(config, "reason")


def test_local_task_timeout_reason(
    db_with_provider, config, provider, definition_factory, mock_diagnostic, thread_pool
):
    executor = _build_executor(db_with_provider, config, thread_pool, task_timeout=1.0)
    definition = definition_factory(diagnostic=mock_diagnostic)
    execution_id, _, _ = _seed_execution(db_with_provider, provider, "mock", config, key="task-timeout")
    future: Future = Future()
    future.set_running_or_notify_cancel()
    _attach_future(executor, definition, execution_id, future, started_at=time.time() - 100)

    executor.join(timeout=10)

    assert "per-task timeout of 1.0s" in _results_log_text(config, "task-timeout")


def test_local_overall_timeout_reason(
    db_with_provider, config, provider, definition_factory, mock_diagnostic, thread_pool
):
    executor = _build_executor(db_with_provider, config, thread_pool, task_timeout=0.0)
    definition = definition_factory(diagnostic=mock_diagnostic)
    execution_id, _, _ = _seed_execution(db_with_provider, provider, "mock", config, key="overall-timeout")
    _attach_future(executor, definition, execution_id, Future())

    with pytest.raises(TimeoutError):
        executor.join(timeout=0.001)

    assert "overall timeout of 0.001s" in _results_log_text(config, "overall-timeout")


def test_local_interrupt_reason(
    db_with_provider, config, provider, definition_factory, mock_diagnostic, thread_pool
):
    executor = _build_executor(db_with_provider, config, thread_pool, task_timeout=0.0)
    definition = definition_factory(diagnostic=mock_diagnostic)
    execution_id, _, _ = _seed_execution(db_with_provider, provider, "mock", config, key="interrupt")
    future = mock.MagicMock()
    future.running.side_effect = KeyboardInterrupt
    _attach_future(executor, definition, execution_id, future)

    with pytest.raises(KeyboardInterrupt):
        executor.join(timeout=10)

    assert "was not collected" in _results_log_text(config, "interrupt")


def _hpc_executor(db, config, tmp_path) -> HPCExecutor:
    return HPCExecutor(
        database=db,
        config=config,
        scheduler="slurm",
        account="myaccount",
        username="myname",
        qos="myqos",
        partition="mypartition",
        log_dir=str(tmp_path / "parsl_runinfo"),
    )


def test_hpc_system_failure_reason(
    tmp_path, db_with_provider, config, provider, definition_factory, mock_diagnostic
):
    executor = _hpc_executor(db_with_provider, config, tmp_path)
    definition = definition_factory(diagnostic=mock_diagnostic)
    execution_id, _, _ = _seed_execution(db_with_provider, provider, "mock", config, key="hpc-system")
    future: Future = Future()
    future.set_exception(OSError("walltime exceeded"))
    executor.parsl_results = [ExecutionFuture(future, definition=definition, execution_id=execution_id)]

    executor.join(timeout=10)

    trailer = _results_log_text(config, "hpc-system")
    assert "system-level failure" in trailer
    assert "walltime exceeded" in trailer


def test_submit_failure_reason(mocker, db_with_provider, config, provider, potential_execution):
    executor = mock.MagicMock()
    executor.run.side_effect = RuntimeError("queue is down")
    mocker.patch.object(ExecutorConfig, "build", return_value=executor)
    mark_failed = mocker.patch("climate_ref.solver.mark_execution_failed")
    solver = mock.MagicMock(spec=ExecutionSolver)
    solver.solve.return_value = [potential_execution]

    solve_required_executions(db_with_provider, solver=solver, config=config)

    assert "queue is down" in mark_failed.call_args.kwargs["detail"]


def test_local_join_does_not_claim_success_when_all_failed(
    db_with_provider, config, provider, definition_factory, mock_diagnostic, thread_pool, caplog
):
    executor = _build_executor(db_with_provider, config, thread_pool, task_timeout=0.0)
    definition = definition_factory(diagnostic=mock_diagnostic)
    execution_id, _, _ = _seed_execution(db_with_provider, provider, "mock", config, key="all-failed")
    future: Future = Future()
    future.set_result(ExecutionResult.build_from_failure(definition, retryable=False))
    _attach_future(executor, definition, execution_id, future)

    with caplog.at_level("INFO"):
        executor.join(timeout=10)

    assert "completed successfully" not in caplog.text
