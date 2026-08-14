import concurrent.futures
import multiprocessing
import re
from concurrent.futures import Future

import pytest
from loguru import logger

from climate_ref.executor.local import ExecutionFuture, LocalExecutor, execute_locally
from climate_ref_core.diagnostics import ExecutionResult
from climate_ref_core.exceptions import ExecutionError
from climate_ref_core.executor import Executor


def test_execute_locally(definition_factory, mock_diagnostic):
    definition = definition_factory(diagnostic=mock_diagnostic)
    result = execute_locally(
        definition,
        log_level="DEBUG",
    )
    assert result.successful is True
    assert definition.output_directory.exists()


def test_execute_locally_failed(definition_factory, mock_diagnostic):
    mock_diagnostic.run = lambda definition: 1 / 0

    # execution raises an exception
    result = execute_locally(
        definition_factory(diagnostic=mock_diagnostic),
        log_level="DEBUG",
    )

    assert result.successful is False


class TestLocalExecutor:
    def test_is_executor(self):
        executor = LocalExecutor()

        assert executor.name == "local"
        assert isinstance(executor, Executor)

    def test_takes_process_pool(self):
        pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        executor = LocalExecutor(pool=pool)

        assert executor.pool == pool

    def test_a_supplied_pool_that_reuses_workers_warns(self):
        records = []
        sink_id = logger.add(lambda message: records.append(message.record), level="WARNING")
        try:
            LocalExecutor(pool=concurrent.futures.ThreadPoolExecutor(max_workers=1))
        finally:
            logger.remove(sink_id)

        assert any("earlier execution" in record["message"] for record in records)

    def test_a_supplied_single_task_pool_does_not_warn(self):
        # A real pool, so the check on the private attribute breaks loudly if its name ever changes.
        pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=1,
            mp_context=multiprocessing.get_context("spawn"),
            max_tasks_per_child=1,
        )

        records = []
        sink_id = logger.add(lambda message: records.append(message.record), level="WARNING")
        try:
            LocalExecutor(pool=pool)
        finally:
            logger.remove(sink_id)
            pool.shutdown(wait=False)

        assert not any("earlier execution" in record["message"] for record in records)

    def test_run_metric(self, metric_definition, provider, mock_diagnostic, mocker, caplog):
        process_pool = mocker.MagicMock(spec=concurrent.futures.ProcessPoolExecutor)
        executor = LocalExecutor(pool=process_pool)

        executor.run(metric_definition, None)
        assert len(executor._results) == 1
        assert executor._results[0].definition == metric_definition
        assert executor._results[0].execution_id is None

        # This directory is created by the executor
        assert process_pool.submit.call_count == 1

    @pytest.mark.parametrize("workers, expected", [(1, True), (2, False)])
    def test_only_a_single_worker_pool_claims_the_cgroup(self, metric_definition, workers, expected):
        """
        Every worker shares this process's cgroup, and none of them can see the others.

        So the executor is the only party that can say whether a cgroup reading
        describes one execution, and it can only say yes when it runs one at a time.
        """
        pool = concurrent.futures.ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            max_tasks_per_child=1,
        )
        try:
            executor = LocalExecutor(pool=pool)
            assert executor._exclusive is expected
        finally:
            pool.shutdown(wait=False)

    def test_the_exclusivity_declaration_reaches_the_worker(self, metric_definition, mocker):
        process_pool = mocker.MagicMock(spec=concurrent.futures.ProcessPoolExecutor)
        executor = LocalExecutor(pool=process_pool)
        executor._exclusive = True

        executor.run(metric_definition, None)

        assert process_pool.submit.call_args.kwargs["exclusive"] is True

    def test_an_unrecognised_pool_is_treated_as_concurrent(self, mocker):
        """A pool that will not say how many workers it has does not get the benefit of the doubt."""
        executor = LocalExecutor(pool=mocker.MagicMock(spec=concurrent.futures.Executor))

        assert executor._exclusive is False

    def test_join(self, metric_definition, mocker):
        executor = LocalExecutor(n=1)
        future = Future()
        executor._results = [ExecutionFuture(future, definition=metric_definition, execution_id=None)]

        # Future isn't done yet -- timeout marks it failed-retryable and clears results.
        # Failure recording flows through ``mark_execution_failed`` in
        # ``executor.result_handling``, so patch ``process_result`` there.
        process_spy = mocker.patch("climate_ref.executor.result_handling.process_result")
        with pytest.raises(TimeoutError):
            executor.join(0.1)

        assert len(executor._results) == 0
        process_spy.assert_called_once()
        forwarded = process_spy.call_args.args[2]
        assert isinstance(forwarded, ExecutionResult)
        assert forwarded.successful is False
        assert forwarded.retryable is True

    def test_join_completes(self, metric_definition, mocker):
        executor = LocalExecutor(n=1)
        future = Future()
        future.set_result(
            ExecutionResult(
                definition=metric_definition,
                successful=False,
                output_bundle_filename=None,
                metric_bundle_filename=None,
            )
        )
        executor._results = [ExecutionFuture(future, definition=metric_definition, execution_id=None)]
        mocker.patch("climate_ref.executor.local.process_result")

        executor.join(0.1)

        assert len(executor._results) == 0

    def test_join_forwards_queue_seconds(self, metric_definition, mocker):
        """Only the parent knows when the task was submitted, so it supplies the latency."""
        executor = LocalExecutor(n=1)
        future = Future()
        future.set_result(
            ExecutionResult(
                definition=metric_definition,
                successful=False,
                output_bundle_filename=None,
                metric_bundle_filename=None,
            )
        )
        executor._results = [
            ExecutionFuture(
                future,
                definition=metric_definition,
                execution_id=None,
                submitted_at=1000.0,
                started_at=1002.5,
            )
        ]
        process_spy = mocker.patch("climate_ref.executor.local.process_result")

        executor.join(0.1)

        assert process_spy.call_args.kwargs["queue_seconds"] == pytest.approx(2.5)

    def test_join_exception(self, metric_definition, mocker):
        executor = LocalExecutor(n=1)
        future = Future()
        executor._results = [ExecutionFuture(future, definition=metric_definition, execution_id=None)]

        future.set_exception(ValueError("Some thing bad went wrong"))

        process_spy = mocker.patch("climate_ref.executor.result_handling.process_result")
        with pytest.raises(ExecutionError, match=re.escape("Failed to execute 'mock_provider/mock/key'")):
            executor.join(0.1)

        # The failed execution must be flushed out of the tracker so the next
        # solve does not see a stuck successful=None row.
        assert len(executor._results) == 0
        process_spy.assert_called_once()
        forwarded = process_spy.call_args.args[2]
        assert isinstance(forwarded, ExecutionResult)
        assert forwarded.retryable is True

    def test_join_marks_outstanding_when_ingestion_fails(self, metric_definition, mocker):
        """
        An execution still in flight when ``join`` raises is recorded as retryable.

        Otherwise it keeps ``successful=None`` and only the stale-execution reaper recovers it,
        hours later.
        """
        executor = LocalExecutor(n=1)
        completed = Future()
        completed.set_result(
            ExecutionResult(
                definition=metric_definition,
                successful=True,
                output_bundle_filename=None,
                metric_bundle_filename=None,
            )
        )
        executor._results = [
            ExecutionFuture(completed, definition=metric_definition, execution_id=None),
            ExecutionFuture(Future(), definition=metric_definition, execution_id=None),
        ]

        mocker.patch("climate_ref.executor.local.process_result", side_effect=ValueError("ingest broke"))
        failure_spy = mocker.patch("climate_ref.executor.result_handling.process_result")

        with pytest.raises(ValueError, match="ingest broke"):
            executor.join(0)

        # Both are recorded: the completed one raised before it could be ingested.
        assert len(executor._results) == 0
        assert failure_spy.call_count == 2
        for call in failure_spy.call_args_list:
            assert call.args[2].successful is False
            assert call.args[2].retryable is True
