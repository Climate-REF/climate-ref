import concurrent.futures
import re
from concurrent.futures import Future

import pytest

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

    def test_run_metric(self, metric_definition, provider, mock_diagnostic, mocker, caplog):
        process_pool = mocker.MagicMock(spec=concurrent.futures.ProcessPoolExecutor)
        executor = LocalExecutor(pool=process_pool)

        executor.run(metric_definition, None)
        assert len(executor._results) == 1
        assert executor._results[0].definition == metric_definition
        assert executor._results[0].execution_id is None

        # This directory is created by the executor
        assert process_pool.submit.call_count == 1

    def test_join(self, metric_definition):
        executor = LocalExecutor(n=1)
        future = Future()
        executor._results = [ExecutionFuture(future, definition=metric_definition, execution_id=None)]

        # Future isn't done yet
        with pytest.raises(TimeoutError):
            executor.join(0.1)

        # The executor should still have the future
        assert len(executor._results) == 1

        future.set_result(
            ExecutionResult(
                definition=metric_definition,
                successful=False,
                output_bundle_filename=None,
                metric_bundle_filename=None,
            )
        )

        executor.join(0.1)

        assert len(executor._results) == 0

    def test_join_exception(self, metric_definition):
        executor = LocalExecutor(n=1)
        future = Future()
        executor._results = [ExecutionFuture(future, definition=metric_definition, execution_id=None)]

        future.set_exception(ValueError("Some thing bad went wrong"))

        with pytest.raises(ExecutionError, match=re.escape("Failed to execute 'mock_provider/mock/key'")):
            executor.join(0.1)
