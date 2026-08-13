from loguru import logger

from climate_ref.executor.synchronous import SynchronousExecutor
from climate_ref_core.executor import Executor


class TestSynchronousExecutor:
    def executor(self) -> SynchronousExecutor:
        return SynchronousExecutor()

    def test_is_executor(self):
        executor = self.executor()

        assert executor.name == "synchronous"
        assert isinstance(executor, Executor)

    def test_warns_that_peaks_can_span_executions(self, config):
        records = []
        sink_id = logger.add(lambda message: records.append(message.record), level="WARNING")
        try:
            SynchronousExecutor(config=config)
        finally:
            logger.remove(sink_id)

        assert any("earlier execution" in record["message"] for record in records)

    def test_does_not_warn_when_measurement_is_off(self, config):
        config.executor.measure_resources = False

        records = []
        sink_id = logger.add(lambda message: records.append(message.record), level="WARNING")
        try:
            SynchronousExecutor(config=config)
        finally:
            logger.remove(sink_id)

        assert not any("earlier execution" in record["message"] for record in records)

    def test_declares_the_cgroup_exclusive(self, metric_definition, mocker):
        """One execution at a time in one process, so a cgroup reading describes that execution."""
        mock_execute = mocker.patch(
            "climate_ref.executor.synchronous.execute_locally",
            return_value=mocker.MagicMock(),
        )
        mocker.patch("climate_ref.executor.synchronous.process_result")

        self.executor().run(metric_definition, None)

        assert mock_execute.call_args.kwargs["exclusive"] is True

    def test_run_metric(self, metric_definition, provider, mock_diagnostic, mocker, caplog):
        mock_handle_result = mocker.patch("climate_ref.executor.result_handling.handle_execution_result")
        mock_execution_result = mocker.MagicMock()
        executor = self.executor()

        executor.run(metric_definition, mock_execution_result)
        # This directory is created by the executor
        assert metric_definition.output_directory.exists()

        mock_handle_result.assert_called_once()
        _config, _db, metric_execution_result, result = mock_handle_result.call_args.args

        assert metric_execution_result == mock_execution_result
        assert result.successful
        assert result.output_bundle_filename == metric_definition.output_directory / "output.json"
        assert result.metric_bundle_filename == metric_definition.output_directory / "diagnostic.json"
        assert (metric_definition.output_directory / "out.log").exists()

    def test_raises_exception(self, mocker, provider, metric_definition, mock_diagnostic):
        mock_handle_result = mocker.patch("climate_ref.executor.result_handling.handle_execution_result")
        mock_execution_result = mocker.MagicMock()

        executor = self.executor()

        mock_diagnostic.run = lambda definition: 1 / 0

        executor.run(metric_definition, mock_execution_result)

        _config, _db, _metric_execution_result, result = mock_handle_result.call_args.args
        assert result.successful is False
        assert result.output_bundle_filename is None
        assert result.metric_bundle_filename is None

    def test_join_empty(self):
        executor = self.executor()

        executor.join(1)
        # This method should return immediately
