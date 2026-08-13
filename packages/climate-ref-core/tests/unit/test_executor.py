import pytest

from climate_ref.executor import LocalExecutor
from climate_ref_core.datasets import ExecutionDatasetCollection
from climate_ref_core.diagnostics import ExecutionDefinition, ExecutionResult
from climate_ref_core.exceptions import CondaCommandError, DiagnosticError, InvalidExecutorException
from climate_ref_core.executor import Executor, _is_system_error, execute_locally, import_executor_cls
from climate_ref_core.resources import ResourceUsage


@pytest.fixture
def make_definition(tmp_path):
    """Create ExecutionDefinition instances for testing in the core package"""

    def _make(diagnostic):
        return ExecutionDefinition(
            diagnostic=diagnostic,
            key="test-key",
            datasets=ExecutionDatasetCollection({}),
            root_directory=tmp_path,
            output_directory=tmp_path / "output",
        )

    return _make


@pytest.mark.parametrize(
    "import_str", ["climate_ref.executor.local.LocalExecutor", "climate_ref.executor.LocalExecutor"]
)
def test_import_executor(import_str):
    executor = import_executor_cls(import_str)

    assert isinstance(executor, Executor)
    assert executor == LocalExecutor


def test_import_executor_missing():
    fqn = "climate_ref.executor.local.WrongExecutor"
    match = f"Invalid executor: '{fqn}'\n Executor 'WrongExecutor' not found in climate_ref.executor.local"
    with pytest.raises(InvalidExecutorException, match=match):
        import_executor_cls(fqn)

    fqn = "missing.executor.local.WrongExecutor"
    match = f"Invalid executor: '{fqn}'\n Module 'missing.executor.local' not found"
    with pytest.raises(InvalidExecutorException, match=match):
        import_executor_cls(fqn)


class TestIsSystemError:
    @pytest.mark.parametrize(
        "exc",
        [
            MemoryError("out of memory"),
            OSError("disk full"),
            SystemExit(137),
            KeyboardInterrupt(),
        ],
        ids=["MemoryError", "OSError", "SystemExit", "KeyboardInterrupt"],
    )
    def test_system_errors(self, exc):
        assert _is_system_error(exc) is True

    @pytest.mark.parametrize(
        "exc",
        [
            ValueError("bad value"),
            TypeError("wrong type"),
            KeyError("missing key"),
            RuntimeError("something broke"),
            ZeroDivisionError("division by zero"),
        ],
        ids=["ValueError", "TypeError", "KeyError", "RuntimeError", "ZeroDivisionError"],
    )
    def test_diagnostic_errors(self, exc):
        assert _is_system_error(exc) is False


class TestExecuteLocally:
    def test_diagnostic_error_not_retryable(self, make_definition, mocker):
        """A ValueError from the diagnostic should produce a non-retryable failure"""
        diagnostic = mocker.Mock()
        diagnostic.run.side_effect = ValueError("bad diagnostic logic")
        definition = make_definition(diagnostic)

        result = execute_locally(definition, log_level="WARNING")

        assert result.successful is False
        assert result.retryable is False

    def test_system_error_retryable(self, make_definition, mocker):
        """A MemoryError from the diagnostic should produce a retryable failure"""
        diagnostic = mocker.Mock()
        diagnostic.run.side_effect = MemoryError("out of memory")
        definition = make_definition(diagnostic)

        result = execute_locally(definition, log_level="WARNING")

        assert result.successful is False
        assert result.retryable is True

    def test_os_error_retryable(self, make_definition, mocker):
        """An OSError (e.g. disk full) should produce a retryable failure"""
        diagnostic = mocker.Mock()
        diagnostic.run.side_effect = OSError("No space left on device")
        definition = make_definition(diagnostic)

        result = execute_locally(definition, log_level="WARNING")

        assert result.successful is False
        assert result.retryable is True

    def test_system_error_retryable_with_raise(self, make_definition, mocker):
        """When raise_error=True, system errors should still set retryable on the result"""
        diagnostic = mocker.Mock()
        diagnostic.run.side_effect = MemoryError("out of memory")
        definition = make_definition(diagnostic)

        with pytest.raises(DiagnosticError) as exc_info:
            execute_locally(definition, log_level="WARNING", raise_error=True)

        assert exc_info.value.result.retryable is True

    def test_diagnostic_error_not_retryable_with_raise(self, make_definition, mocker):
        """When raise_error=True, diagnostic errors should still set retryable=False on the result"""
        diagnostic = mocker.Mock()
        diagnostic.run.side_effect = ValueError("bad value")
        definition = make_definition(diagnostic)

        with pytest.raises(DiagnosticError) as exc_info:
            execute_locally(definition, log_level="WARNING", raise_error=True)

        assert exc_info.value.result.retryable is False


class TestExecuteLocallyResourceUsage:
    def test_default_is_none(self, make_definition, mocker):
        """A result built without the field carries no usage"""
        result = ExecutionResult.build_from_failure(make_definition(mocker.Mock()))

        assert result.resource_usage is None

    def test_successful_execution(self, make_definition, mocker):
        """A successful execution carries the usage measured around it"""
        definition = make_definition(mocker.Mock())
        diagnostic = definition.diagnostic
        diagnostic.run.return_value = ExecutionResult(definition=definition, successful=True)

        result = execute_locally(definition, log_level="WARNING")

        assert result.successful is True
        assert isinstance(result.resource_usage, ResourceUsage)
        assert result.resource_usage.wall_seconds >= 0.0

    @pytest.mark.parametrize("exclusive", [True, False])
    def test_the_executors_declaration_is_recorded(self, make_definition, mocker, exclusive):
        """
        Only the caller knows whether this process shares its cgroup, so its word is what is stored.

        Defaulting to False rather than True is the point:
        a worker that cannot see its siblings must not claim the container's memory as its own.
        """
        definition = make_definition(mocker.Mock())
        definition.diagnostic.run.return_value = ExecutionResult(definition=definition, successful=True)

        result = execute_locally(definition, log_level="WARNING", exclusive=exclusive)

        assert result.resource_usage.exclusive is exclusive
        assert result.resource_usage.context["cgroup_exclusive_declared"] is exclusive

    def test_defaults_to_a_shared_cgroup(self, make_definition, mocker):
        """A caller that says nothing gets the conservative reading."""
        definition = make_definition(mocker.Mock())
        definition.diagnostic.run.return_value = ExecutionResult(definition=definition, successful=True)

        result = execute_locally(definition, log_level="WARNING")

        assert result.resource_usage.exclusive is False

    def test_diagnostic_failure(self, make_definition, mocker):
        """A diagnostic that raises still reports what it consumed before failing"""
        diagnostic = mocker.Mock()
        diagnostic.run.side_effect = ValueError("bad diagnostic logic")
        definition = make_definition(diagnostic)

        result = execute_locally(definition, log_level="WARNING")

        assert result.successful is False
        assert isinstance(result.resource_usage, ResourceUsage)

    def test_conda_command_failure(self, make_definition, mocker):
        """The conda failure branch carries the usage too"""
        diagnostic = mocker.Mock()
        diagnostic.run.side_effect = CondaCommandError("conda failed", stdout="", stderr="boom")
        definition = make_definition(diagnostic)

        result = execute_locally(definition, log_level="WARNING")

        assert result.successful is False
        assert isinstance(result.resource_usage, ResourceUsage)

    def test_raised_error_carries_usage(self, make_definition, mocker):
        """The result attached to a raised DiagnosticError carries the usage"""
        diagnostic = mocker.Mock()
        diagnostic.run.side_effect = MemoryError("out of memory")
        definition = make_definition(diagnostic)

        with pytest.raises(DiagnosticError) as exc_info:
            execute_locally(definition, log_level="WARNING", raise_error=True)

        assert isinstance(exc_info.value.result.resource_usage, ResourceUsage)

    def test_measure_false_records_nothing(self, make_definition, mocker):
        """Turning the measurement off leaves the result without usage, and the run untouched"""
        definition = make_definition(mocker.Mock())
        definition.diagnostic.run.return_value = ExecutionResult(definition=definition, successful=True)

        result = execute_locally(definition, log_level="WARNING", measure=False)

        assert result.successful is True
        assert result.resource_usage is None

    def test_measurement_failure_does_not_fail_the_run(self, make_definition, mocker):
        """A recorder that cannot report leaves a successful result successful"""
        definition = make_definition(mocker.Mock())
        definition.diagnostic.run.return_value = ExecutionResult(definition=definition, successful=True)
        mocker.patch(
            "climate_ref_core.resources.ResourceRecorder._build_usage",
            side_effect=RuntimeError("sampler exploded"),
        )

        result = execute_locally(definition, log_level="WARNING")

        assert result.successful is True
        assert result.resource_usage is None
