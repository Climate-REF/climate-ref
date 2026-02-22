import pytest

from climate_ref.executor import LocalExecutor
from climate_ref_core.datasets import ExecutionDatasetCollection
from climate_ref_core.diagnostics import ExecutionDefinition
from climate_ref_core.exceptions import DiagnosticError, InvalidExecutorException
from climate_ref_core.executor import Executor, _is_system_error, execute_locally, import_executor_cls


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
