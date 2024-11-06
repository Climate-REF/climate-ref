import pytest
from ref_core.executor import Executor, ExecutorManager, run_metric
from ref_core.executor.local import LocalExecutor


class TestExecutorManager:
    def test_executor_register(self):
        manager = ExecutorManager()
        manager.register(LocalExecutor())

        assert len(manager._executors) == 1
        assert "local" in manager._executors
        assert isinstance(manager.get("local"), LocalExecutor)


class TestLocalExecutor:
    def test_is_executor(self):
        executor = LocalExecutor()

        assert executor.name == "local"
        assert isinstance(executor, Executor)

    def test_run_metric(self, configuration, mock_metric):
        executor = LocalExecutor()

        result = executor.run_metric(mock_metric, configuration, trigger=None)
        assert result.successful
        assert result.output_bundle == configuration.output_fragment / "output.json"


@pytest.mark.parametrize("executor_name", ["local", None])
def test_run_metric_local(monkeypatch, executor_name, mock_metric, provider, configuration):
    if executor_name:
        monkeypatch.setenv("REF_EXECUTOR", executor_name)
    result = run_metric("mock", provider, configuration=configuration)
    assert result.successful


def test_run_metric_unknown_executor(monkeypatch, provider):
    monkeypatch.setenv("REF_EXECUTOR", "missing")
    with pytest.raises(KeyError):
        run_metric("mock", metrics_provider=provider, kwarg="test")


def test_run_metric_unknown_metric(monkeypatch, provider):
    with pytest.raises(KeyError):
        run_metric("missing", metrics_provider=provider, kwarg="test")
