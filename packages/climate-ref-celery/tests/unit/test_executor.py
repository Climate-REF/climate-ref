import pytest
from climate_ref_celery.executor import CeleryExecutor
from climate_ref_celery.routing import ROUTES_ENV_VAR, RoutingTableError
from climate_ref_celery.worker_tasks import handle_failure, handle_result


@pytest.mark.parametrize("include_execution_result", [True, False])
def test_run_metric(provider, config, mock_diagnostic, metric_definition, mocker, include_execution_result):
    executor = CeleryExecutor(config=config)
    mock_app = mocker.patch("climate_ref_celery.executor.app")
    mock_execution_result = mocker.MagicMock()

    if include_execution_result:
        executor.run(metric_definition, mock_execution_result)

        mock_app.send_task.assert_called_once_with(
            "mock_provider.mock",
            args=[metric_definition, "INFO"],
            link=handle_result.s(execution_id=mock_execution_result.id).set(queue="celery"),
            link_error=handle_failure.s(execution_id=mock_execution_result.id).set(queue="celery"),
            queue="mock_provider",
        )
    else:
        executor.run(metric_definition, None)

        mock_app.send_task.assert_called_once_with(
            "mock_provider.mock",
            args=[metric_definition, "INFO"],
            link=None,
            link_error=None,
            queue="mock_provider",
        )

    assert executor._results == [mock_app.send_task.return_value]


def test_run_routed_queue(config, metric_definition, mocker, monkeypatch, tmp_path):
    routes = tmp_path / "routes.toml"
    routes.write_text('[mock_provider]\nrules = [{ match = "mock", size = "large" }]\n')
    monkeypatch.setenv(ROUTES_ENV_VAR, str(routes))

    executor = CeleryExecutor(config=config)
    mock_app = mocker.patch("climate_ref_celery.executor.app")

    executor.run(metric_definition, None)

    assert mock_app.send_task.call_args.kwargs["queue"] == "mock_provider-large"


def test_run_routed_queue_no_match(config, metric_definition, mocker, monkeypatch, tmp_path):
    routes = tmp_path / "routes.toml"
    routes.write_text('[mock_provider]\nrules = [{ match = "other-*", size = "large" }]\n')
    monkeypatch.setenv(ROUTES_ENV_VAR, str(routes))

    executor = CeleryExecutor(config=config)
    mock_app = mocker.patch("climate_ref_celery.executor.app")

    executor.run(metric_definition, None)

    assert mock_app.send_task.call_args.kwargs["queue"] == "mock_provider"


def test_malformed_routes_fails_construction(config, monkeypatch, tmp_path):
    routes = tmp_path / "routes.toml"
    routes.write_text("default = 3\n")
    monkeypatch.setenv(ROUTES_ENV_VAR, str(routes))

    with pytest.raises(RoutingTableError):
        CeleryExecutor(config=config)


def test_log_submission_summary(config, metric_definition, mocker, monkeypatch, tmp_path, caplog):
    routes = tmp_path / "routes.toml"
    routes.write_text('[mock_provider]\nrules = [{ match = "mock", size = "large" }]\n')
    monkeypatch.setenv(ROUTES_ENV_VAR, str(routes))

    executor = CeleryExecutor(config=config)
    mocker.patch("climate_ref_celery.executor.app")

    executor.run(metric_definition, None)
    executor.run(metric_definition, None)
    executor.log_submission_summary()

    assert "Submitted 2 executions to queue mock_provider-large" in caplog.text


def test_join_empty():
    executor = CeleryExecutor(config=None)

    executor.join(1)


def test_join_returns_on_completion(mocker):
    executor = CeleryExecutor(config=None)
    result = mocker.Mock()
    result.ready.return_value = True
    executor._results = [result]

    executor.join(2)

    assert len(executor._results) == 0


def test_join_raises(mocker):
    executor = CeleryExecutor(config=None)  # type: ignore
    result = mocker.Mock()
    result.ready.return_value = False
    executor._results = [result]

    with pytest.raises(TimeoutError):
        executor.join(0.1)
