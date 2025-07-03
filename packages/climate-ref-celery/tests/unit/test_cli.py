import importlib.metadata

import pytest
from climate_ref_celery.cli import app
from typer.testing import CliRunner

from climate_ref_core.providers import DiagnosticProvider

runner = CliRunner()


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0


@pytest.fixture
def mock_create_celery_app(mocker):
    return mocker.patch("climate_ref_celery.cli.create_celery_app")


@pytest.fixture
def mock_register_celery_tasks(mocker):
    return mocker.patch("climate_ref_celery.cli.register_celery_tasks")


def test_start_worker_success(mocker, mock_create_celery_app, mock_register_celery_tasks):
    mock_celery_app = mock_create_celery_app.return_value
    mock_provider = mocker.MagicMock(spec=DiagnosticProvider)
    mock_provider.slug = "example"

    mock_entry_point = mocker.Mock(spec=importlib.metadata.EntryPoint)
    mock_entry_point.dist.name = "test_package"
    mock_entry_point.load.return_value = mock_provider
    mock_entry_points = mocker.patch("importlib.metadata.entry_points", return_value=[mock_entry_point])

    result = runner.invoke(app, ["start-worker", "--package", "test_package"])

    assert result.exit_code == 0
    mock_entry_points.assert_called_once_with(group="climate-ref.providers")
    mock_register_celery_tasks.assert_called_once_with(mock_create_celery_app.return_value, mock_provider)
    mock_celery_app.worker_main.assert_called_once_with(
        argv=["worker", "-E", "--loglevel=info", "--queues=example"]
    )


def test_start_core_worker_success(mock_create_celery_app, mock_register_celery_tasks):
    mock_celery_app = mock_create_celery_app.return_value

    result = runner.invoke(app, ["start-worker"])

    assert result.exit_code == 0
    mock_celery_app.worker_main.assert_called_once_with(
        argv=["worker", "-E", "--loglevel=info", "--queues=celery"]
    )


def test_start_worker_success_extra_args(mocker, mock_create_celery_app, mock_register_celery_tasks):
    mock_worker_main = mock_create_celery_app.return_value
    mock_provider = mocker.MagicMock(spec=DiagnosticProvider)
    mock_provider.slug = "example"

    mock_entry_point = mocker.Mock(spec=importlib.metadata.EntryPoint)
    mock_entry_point.dist.name = "test_package"
    mock_entry_point.load.return_value = mock_provider
    mocker.patch("importlib.metadata.entry_points", return_value=[mock_entry_point])

    result = runner.invoke(
        app,
        [
            "start-worker",
            "--loglevel",
            "error",
            "--package",
            "test_package",
            "--",
            "--extra-args",
            "--concurrency=2",
        ],
    )

    assert result.exit_code == 0, result.output
    mock_worker_main.worker_main.assert_called_once_with(
        argv=["worker", "-E", "--loglevel=error", "--queues=example", "--extra-args", "--concurrency=2"]
    )


def test_start_worker_package_not_registered(mocker, mock_create_celery_app):
    mocker.patch("importlib.metadata.entry_points", return_value=[])

    result = runner.invoke(app, ["start-worker", "--package", "unregistered_package"])

    assert result.exit_code == 1
    assert "Package 'unregistered_package' is missing" in result.output
    mock_create_celery_app.assert_called_once_with("climate_ref_celery")


def test_start_worker_package_not_found(mocker, mock_create_celery_app):
    mock_entry_point = mocker.Mock(spec=importlib.metadata.EntryPoint)
    mock_entry_point.dist.name = "missing_package"
    mock_entry_point.load.side_effect = ModuleNotFoundError
    mock_entry_points = mocker.patch("importlib.metadata.entry_points", return_value=[mock_entry_point])

    result = runner.invoke(app, ["start-worker", "--package", "missing_package"])

    assert result.exit_code == 1
    assert "Package 'missing_package' not found" in result.output
    mock_create_celery_app.assert_called_once_with("climate_ref_celery")
    mock_entry_points.assert_called_once_with(group="climate-ref.providers")


def test_start_worker_missing_provider(mocker, mock_create_celery_app):
    mock_entry_point = mocker.Mock(spec=importlib.metadata.EntryPoint)
    mock_entry_point.dist.name = "test_package"
    mock_entry_point.load.side_effect = AttributeError
    mocker.patch("importlib.metadata.entry_points", return_value=[mock_entry_point])

    result = runner.invoke(app, ["start-worker", "--package", "test_package"])

    assert result.exit_code == 1, result.output
    assert "The package must define a 'provider' attribute" in result.output


def test_start_worker_incorrect_provider(mocker, mock_create_celery_app):
    # Not a DiagnosticProvider
    mock_provider = mocker.Mock()

    mock_entry_point = mocker.Mock(spec=importlib.metadata.EntryPoint)
    mock_entry_point.dist.name = "test_package"
    mock_entry_point.load.return_value = mock_provider
    mocker.patch("importlib.metadata.entry_points", return_value=[mock_entry_point])

    result = runner.invoke(app, ["start-worker", "--package", "test_package"])

    assert result.exit_code == 1, result.output
    assert "Expected DiagnosticProvider, got <class 'unittest.mock.Mock'>" in result.output


def test_list_config():
    result = runner.invoke(app, ["list-config"])

    assert result.exit_code == 0, result.output
    assert "broker_url: 'redis://localhost:6379/1'" in result.stdout
