from ref_celery.cli import app
from typer.testing import CliRunner

runner = CliRunner()


def test_cli_help():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0


def test_cli_spawns_worker(mocker):
    mock_app = mocker.patch("ref_celery.cli.create_celery_app")
    result = runner.invoke(app, ["--package", "ref-metrics-example"])
    assert result.exit_code == 0

    mock_app().worker_main.assert_called_once()


def test_cli_wrong_package():
    result = runner.invoke(app, ["--package", "missing"])
    assert result.exit_code == 1

    print(result.output)
    assert "Package 'missing' not found" in result.output


def test_cli_missing_provider():
    result = runner.invoke(app, ["--package", "pandas"])
    assert result.exit_code == 1

    print(result.output)
    assert "The package must define a 'provider' variable" in result.output
