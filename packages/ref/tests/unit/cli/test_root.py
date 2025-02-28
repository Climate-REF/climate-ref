import re

import pytest

from cmip_ref import __version__
from cmip_ref.cli import build_app
from cmip_ref_core import __version__ as __core_version__


def test_without_subcommand(invoke_cli):
    result = invoke_cli([])
    assert "Usage:" in result.stdout
    assert "cmip_ref [OPTIONS] COMMAND [ARGS]" in result.stdout
    assert "cmip_ref: A CLI for the CMIP Rapid Evaluation Framework" in result.stdout


def test_version(invoke_cli):
    result = invoke_cli(["--version"])
    assert f"cmip_ref: {__version__}\ncmip_ref-core: {__core_version__}" in result.stdout


def test_verbose(invoke_cli):
    exp_log = r"\| DEBUG    \| cmip_ref\.config:default:\d+ - Loading default configuration from"
    result = invoke_cli(
        ["--verbose", "config", "list"],
    )
    assert re.search(exp_log, result.stderr)

    result = invoke_cli(
        ["config", "list"],
    )
    # Only info and higher messages logged
    assert not re.search(exp_log, result.stderr)


def test_config_directory_custom(config, invoke_cli):
    config.paths.scratch = "test-value"
    config.save()

    result = invoke_cli(
        [
            "--configuration-directory",
            str(config._config_file.parent),
            "config",
            "list",
        ],
    )
    assert 'scratch = "test-value"\n' in result.output


def test_config_directory_append(config, invoke_cli):
    # configuration directory must be passed before command
    invoke_cli(
        [
            "config",
            "list",
            "--configuration-directory",
            str(config._config_file.parent),
        ],
        expected_exit_code=2,
    )


@pytest.fixture()
def expected_groups() -> set[str]:
    return {"config", "datasets", "executions", "celery"}


def test_build_app(expected_groups):
    app = build_app()

    registered_commands = [command.name for command in app.registered_commands]
    registered_groups = [group.name for group in app.registered_groups]

    assert registered_commands == ["solve"]
    assert set(registered_groups) == expected_groups


def test_build_app_without_celery(mocker, expected_groups):
    mocker.patch("cmip_ref.cli.importlib.import_module", side_effect=ModuleNotFoundError)
    app = build_app()

    registered_commands = [command.name for command in app.registered_commands]
    registered_groups = [group.name for group in app.registered_groups]

    assert ["solve"] == registered_commands
    assert set(registered_groups) == expected_groups - {"celery"}
