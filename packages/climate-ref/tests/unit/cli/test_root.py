import re
from pathlib import Path

import pytest
from rich.console import Console

from climate_ref import __version__
from climate_ref.cli import CLIContext, build_app
from climate_ref_core import __version__ as __core_version__


def escape_ansi(line):
    ansi_escape = re.compile(r"(?:\x1B[@-_]|[\x80-\x9F])[0-?]*[ -/]*[@-~]")
    return ansi_escape.sub("", line)


def test_without_subcommand(invoke_cli):
    result = invoke_cli([])
    assert "Usage:" in result.stdout
    assert "ref [OPTIONS] COMMAND [ARGS]" in result.stdout
    assert "A CLI for the Assessment Fast Track Rapid Evaluation Framework" in result.stdout


def test_version(invoke_cli):
    result = invoke_cli(["--version"])
    assert f"climate_ref: {__version__}\nclimate_ref-core: {__core_version__}" in result.stdout


def test_verbose(invoke_cli):
    exp_log = r"\| DEBUG    \| climate_ref\.cli - Configuration loaded from"
    result = invoke_cli(
        ["--verbose", "config", "list"],
    )
    assert re.search(exp_log, escape_ansi(result.stderr))

    result = invoke_cli(
        ["config", "list"],
    )
    # Only info and higher messages logged
    assert not re.search(exp_log, escape_ansi(result.stderr))


@pytest.mark.parametrize(
    "cmds, expected_log_level",
    [
        [["--log-level", "DEBUG"], "DEBUG"],
        [["--log-level", "INFO"], "INFO"],
        [["--log-level", "WARNING"], "WARNING"],
        [["--log-level", "ERROR"], "ERROR"],
        [["-v"], "DEBUG"],
        [["-q"], "WARNING"],
        # Verbose wins
        [["-v", "-q"], "DEBUG"],
        [["-q", "-v"], "DEBUG"],
        # -q/-v wins over --log-level
        [["-v", "--log-level", "ERROR"], "DEBUG"],
        [["-q", "--log-level", "INFO"], "WARNING"],
    ],
)
def test_log_level(invoke_cli, cmds, expected_log_level):
    result = invoke_cli(
        [*cmds, "config", "list"],
    )
    assert f'log_level = "{expected_log_level}"' in result.stdout


def test_config_directory_custom(config, invoke_cli):
    config.paths.scratch = "test-value"
    config.save()

    # The loaded value is converted into an absolute path
    expected_value = Path("test-value").resolve()

    result = invoke_cli(
        [
            "--configuration-directory",
            str(config._config_file.parent),
            "config",
            "list",
        ],
    )
    assert f'scratch = "{expected_value}"\n' in result.output


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
    return {"config", "datasets", "executions", "providers", "celery", "test-cases"}


def test_build_app(expected_groups):
    app = build_app()

    registered_commands = [command.name for command in app.registered_commands]
    registered_groups = [group.name for group in app.registered_groups]

    assert registered_commands == ["solve"]
    assert set(registered_groups) == expected_groups


def test_build_app_without_celery(mocker, expected_groups):
    mocker.patch("climate_ref.cli.importlib.import_module", side_effect=ModuleNotFoundError)
    app = build_app()

    registered_commands = [command.name for command in app.registered_commands]
    registered_groups = [group.name for group in app.registered_groups]

    assert ["solve"] == registered_commands
    assert set(registered_groups) == expected_groups - {"celery"}


def test_cli_context_lazy_database(config, mocker):
    """Test that CLIContext creates database lazily on first access."""
    # Mock Database.from_config to verify it's called lazily
    mock_from_config = mocker.patch("climate_ref.cli.Database.from_config")
    mock_db = mocker.MagicMock()
    mock_from_config.return_value = mock_db

    # Create context - database should NOT be created yet
    ctx = CLIContext(config=config, console=Console())
    mock_from_config.assert_not_called()

    # Access database - NOW it should be created
    db = ctx.database
    mock_from_config.assert_called_once_with(config, skip_backup=False)
    assert db is mock_db

    # Second access should return same instance, not create new one
    db2 = ctx.database
    assert db2 is mock_db
    mock_from_config.assert_called_once()  # Still only called once


def test_cli_context_skip_backup(config, mocker):
    """Test that CLIContext passes skip_backup to Database.from_config."""
    mock_from_config = mocker.patch("climate_ref.cli.Database.from_config")
    mock_db = mocker.MagicMock()
    mock_from_config.return_value = mock_db

    # Create context with skip_backup=True
    ctx = CLIContext(config=config, console=Console(), skip_backup=True)
    _ = ctx.database

    mock_from_config.assert_called_once_with(config, skip_backup=True)


def test_cli_context_close_without_database(config):
    """Test that CLIContext.close() works when database was never accessed."""
    ctx = CLIContext(config=config, console=Console())
    # Should not raise even though database was never created
    ctx.close()


def test_config_list_skips_database(invoke_cli, mocker):
    """Test that 'config list' doesn't access the database."""
    mock_from_config = mocker.patch("climate_ref.cli.Database.from_config")

    result = invoke_cli(["config", "list"])
    assert result.exit_code == 0

    # Database should not have been created for config list
    mock_from_config.assert_not_called()
