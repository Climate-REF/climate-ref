"""Entrypoint for the CLI"""

import importlib
import sys
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer
from attrs import define, field
from loguru import logger
from rich.console import Console

from climate_ref import __version__
from climate_ref.config import Config
from climate_ref.constants import CONFIG_FILENAME
from climate_ref.database import Database
from climate_ref_core import __version__ as __core_version__
from climate_ref_core.logging import initialise_logging

# Registry of read-only command paths (group, command)
# These commands skip database backup before migrations since they don't modify data
_READ_ONLY_COMMANDS: set[tuple[str, str]] = {
    ("config", "list"),
    ("datasets", "list"),
    ("datasets", "list-columns"),
    ("executions", "list-groups"),
    ("executions", "inspect"),
    ("providers", "list"),
    ("providers", "show"),
    ("test-cases", "list"),
}


def _is_read_only_command() -> bool:
    """
    Check if the current command is a read-only command by inspecting sys.argv.

    This checks against a registry of known read-only command paths since
    the Typer callback runs before nested commands are resolved.
    """
    # Parse sys.argv to find the command path
    # Skip options (--flag) and the program name
    args = [arg for arg in sys.argv[1:] if not arg.startswith("-")]

    if len(args) >= 2:  # noqa: PLR2004
        # Check if (group, command) is in our read-only registry
        return (args[0], args[1]) in _READ_ONLY_COMMANDS

    return False


class LogLevel(str, Enum):
    """
    Log levels for the CLI
    """

    Error = "ERROR"
    Warning = "WARNING"
    Debug = "DEBUG"
    Info = "INFO"


@define
class CLIContext:
    """
    Context object that can be passed to commands.

    The database is created lazily on first access to avoid running
    migrations and creating backups for commands that don't need the database.
    """

    config: Config
    console: Console
    skip_backup: bool = False
    _database: Database | None = field(default=None, alias="_database")

    @property
    def database(self) -> Database:
        """
        Get the database instance, creating it lazily if needed.

        The database is created on first access, which triggers migrations.
        Backup creation is skipped for read-only commands to reduce overhead.
        """
        if self._database is None:
            self._database = Database.from_config(self.config, skip_backup=self.skip_backup)
        return self._database

    def close(self) -> None:
        """Close the database connection if it was opened."""
        if self._database is not None:
            self._database.close()


def _version_callback(value: bool) -> None:
    if value:
        print(f"climate_ref: {__version__}")
        print(f"climate_ref-core: {__core_version__}")
        raise typer.Exit()


def _create_console() -> Console:
    # Hook for testing to disable color output

    # Rich respects the NO_COLOR environment variabl
    return Console()


def _load_config(configuration_directory: Path | None = None) -> Config:
    """
    Load the configuration from the specified directory

    Parameters
    ----------
    configuration_directory
        The directory to load the configuration from

        If the specified directory is not found, the process will exit with an exit code of 1

        If None, the default configuration will be loaded

    Returns
    -------
    :
        The configuration loaded from the specified directory
    """
    try:
        if configuration_directory:
            config = Config.load(configuration_directory / CONFIG_FILENAME, allow_missing=False)
        else:
            config = Config.default()
    except FileNotFoundError:
        typer.secho("Configuration file not found", fg=typer.colors.RED)
        raise typer.Exit(1)
    return config


def build_app() -> typer.Typer:
    """
    Build the CLI app

    This registers all the commands and subcommands of the CLI app.
    Some commands may not be available if certain dependencies are not installed,
    for example the Celery CLI is only available if the `climate-ref-celery` package is installed.

    Returns
    -------
    :
        The CLI app
    """
    # Import here to avoid circular imports since submodules import read_only from this module
    from climate_ref.cli import (
        config,
        datasets,
        executions,
        providers,
        solve,
        test_cases,
    )

    app = typer.Typer(name="ref", no_args_is_help=True)

    app.command(name="solve")(solve.solve)
    app.add_typer(config.app, name="config")
    app.add_typer(datasets.app, name="datasets")
    app.add_typer(executions.app, name="executions")
    app.add_typer(providers.app, name="providers")
    app.add_typer(test_cases.app, name="test-cases")

    try:
        celery_app = importlib.import_module("climate_ref_celery.cli").app

        app.add_typer(celery_app, name="celery")
    except ImportError:
        logger.debug("Celery CLI not available")

    return app


app = build_app()


@app.callback()
def main(  # noqa: PLR0913
    ctx: typer.Context,
    configuration_directory: Annotated[
        Path | None,
        typer.Option(help="Configuration directory"),
    ] = None,
    verbose: Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Set the log level to DEBUG"),
    ] = False,
    quiet: Annotated[
        bool,
        typer.Option("--quiet", "-q", help="Set the log level to WARNING"),
    ] = False,
    log_level: Annotated[
        LogLevel,
        typer.Option(case_sensitive=False, help="Set the level of logging information to display"),
    ] = LogLevel.Info,
    version: Annotated[
        bool | None,
        typer.Option(
            "--version", callback=_version_callback, is_eager=True, help="Print the version and exit"
        ),
    ] = None,
) -> None:
    """
    A CLI for the Assessment Fast Track Rapid Evaluation Framework

    This CLI provides a number of commands for managing and executing diagnostics.
    """  # noqa: D401
    if quiet:
        log_level = LogLevel.Warning
    if verbose:
        log_level = LogLevel.Debug

    logger.remove()

    config = _load_config(configuration_directory)
    config.log_level = log_level.value

    log_format = config.log_format
    initialise_logging(level=config.log_level, format=log_format, log_directory=config.paths.log)

    logger.debug(f"Configuration loaded from: {config._config_file!s}")

    # Create context with lazy database initialization
    # The database is only created when first accessed
    # Skip backup for read-only commands to reduce overhead
    skip_backup = _is_read_only_command()
    cli_context = CLIContext(config=config, console=_create_console(), skip_backup=skip_backup)
    ctx.obj = cli_context

    # Register cleanup to close database connection when CLI exits
    ctx.call_on_close(cli_context.close)


if __name__ == "__main__":
    app()
