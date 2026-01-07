"""Entrypoint for the CLI"""

import importlib
from enum import Enum
from pathlib import Path
from typing import Annotated

import typer
from attrs import define
from loguru import logger
from rich.console import Console

from climate_ref import __version__
from climate_ref.cli import config, datasets, executions, providers, solve, test_cases
from climate_ref.config import Config
from climate_ref.constants import CONFIG_FILENAME
from climate_ref.database import Database
from climate_ref_core import __version__ as __core_version__
from climate_ref_core.logging import initialise_logging


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
    Context object that can be passed to commands
    """

    config: Config
    database: Database
    console: Console


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

    ctx.obj = CLIContext(config=config, database=Database.from_config(config), console=_create_console())


if __name__ == "__main__":
    app()
