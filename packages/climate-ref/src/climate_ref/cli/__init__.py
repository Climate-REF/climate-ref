"""Entrypoint for the CLI"""

import functools
import importlib
import importlib.util
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import typer
from attrs import define, field, frozen
from loguru import logger
from rich.console import Console
from typer.core import TyperCommand, TyperGroup

from climate_ref import __version__
from climate_ref.config import Config
from climate_ref.constants import CONFIG_FILENAME
from climate_ref_core import __version__ as __core_version__
from climate_ref_core.env import env
from climate_ref_core.logging import initialise_logging

if TYPE_CHECKING:
    from climate_ref.database import Database

# Registry of read-only command paths (group, command)
# These commands skip database backup before migrations since they don't modify data
_READ_ONLY_COMMANDS: set[tuple[str, str]] = {
    ("config", "list"),
    ("config", "get"),
    ("config", "validate"),
    ("datasets", "list"),
    ("datasets", "list-columns"),
    ("datasets", "stats"),
    ("db", "status"),
    ("db", "heads"),
    ("db", "history"),
    ("db", "tables"),
    ("diagnostics", "list"),
    ("executions", "list-groups"),
    ("executions", "inspect"),
    ("executions", "stats"),
    ("executions", "values"),
    ("executions", "resources"),
    ("providers", "list"),
    ("providers", "show"),
    ("solves", "list"),
    ("solves", "inspect"),
    # test-cases commands operate on test artifacts (filesystem, native store, manifests).
    # None persist diagnostic executions to the database, so they never need a pre-migration backup.
    ("test-cases", "fetch"),
    ("test-cases", "list"),
    ("test-cases", "run"),
    ("test-cases", "replay"),
    ("test-cases", "mint"),
    ("test-cases", "ci-gate"),
}

# Read-only commands that sit at the top level rather than inside a group.
_READ_ONLY_TOP_LEVEL_COMMANDS: set[str] = {"doctor"}

# Commands whose job is to create or inspect the configuration file, so they must
# run even when that file is missing or malformed.
_CONFIG_BOOTSTRAP_COMMANDS: set[tuple[str, str]] = {
    ("config", "init"),
    ("config", "validate"),
}


_VALUE_TAKING_OPTIONS: set[str] = {"--configuration-directory", "--log-level"}
"""Options on the root callback that consume the argument after them."""


def _command_argv() -> list[str]:
    """Read the command path given on the command line, with options and their values removed."""
    path = []
    skip_value = False
    for arg in sys.argv[1:]:
        if skip_value:
            skip_value = False
            continue
        if arg.startswith("-"):
            skip_value = arg in _VALUE_TAKING_OPTIONS
            continue
        path.append(arg)
    return path


def _command_path_in_argv(command_paths: set[tuple[str, str]]) -> bool:
    return tuple(_command_argv()[:2]) in command_paths


def _is_read_only_command() -> bool:
    """
    Check if the current command is a read-only command by inspecting sys.argv.

    This checks against a registry of known read-only command paths since
    the Typer callback runs before nested commands are resolved.
    """
    # A top-level command is the first argument of the command path. Looking anywhere else would
    # treat a path or an option value that happens to read as a command name as read-only.
    command = _command_argv()
    if command and command[0] in _READ_ONLY_TOP_LEVEL_COMMANDS:
        return True
    return _command_path_in_argv(_READ_ONLY_COMMANDS)


def _is_config_bootstrap_command() -> bool:
    """Check if the current command must tolerate missing or malformed config."""
    return _command_path_in_argv(_CONFIG_BOOTSTRAP_COMMANDS)


class LogLevel(StrEnum):
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
    configuration_directory: Path | None = None
    config_load_error: str | None = None
    skip_backup: bool = False
    _database: "Database | None" = field(default=None, alias="_database")
    _database_unmigrated: "Database | None" = field(default=None, alias="_database_unmigrated")

    @property
    def database(self) -> "Database":
        """
        Get the database instance, creating it lazily if needed.

        The database is created on first access, which triggers migrations.
        Backup creation is skipped for read-only commands to reduce overhead.
        """
        if self._database is None:
            from climate_ref.database import Database

            self._database = Database.from_config(self.config, skip_backup=self.skip_backup)
        return self._database

    @property
    def database_unmigrated(self) -> "Database":
        """
        Get a database instance without running migrations.

        Used by ``db`` subcommands that inspect or manage migration state directly.
        """
        if self._database_unmigrated is None:
            from climate_ref.database import Database

            self._database_unmigrated = Database.from_config(self.config, run_migrations=False)
        return self._database_unmigrated

    def close(self) -> None:
        """Close the database connection if it was opened."""
        if self._database is not None:
            self._database.close()
        if self._database_unmigrated is not None:
            self._database_unmigrated.close()


def _version_callback(value: bool) -> None:
    if value:
        print(f"climate_ref: {__version__}")
        print(f"climate_ref-core: {__core_version__}")
        raise typer.Exit()


def _create_console() -> Console:
    # Hook for testing to disable color output

    # Rich respects the NO_COLOR environment variabl
    return Console()


def _load_config(
    configuration_directory: Path | None = None, tolerate_problems: bool = False
) -> tuple[Config, str | None]:
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
    if tolerate_problems:
        if configuration_directory:
            target = configuration_directory / CONFIG_FILENAME
        else:
            target = env.path("REF_CONFIGURATION") / CONFIG_FILENAME
        try:
            return Config.load(target, allow_missing=True), None
        except Exception as exc:
            config = Config()
            config._config_file = target
            return config, str(exc)

    try:
        if configuration_directory:
            config = Config.load(configuration_directory / CONFIG_FILENAME, allow_missing=False)
        else:
            config = Config.default()
    except FileNotFoundError:
        typer.secho("Configuration file not found", fg=typer.colors.RED)
        raise typer.Exit(1)
    return config, None


@frozen
class _Subcommand:
    """A top-level command, imported only when it is invoked."""

    module: str
    attr: str
    help: str
    """Short help shown in the root ``--help`` listing, so listing needs no import."""
    requires: str | None = None
    """Optional package that must be installed for the command to be available."""

    def load(self, name: str) -> Any:
        """Import the command and convert it to a Click command named ``name``."""
        target = getattr(importlib.import_module(self.module), self.attr)
        command: Any
        if isinstance(target, typer.Typer):
            command = typer.main.get_group(target)
        else:
            wrapper = typer.Typer(add_completion=False)
            wrapper.command()(target)
            command = typer.main.get_command(wrapper)
        command.name = name
        return command


_SUBCOMMANDS: dict[str, _Subcommand] = {
    "solve": _Subcommand("climate_ref.cli.solve", "solve", "Solve for executions that require recalculation"),
    "doctor": _Subcommand(
        "climate_ref.cli.doctor", "doctor", "Check this deployment for data and configuration problems."
    ),
    "config": _Subcommand("climate_ref.cli.config", "app", "View and update the REF configuration"),
    "datasets": _Subcommand("climate_ref.cli.datasets", "app", "View and ingest input datasets"),
    "db": _Subcommand("climate_ref.cli.db", "app", "Database management commands"),
    "diagnostics": _Subcommand("climate_ref.cli.diagnostics", "app", "View diagnostic metadata"),
    "executions": _Subcommand("climate_ref.cli.executions", "app", "View execution groups and their results"),
    "providers": _Subcommand("climate_ref.cli.providers", "app", "Manage the REF providers."),
    "solves": _Subcommand("climate_ref.cli.solves", "app", "View past solves and what they did"),
    "test-cases": _Subcommand(
        "climate_ref.cli.test_cases", "app", "Test data management commands for diagnostic development."
    ),
    "celery": _Subcommand(
        "climate_ref_celery.cli", "app", "Managing remote celery workers", requires="climate_ref_celery"
    ),
}


@functools.cache
def available_subcommands() -> dict[str, _Subcommand]:
    """
    Get the top-level commands that can run in this environment.

    A command with ``requires`` set is only available if that package is installed.
    """
    return {
        name: sub
        for name, sub in _SUBCOMMANDS.items()
        if sub.requires is None or importlib.util.find_spec(sub.requires) is not None
    }


class _RootGroup(TyperGroup):
    """
    Root command group that imports each subcommand on first use.

    Importing every subcommand up front pulls in the database, ESGF and HPC stacks,
    which made even ``ref --help`` take over a second.
    """

    _listing_help = False

    def list_commands(self, ctx: Any) -> list[str]:
        return list(available_subcommands())

    def get_command(self, ctx: Any, cmd_name: str) -> Any:
        if cmd_name not in self.commands:
            subcommand = available_subcommands().get(cmd_name)
            if subcommand is None:
                return None
            if self._listing_help:
                return TyperCommand(name=cmd_name, help=subcommand.help)
            self.commands[cmd_name] = subcommand.load(cmd_name)
        return super().get_command(ctx, cmd_name)

    @contextmanager
    def _listing(self) -> Iterator[None]:
        """Serve placeholders from `get_command` while only names and short help are needed."""
        self._listing_help = True
        try:
            yield
        finally:
            self._listing_help = False

    def format_help(self, ctx: Any, formatter: Any) -> None:
        with self._listing():
            super().format_help(ctx, formatter)

    def shell_complete(self, ctx: Any, incomplete: str) -> Any:
        with self._listing():
            return super().shell_complete(ctx, incomplete)

    def invoke(self, ctx: Any) -> Any:
        try:
            return super().invoke(ctx)
        except (typer.Exit, typer.Abort, typer.BadParameter):
            raise
        except Exception:
            # Typer prints the traceback to stderr, so keep this below the default console level
            logger.opt(exception=True).debug("Command failed with an uncaught exception")
            raise


def build_app() -> typer.Typer:
    """
    Build the CLI app

    The subcommands are registered lazily by the root group, see `available_subcommands`.

    Returns
    -------
    :
        The CLI app
    """
    return typer.Typer(name="ref", no_args_is_help=True, cls=_RootGroup)


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
        LogLevel | None,
        typer.Option(case_sensitive=False, help="Set the level of logging information to display"),
    ] = None,
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

    tolerate_config_problems = ctx.invoked_subcommand == "config" or _is_config_bootstrap_command()
    config, config_load_error = _load_config(
        configuration_directory,
        tolerate_problems=tolerate_config_problems,
    )
    if log_level is not None:
        config.log_level = log_level.value

    log_format = config.log_format
    initialise_logging(
        level=config.log_level,
        format=log_format,
        log_directory=config.paths.log,
        serialize=config.log_json,
    )

    logger.debug(f"Configuration loaded from: {config._config_file!s}")

    if "celery" not in available_subcommands():
        logger.debug("Celery CLI not available")

    # Create context with lazy database initialization
    # The database is only created when first accessed
    # Skip backup for read-only commands to reduce overhead
    skip_backup = _is_read_only_command()
    cli_context = CLIContext(
        config=config,
        console=_create_console(),
        configuration_directory=configuration_directory,
        config_load_error=config_load_error,
        skip_backup=skip_backup,
    )
    ctx.obj = cli_context

    # Register cleanup to close database connection when CLI exits
    ctx.call_on_close(cli_context.close)


if __name__ == "__main__":
    app()
