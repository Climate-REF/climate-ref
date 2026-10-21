"""
Logging utilities

The REF uses [loguru](https://loguru.readthedocs.io/en/stable/), a simple logging framework.
The log level and format are configured via the REF configuration file.
"""

import contextlib
import inspect
import logging
import multiprocessing
import os
import sys
from collections.abc import Generator
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pooch
from loguru import logger
from rich.pretty import pretty_repr

if TYPE_CHECKING:
    from loguru import Logger

    from climate_ref_core.diagnostics import ExecutionDefinition

EXECUTION_LOG_FILENAME = "out.log"
"""
Filename for the execution log.

This file is written via [climate_ref_core.logging.redirect_logs][].
"""

DEFAULT_LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS Z}</green> | <level>{level: <8}</level> | "
    "<cyan>{name}</cyan> - {extra[execution_prefix]}<level>{message}</level>"
)
"""
Default log format used by the REF

`{extra[execution_prefix]}` is `[<execution slug>] ` for records logged during an execution,
and empty otherwise.
"""

VERBOSE_LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS Z} e{elapsed}s</green> | "
    "<level>{level: <8}</level> | "
    "{process.name}:{process.id} | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "{extra[execution_prefix]}<level>{message}</level>"
)
"""
The verbose log format is used for debugging and development.

This is the format that is used when writing the log messages to file for later debugging.
It contains information about the process and function that the log message was generated in.
"""


class _InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding Loguru level if it exists.
        level: str | int
        try:
            level = logger.level(record.levelname).name
        except ValueError:  # pragma: no cover
            level = record.levelno

        # Find caller from where originated the logged message.
        frame, depth = inspect.currentframe(), 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


def _retention_count_ignore_missing(logs: list[str]) -> None:  # pragma: no cover
    """Remove old log files, tolerating concurrent cleanup by another process."""
    keep = 10
    existing_logs = []
    for log in logs:
        path = Path(log)
        try:
            mtime = path.stat().st_mtime
        except FileNotFoundError:
            continue
        existing_logs.append((mtime, path))

    for _, path in sorted(existing_logs, key=lambda item: (-item[0], item[1].as_posix()))[keep:]:
        with contextlib.suppress(FileNotFoundError):
            path.unlink()


def initialise_logging(
    level: int | str,
    format: str,  # noqa: A002
    log_directory: str | Path,
    serialize: bool = False,
) -> None:  # pragma: no cover
    """
    Initialise the logging for the REF

    This sets up the default log handler and configures the REF logger.

    When ``serialize`` is True the console handler writes one JSON object per record,
    so a multi-line traceback stays a single entry for log aggregators.
    """
    capture_logging()
    log_directory = Path(log_directory)
    process_name = multiprocessing.current_process().name

    # Remove any existing handlers
    logger.remove()

    # Write out debug logs to a file
    log_directory.mkdir(parents=True, exist_ok=True)
    filename = f"climate-ref_{{time:YYYY-MM-DD_HH-mm}}_{process_name}.log"
    logger.add(
        sink=log_directory / filename,
        retention=_retention_count_ignore_missing,
        level="DEBUG",
        format=VERBOSE_LOG_FORMAT,
        colorize=False,
    )
    logger.info("Starting REF logging")
    logger.info(f"arguments: {sys.argv}")

    # NO_COLOR is honoured instead of LOGURU_COLORIZE because it is more widely supported.
    # Otherwise loguru colours only a TTY, so container logs stay free of ANSI codes.
    no_color = os.environ.get("NO_COLOR") is not None
    add_log_handler(
        level=level,
        format=format,
        colorize=False if no_color else None,
        serialize=serialize,
        # Variable values in tracebacks are bulky and can leak secrets, so they stay in the debug file
        diagnose=False,
    )


def _add_execution_prefix(record: Any) -> None:
    execution = record["extra"].get("execution")
    record["extra"]["execution_prefix"] = f"[{execution}] " if execution else ""


def capture_logging() -> None:
    """
    Capture logging from the standard library and redirect it to Loguru

    This also sets the `execution_prefix` extra field used by the REF log formats.

    Note that this replaces the root logger, so any other handlers attached to it will be removed.
    """
    logger.configure(patcher=_add_execution_prefix)
    # Pooch adds a handler to its own logger which circumvents the REF logger
    pooch.get_logger().handlers.clear()
    pooch.get_logger().addHandler(_InterceptHandler())

    logging.basicConfig(handlers=[_InterceptHandler()], level=0, force=True)

    # Disable some overly verbose logs
    logger.disable("alembic.runtime.migration")
    logger.disable("matplotlib.colorbar")
    logger.disable("matplotlib.ticker")
    logger.disable("matplotlib.font_manager")
    logger.disable("pyproj.transformer")
    logger.disable("pint.facets.plain.registry")


def add_log_handler(**kwargs: Any) -> None:
    """
    Add a log sink to the logger to capture logs.

    This is useful for testing purposes, to ensure that logs are captured correctly.
    """
    if hasattr(logger, "default_handler_id"):
        raise AssertionError("The default log handler has already been created")

    kwargs.setdefault("sink", sys.stderr)

    handled_id = logger.add(**kwargs)

    # Track the current handler via custom attributes on the logger
    # This is a bit of a workaround because of loguru's super slim API that doesn't allow for
    # modification of existing handlers.
    logger.default_handler_id = handled_id  # type: ignore[attr-defined]
    logger.default_handler_kwargs = kwargs  # type: ignore[attr-defined]

    capture_logging()


def remove_log_handler() -> None:
    """
    Remove the default log handler from the logger.

    This is useful for cleaning up after tests or when changing logging configurations.
    The previously used logger kwargs are kept in `logger.default_handler_kwargs` if the
    logger should be readded later
    """
    if hasattr(logger, "default_handler_id"):
        try:
            logger.remove(logger.default_handler_id)  # pyright: ignore[reportAttributeAccessIssue]
        except ValueError:
            # This can happen if the handler has already been removed
            # or if the logger was never configured
            pass
        del logger.default_handler_id  # pyright: ignore[reportAttributeAccessIssue]
    else:
        raise AssertionError("No default log handler to remove.")


def _at_least_warning(level: int | str) -> int:
    level_no = level if isinstance(level, int) else logger.level(level).no
    return max(level_no, logger.level("WARNING").no)


def execution_log_only() -> "Logger":
    """
    Get a logger whose records go to the execution log but not the console

    Use this for detail that is too bulky for collected logs, such as a failed command's full output.
    Outside [redirect_logs][climate_ref_core.logging.redirect_logs] the records are logged as usual.
    """
    return logger.bind(execution_log_only=True)


def _skip_execution_log_only(inner: Any) -> Any:
    def _filter(record: Any) -> bool:
        if record["extra"].get("execution_log_only"):
            return False
        return inner is None or inner(record)

    return _filter


@contextlib.contextmanager
def redirect_logs(definition: "ExecutionDefinition", log_level: str) -> Generator[None, None, None]:
    """
    Temporarily redirect log output to a file.

    The console handler, if configured, only shows warnings and errors while redirected,
    so failures still reach stdout-based log collection without the diagnostic's full output.

    This also writes some common log messages

    Parameters
    ----------
    definition
        Diagnostic definition to capture logging for

    log_level
        Log level as a string e.g. INFO, WARNING, DEBUG.
        This log level will dictate what logs will be sent to disk
        The logger will also be reset to this level after leaving the context manager.

    """
    app_logger_configured = hasattr(logger, "default_handler_id")

    # If the app logger hasn't been configured yet, we don't need to replace it,
    # as logs will also be written to the console as loguru adds a stderr handler by default
    console_handler_id = None
    file_handler_id = None
    if app_logger_configured:
        remove_log_handler()
    try:
        if app_logger_configured:
            console_kwargs = dict(logger.default_handler_kwargs)  # type: ignore[attr-defined]
            console_kwargs["level"] = _at_least_warning(console_kwargs.get("level", "DEBUG"))
            console_kwargs["filter"] = _skip_execution_log_only(console_kwargs.get("filter"))
            console_handler_id = logger.add(**console_kwargs)

        # Add a new log handler for the execution log
        output_file = definition.output_directory / EXECUTION_LOG_FILENAME
        file_handler_id = logger.add(output_file, level=log_level, colorize=False)
        capture_logging()

        logger.info(f"Running definition {pretty_repr(definition)}")
        failed = False
        try:
            yield
        except:
            failed = True
            # The executor logs this failure to the console itself
            execution_log_only().exception("Execution failed")
            raise
        finally:
            outcome = "failed" if failed else "finished"
            logger.info(
                f"Diagnostic execution {outcome}. Results available in {definition.output_fragment()}"
            )
    finally:
        # Reset the logger to the default, even if setting up the execution log failed
        if file_handler_id is not None:
            logger.remove(file_handler_id)
        if console_handler_id is not None:
            logger.remove(console_handler_id)
        # We only re-add the app handler if it was configured before
        if app_logger_configured:
            add_log_handler(**logger.default_handler_kwargs)  # type: ignore[attr-defined]


__all__ = [
    "EXECUTION_LOG_FILENAME",
    "capture_logging",
    "execution_log_only",
    "initialise_logging",
    "redirect_logs",
]
