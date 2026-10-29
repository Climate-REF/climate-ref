from __future__ import annotations

import fcntl
import json
import os
from collections.abc import Generator
from contextlib import contextmanager
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from loguru import logger
from rich import box
from rich.console import Console
from rich.table import Table

from climate_ref_core.env import env

if TYPE_CHECKING:
    import pandas as pd

_BYTES_PER_UNIT = 1024

LOCK_ENVVAR = "REF_LOCK"
LOCK_FILE_ENVVAR = "REF_LOCK_FILE"
LOCK_HELD_EXIT_CODE = 75
"""``EX_TEMPFAIL``, so callers can tell a skipped run from a failed one"""


class OutputFormat(StrEnum):
    """
    Output format for tabular CLI commands.
    """

    table = "table"
    json = "json"


def format_size(size_bytes: int | float) -> str:
    """
    Format file size in human-readable form.

    Parameters
    ----------
    size_bytes
        Size in bytes

    Returns
    -------
    :
        Human-readable size string (e.g., "1.5 MB")
    """
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < _BYTES_PER_UNIT:
            return f"{size:.1f} {unit}"
        size /= _BYTES_PER_UNIT
    return f"{size:.1f} TB"


def parse_facet_filters(filters: list[str] | None) -> dict[str, list[str]]:
    """
    Parse facet filters from key=value format into a dictionary.

    Multiple values for the same key are collected into a list (OR logic).
    Different keys are ANDed together when used for filtering.

    Parameters
    ----------
    filters
        List of filter strings in 'key=value' format.
        Multiple entries with the same key are ORed
        (e.g., ``["source_id=A", "source_id=B"]`` matches A or B).

    Returns
    -------
    dict[str, list[str]]
        Dictionary mapping facet keys to lists of allowed values

    Raises
    ------
    ValueError
        If a filter string is not in valid 'key=value' format

    Examples
    --------
    >>> parse_facet_filters(["source_id=GFDL-ESM4", "variable_id=tas"])
    {'source_id': ['GFDL-ESM4'], 'variable_id': ['tas']}

    >>> parse_facet_filters(["source_id=A", "source_id=B"])
    {'source_id': ['A', 'B']}
    """
    if not filters:
        return {}

    parsed: dict[str, list[str]] = {}
    for filter_str in filters:
        if "=" not in filter_str:
            raise ValueError(
                f"Invalid filter format: '{filter_str}'. "
                f"Expected format: 'key=value' or 'dataset_type.key=value' "
                f"(e.g., 'source_id=GFDL-ESM4' or 'cmip6.source_id=GFDL-ESM4')"
            )

        key, value = filter_str.split("=", 1)
        key = key.strip()
        value = value.strip()

        if not key:
            raise ValueError(f"Empty key in filter: '{filter_str}'")
        if not value:
            raise ValueError(f"Empty value in filter: '{filter_str}'")

        parsed.setdefault(key, []).append(value)

    return parsed


def df_to_table(df: pd.DataFrame, max_col_count: int = -1) -> Table:
    """
    Convert a DataFrame to a rich Table instance

    Parameters
    ----------
    df
        DataFrame to convert
    max_col_count
        Maximum number of columns to display

        If the DataFrame has more columns than this, the excess columns will be truncated
        If set to -1, all columns will be displayed.
        For very wide DataFrames, then this may result in no values at all being displayed
        if the column-width ends up being less than 1 char.

    Returns
    -------
        Rich Table instance representing the DataFrame
    """
    # Initiate a Table instance to be modified
    if max_col_count > 0 and len(df.columns) > max_col_count:
        logger.warning(f"Too many columns to display ({len(df.columns)}), truncating to {max_col_count}")
        df = df.iloc[:, :max_col_count]

    table = Table(*[str(column) for column in df.columns])

    for value_list in df.values.tolist():
        row = [str(x) for x in value_list]
        table.add_row(*row)

    # Update the style of the table
    table.row_styles = ["none", "dim"]
    table.box = box.SIMPLE_HEAD

    return table


def pretty_print_df(df: pd.DataFrame, console: Console | None = None) -> None:
    """
    Pretty print a DataFrame

    Parameters
    ----------
    df
        DataFrame to print
    console
        Console to use for printing

        If not provided, a new Console instance will be created
    """
    # Drop duplicates as they are not informative to CLI users.
    try:
        df = df.drop_duplicates()
    except TypeError:
        # Some columns may hold unhashable values so fallback to str
        df = df.loc[~df.astype(str).duplicated()]

    if console is None:  # pragma: no branch
        logger.debug("Creating new console for pretty printing")
        console = Console()

    max_col_count = console.width // 10
    table = df_to_table(df, max_col_count=max_col_count)

    console.print(table)


def render_dataframe(
    df: pd.DataFrame,
    console: Console | None = None,
    output_format: OutputFormat = OutputFormat.table,
) -> None:
    """
    Render a DataFrame as either a rich table or machine-readable JSON.

    Parameters
    ----------
    df
        DataFrame to render
    console
        Console to use for table output (ignored for JSON)
    output_format
        ``table`` (default) for a human-readable rich table,
        or ``json`` for a list of records printed to stdout.

        Unlike the table output, the JSON output is not truncated and does not
        drop duplicate rows, so it always contains the full values
        (e.g. complete IDs and paths) and is suitable for scripting.
    """
    if output_format == OutputFormat.json:
        # Coerce pandas nulls
        # (``NaT`` in datetime columns such as ``retracted_at``,
        # ``NaN`` in numeric columns) to ``None`` so they serialise as JSON ``null``.
        records = df.astype(object).where(df.notna(), None).to_dict(orient="records")  # type: ignore[arg-type]
        print(json.dumps(records, default=str, indent=2))
    else:
        pretty_print_df(df, console=console)


def resolve_lock_file(lock: bool, lock_file: Path | None) -> Path | None:
    """
    Pick the lock file for a command, or None when locking is off

    A lock file path turns locking on by itself.
    ``--lock`` alone uses ``ref.lock`` in the REF configuration directory.
    """
    if lock_file is not None:
        return lock_file
    if lock:
        return env.path("REF_CONFIGURATION").resolve() / "ref.lock"
    return None


@contextmanager
def exclusive_lock(path: Path | None) -> Generator[None]:
    """
    Hold an exclusive advisory lock on ``path`` for the duration of the block

    Exits with ``LOCK_HELD_EXIT_CODE`` straight away if another process holds the lock.
    The holder's PID is written to the file to help track down a stale holder.
    POSIX locks are used because NFS clients and the exporting host only see each other's POSIX locks.
    """
    if path is None:
        yield
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as lock_file:
        try:
            fcntl.lockf(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, PermissionError):
            lock_file.seek(0)
            holder = lock_file.read().strip() or "unknown"
            logger.error(f"Another process holds the lock file {path} (PID {holder}). Exiting.")
            raise typer.Exit(code=LOCK_HELD_EXIT_CODE)
        try:
            lock_file.truncate(0)
            lock_file.write(f"{os.getpid()}\n")
            lock_file.flush()
            yield
        finally:
            fcntl.lockf(lock_file, fcntl.LOCK_UN)
