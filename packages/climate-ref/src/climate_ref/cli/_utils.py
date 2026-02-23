from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger
from rich import box
from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    import pandas as pd

_BYTES_PER_UNIT = 1024


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


def parse_facet_filters(filters: list[str] | None) -> dict[str, str]:
    """
    Parse facet filters from key=value format into a dictionary.

    Parameters
    ----------
    filters
        List of filter strings in 'key=value' format

    Returns
    -------
    dict[str, str]
        Dictionary mapping facet keys to values

    Raises
    ------
    ValueError
        If a filter string is not in valid 'key=value' format

    Examples
    --------
    >>> parse_facet_filters(["source_id=GFDL-ESM4", "variable_id=tas"])
    {'source_id': 'GFDL-ESM4', 'variable_id': 'tas'}
    """
    if not filters:
        return {}

    parsed: dict[str, str] = {}
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

        if key in parsed:
            logger.warning(f"Filter key '{key}' specified multiple times. Using last value: '{value}'")

        parsed[key] = value

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
    df = df.drop_duplicates()

    if console is None:  # pragma: no branch
        logger.debug("Creating new console for pretty printing")
        console = Console()

    max_col_count = console.width // 10
    table = df_to_table(df, max_col_count=max_col_count)

    console.print(table)
