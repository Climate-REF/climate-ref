import pandas as pd
from loguru import logger
from rich import box
from rich.console import Console
from rich.table import Table


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

    for index, value_list in enumerate(df.values.tolist()):
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

    if console is None:
        console = Console()

    max_col_count = console.width // 10
    table = df_to_table(df, max_col_count=max_col_count)

    console.print(table)
