"""
View diagnostic metadata
"""

from typing import Annotated

import typer
from loguru import logger

from climate_ref.cli._utils import OutputFormat, render_dataframe
from climate_ref.results import DiagnosticFilter, Reader

app = typer.Typer(help=__doc__)


@app.command(name="list")
def list_(  # noqa: PLR0913
    ctx: typer.Context,
    provider: Annotated[
        list[str] | None,
        typer.Option(
            help="Filter by provider slug (substring match, case-insensitive). "
            "Multiple values can be provided."
        ),
    ] = None,
    diagnostic: Annotated[
        list[str] | None,
        typer.Option(
            help="Filter by diagnostic slug (substring match, case-insensitive). "
            "Multiple values can be provided."
        ),
    ] = None,
    column: Annotated[
        list[str] | None,
        typer.Option(help="Only include specified columns in the output"),
    ] = None,
    limit: int = typer.Option(100, help="Limit the number of rows to display"),
    output_format: Annotated[
        OutputFormat,
        typer.Option(
            "--format",
            help="Output format: 'table' (default) or machine-readable 'json'.",
        ),
    ] = OutputFormat.table,
) -> None:
    """
    List the registered diagnostics.

    Columns:

    - provider: the provider slug.
    - diagnostic: the diagnostic slug.
    - name: the human-readable diagnostic name.
    - promoted_version: the currently promoted diagnostic version.
    - execution_group_count: execution groups created for it across all versions.
    - successful: promoted-version execution groups whose latest execution succeeded.
    - inflight: promoted-version execution groups whose latest execution is still running.
    - total: execution groups at the promoted version.
    """
    console = ctx.obj.console
    reader = Reader(ctx.obj.database)

    collection = reader.diagnostics.list(
        DiagnosticFilter(provider_contains=provider, diagnostic_contains=diagnostic),
        limit=limit,
    )

    results_df = collection.to_pandas()

    if column:
        if not all(col in results_df.columns for col in column):
            logger.error(f"Column not found: {column}")
            raise typer.Exit(code=1)
        results_df = results_df[column]

    render_dataframe(results_df, console=console, output_format=output_format)

    filtered_count = collection.total_count
    if filtered_count > limit:
        logger.warning(
            f"Displaying {limit} of {filtered_count} filtered results. "
            f"Use the `--limit` option to display more."
        )
