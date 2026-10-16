"""
View past solves and what they did
"""

import json
from typing import Annotated

import typer
from loguru import logger
from rich.markup import escape
from rich.panel import Panel
from rich.text import Text

from climate_ref.cli._utils import OutputFormat, df_to_table, render_dataframe
from climate_ref.results import Reader
from climate_ref.results.datasets import DatasetCollection
from climate_ref.results.solves import SolveOutcomes, SolveView, SolveWindow

app = typer.Typer(help=__doc__)


@app.command(name="list")
def list_(
    ctx: typer.Context,
    limit: int = typer.Option(20, help="Limit the number of rows to display"),
    output_format: Annotated[
        OutputFormat,
        typer.Option(
            "--format",
            help="Output format: 'table' (default) or machine-readable 'json'.",
        ),
    ] = OutputFormat.table,
) -> None:
    """
    List solves, most recent first.

    Dry runs are not recorded, so they are not listed.
    """
    collection = Reader(ctx.obj.database).solves.list(limit=limit)

    render_dataframe(collection.to_pandas(), console=ctx.obj.console, output_format=output_format)

    if collection.total_count > limit:
        logger.warning(
            f"Displaying {limit} of {collection.total_count} solves. "
            f"Use the `--limit` option to display more."
        )


def _solve_panel(solve: SolveView, window: SolveWindow | None) -> Panel:
    window_start = window.start if window and window.start else "none (first solve)"
    return Panel(
        f"Status: [bold]{solve.status.value}[/]\n"
        f"Stage: [bold]{solve.stage.value}[/]\n"
        f"Started At: [bold]{solve.started_at}[/]\n"
        f"Finished At: [bold]{solve.finished_at or '-'}[/]\n"
        f"Error: [bold]{escape(solve.error or '-')}[/]\n"
        f"Skipped providers: [bold]{', '.join(solve.skipped_providers) or '-'}[/]\n"
        f"Host: [bold]{escape(solve.hostname)}[/]\n"
        f"REF version: [bold]{solve.ref_version}[/]\n"
        f"Window start: [bold]{window_start}[/]\n"
        f"Options: {escape(json.dumps(dict(solve.options), sort_keys=True, default=str))}",
        title=f"Solve Details: [bold]{solve.id}[/]",
    )


def _outcomes_panel(outcomes: SolveOutcomes) -> Panel:
    reasons = ", ".join(f"{reason}: {count}" for reason, count in sorted(outcomes.failure_reasons.items()))
    return Panel(
        f"Succeeded: [bold]{outcomes.succeeded}[/]\n"
        f"Failed: [bold]{outcomes.failed}[/]" + (f" ({reasons})" if reasons else "") + "\n"
        f"Pending: [bold]{outcomes.pending}[/]\n"
        f"Total: [bold]{outcomes.total}[/]",
        title="Executions",
    )


def _datasets_panel(datasets: DatasetCollection, title: str) -> Panel:
    import pandas as pd

    if not datasets.items:
        return Panel(Text("None."), title=f"{title} (0)")

    table = df_to_table(
        pd.DataFrame(
            [{"id": d.id, "slug": d.slug, "dataset_type": d.dataset_type.value} for d in datasets.items]
        )
    )
    if datasets.total_count > len(datasets):
        table.caption = f"Showing {len(datasets)} of {datasets.total_count}"
    return Panel(table, title=f"{title} ({datasets.total_count})")


@app.command()
def inspect(
    ctx: typer.Context,
    solve_id: int,
    limit: int = typer.Option(20, help="Maximum number of datasets to list in each section"),
) -> None:
    """
    Inspect a solve by its ID

    This shows how the solve ended, the outcomes of the executions it created,
    and the datasets created or retracted since the previous solve started.
    """
    console = ctx.obj.console
    solves = Reader(ctx.obj.database).solves

    solve = solves.get(solve_id)
    if solve is None:
        logger.error(f"Solve not found: {solve_id}")
        raise typer.Exit(code=1)

    console.print(_solve_panel(solve, solves.window(solve_id)))
    console.print(_outcomes_panel(solves.outcomes(solve_id)))
    console.print(_datasets_panel(solves.new_datasets(solve_id, limit=limit), "New datasets"))
    console.print(_datasets_panel(solves.retracted_datasets(solve_id, limit=limit), "Retracted datasets"))
