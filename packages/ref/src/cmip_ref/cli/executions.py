"""
View metric executions
"""

import pathlib
from typing import Annotated
from urllib.parse import quote

import pandas as pd
import typer
from loguru import logger
from rich.console import Console, Group
from rich.filesize import decimal
from rich.markup import escape
from rich.panel import Panel
from rich.text import Text
from rich.tree import Tree

from cmip_ref.cli._utils import df_to_table, pretty_print_df
from cmip_ref.models import MetricExecutionGroup, MetricExecutionResult
from cmip_ref.models.metric_execution import get_execution_group_and_latest_result

app = typer.Typer(help=__doc__)
console = Console()


@app.command()
def list_groups(
    ctx: typer.Context,
    column: Annotated[list[str] | None, typer.Option()] = None,
    limit: int = typer.Option(100, help="Limit the number of rows to display"),
) -> None:
    """
    List the metric execution groups that have been identified
    """
    session = ctx.obj.database.session

    execution_groups_results = get_execution_group_and_latest_result(session).limit(limit).all()
    execution_count = session.query(MetricExecutionGroup).count()

    results_df = pd.DataFrame(
        [
            {
                "id": execution_groups.id,
                "key": execution_groups.dataset_key,
                "provider": execution_groups.metric.provider.slug,
                "metric": execution_groups.metric.slug,
                "dirty": execution_groups.dirty,
                "successful": result.successful if result else None,
                "created_at": execution_groups.created_at,
                "updated_at": execution_groups.updated_at,
            }
            for execution_groups, result in execution_groups_results
        ]
    )

    if column:
        if not all(col in results_df.columns for col in column):
            logger.error(f"Column not found in data catalog: {column}")
            raise typer.Exit(code=1)
        results_df = results_df[column]

    pretty_print_df(results_df, console=console)
    if execution_count > limit:
        logger.warning(
            f"Displaying {limit} of {execution_count} results. Use the `--limit` option to display more."
        )


def walk_directory(directory: pathlib.Path, tree: Tree) -> None:
    """Recursively build a Tree with directory contents."""
    # Sort dirs first then by filename
    paths = sorted(
        pathlib.Path(directory).iterdir(),
        key=lambda path: (path.is_file(), path.name.lower()),
    )
    for path in paths:
        # Remove hidden files
        if path.name.startswith("."):
            continue
        if path.is_dir():
            style = "dim" if path.name.startswith("__") else ""
            branch = tree.add(
                f"[bold magenta]:open_file_folder: [link file://{path}]{escape(path.name)}",
                style=style,
                guide_style=style,
            )
            walk_directory(path, branch)
        else:
            text_filename = Text(path.name, "green")
            text_filename.highlight_regex(r"\..*$", "bold red")
            text_filename.stylize(f"link file://{path}")
            file_size = path.stat().st_size
            text_filename.append(f" ({decimal(file_size)})", "blue")
            tree.add(text_filename)


def _execution_panel(execution: MetricExecutionGroup) -> Panel:
    if len(execution.results) == 0:
        result = None
    else:
        result = execution.results[-1]

    panel = Panel(
        f"Key: [bold]{execution.dataset_key}[/]\n"
        f"Metric: [bold]{execution.metric.slug}[/]\n"
        f"Provider: [bold]{execution.metric.provider.slug}[/]\n"
        f"Dirty: [bold]{execution.dirty}[/]\n"
        f"Successful: [bold]{result.successful if result else 'not-started'}[/]\n"
        f"Created At: [bold]{execution.created_at}[/]\n"
        f"Updated At: [bold]{execution.updated_at}[/]\n"
        f"Number of attempted executions: [bold]{len(execution.results)}[/]",
        title=f"Execution Details: [bold]{execution.id}[/]",
    )
    return panel


def _datasets_panel(result: MetricExecutionResult) -> Panel:
    datasets = result.datasets

    datasets_df = pd.DataFrame(
        [
            {"id": dataset.id, "slug": dataset.slug, "dataset_type": dataset.dataset_type}
            for dataset in datasets
        ]
    )

    return Panel(
        df_to_table(datasets_df),
        title=f"Datasets hash: {result.dataset_hash}",
    )


def _results_directory_panel(result_directory: pathlib.Path) -> Panel:
    if result_directory.exists():
        tree = Tree(
            f":open_file_folder: [link file://{result_directory}]{result_directory}",
            guide_style="bold bright_blue",
        )
        walk_directory(result_directory, tree)
        return Panel(tree, title="File Tree")
    else:
        target_directory = f"file://{quote(str(result_directory.parent))}"
        link_text = escape(str(result_directory))

        return Panel(
            Group(
                Text("Result directory not found.", "bold red"),
                # Link to the parent directory otherwise this link will never be resolved
                Text.from_markup(f"[bold magenta]:open_file_folder:[link={target_directory}]{link_text}"),
            ),
            title="File Tree",
        )


@app.command()
def inspect(ctx: typer.Context, execution_id: int) -> None:
    """
    Inspect a specific execution by its ID
    """
    config = ctx.obj.config
    session = ctx.obj.database.session
    execution = session.get(MetricExecutionGroup, execution_id)

    if not execution:
        logger.error(f"Execution not found: {execution_id}")
        raise typer.Exit(code=1)

    console.print(_execution_panel(execution))

    if not execution.results:
        logger.error(f"No results found for execution: {execution_id}")
        return

    result: MetricExecutionResult = execution.results[-1]
    result_directory = config.paths.results / result.output_fragment

    console.print(_datasets_panel(result))
    console.print(_results_directory_panel(result_directory))
