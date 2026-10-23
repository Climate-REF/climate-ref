"""
View execution groups and their results
"""

import contextlib
import datetime
import pathlib
import shutil
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated
from urllib.parse import quote

import typer
from loguru import logger
from rich.console import Console, Group
from rich.filesize import decimal
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from climate_ref.cli._utils import (
    OutputFormat,
    df_to_table,
    format_size,
    parse_facet_filters,
    pretty_print_df,
    render_dataframe,
)
from climate_ref.config import Config
from climate_ref.models import Execution, ExecutionGroup
from climate_ref.models.execution import (
    ExecutionFailureReason,
    execution_datasets,
)
from climate_ref.results import (
    DiagnosticFilter,
    ExecutionGroupFilter,
    MetricValueFilter,
    OutlierPolicy,
    Reader,
    ResourceFilter,
)
from climate_ref.results.executions import OutputView
from climate_ref.results.queries import select_group_counts
from climate_ref.results.resources import ResourceProfile
from climate_ref.results.values import ValuesReader
from climate_ref_core.logging import EXECUTION_LOG_FILENAME
from climate_ref_core.paths import safe_path

app = typer.Typer(help=__doc__)


@dataclass
class ListGroupsFilterOptions:
    """Options to filter execution groups in list-groups command"""

    diagnostic: list[str] | None = None
    """Filter by diagnostic slug (substring, case-insensitive)"""

    provider: list[str] | None = None
    """Filter by provider slug (substring, case-insensitive)"""

    selectors: dict[str, list[str]] | None = None
    """Filter by execution group selector key-value pairs (AND across keys, OR within key)"""

    failure_reasons: list[ExecutionFailureReason] | None = None
    """Filter by the latest execution's failure reason"""


class ValueKind(StrEnum):
    """Which kind of metric value the ``values`` command should show."""

    scalar = "scalar"
    series = "series"


class ResourceGroupBy(StrEnum):
    """Axis the ``resources`` command aggregates over."""

    diagnostic = "diagnostic"
    provider = "provider"


_DIAGNOSTIC_ARG = Annotated[
    list[str] | None,
    typer.Option(
        help="Filter by diagnostic slug (substring match, case-insensitive). Multiple values can be provided."
    ),
]
_PROVIDER_ARG = Annotated[
    list[str] | None,
    typer.Option(
        help="Filter by provider slug (substring match, case-insensitive). Multiple values can be provided."
    ),
]
_SINCE_ARG = Annotated[
    str | None,
    typer.Option(help="Only consider executions newer than this age (90d, 12h, 4w) or ISO date."),
]


@contextlib.contextmanager
def _exit_on_value_error() -> Iterator[None]:
    """Turn a ``ValueError`` from bad input into a logged error and exit code 1."""
    try:
        yield
    except ValueError as e:
        logger.error(str(e))
        raise typer.Exit(code=1)


def _since_cutoff(since: str | None) -> datetime.datetime | None:
    """Parse ``--since``, exiting with an error when it is invalid."""
    if since is None:
        return None
    with _exit_on_value_error():
        return _parse_since(since)


# Default columns for the ``list-groups`` table.
_DEFAULT_LIST_GROUPS_COLUMNS = [
    "id",
    "key",
    "provider",
    "diagnostic",
    "dirty",
    "successful",
    "failure_reason",
    "created_at",
]


@app.command()
def list_groups(  # noqa: PLR0913
    ctx: typer.Context,
    column: Annotated[
        list[str] | None,
        typer.Option(help="Only include specified columns in the output"),
    ] = None,
    limit: int = typer.Option(100, min=0, help="Limit the number of rows to display"),
    diagnostic: _DIAGNOSTIC_ARG = None,
    provider: _PROVIDER_ARG = None,
    since: _SINCE_ARG = None,
    filter: Annotated[  # noqa: A002
        list[str] | None,
        typer.Option(
            "--filter",
            help="Filter by group selector key=value pairs (exact match). Multiple filters can be provided.",
        ),
    ] = None,
    successful: Annotated[
        bool | None,
        typer.Option(
            "--successful/--not-successful",
            help="Filter by successful or unsuccessful executions.",
        ),
    ] = None,
    dirty: Annotated[
        bool | None,
        typer.Option(
            "--dirty/--not-dirty",
            help="Filter to include only dirty or clean execution groups."
            "These execution groups will be re-computed on the next run.",
        ),
    ] = None,
    failure_reason: Annotated[
        list[ExecutionFailureReason] | None,
        typer.Option(
            help="Filter to groups whose latest execution failed for this reason. "
            "Multiple values can be provided."
        ),
    ] = None,
    output_format: Annotated[
        OutputFormat,
        typer.Option(
            "--format",
            help="Output format: 'table' (default) or machine-readable 'json'.",
        ),
    ] = OutputFormat.table,
) -> None:
    """
    List the diagnostic execution groups that have been identified

    Groups are sorted by id, so the oldest comes first.
    Only groups at each diagnostic's promoted version are listed,
    and a group whose newest execution is retracted is left out.
    With ``--since``, only groups that ran since then are listed, judged by their newest run in that time.

    By default the table shows a subset of columns with the verbose ``selectors``
    and ``updated_at`` fields omitted. Use ``--column`` to choose exactly which columns to display,
    or ``--format json`` to emit the full record for scripting.

    Filters can be combined using AND logic across filter types and OR logic within a filter type.

    The output is a table by default, or machine-readable JSON with ``--format json``.
    """
    session = ctx.obj.database.session
    console = ctx.obj.console
    reader = Reader(ctx.obj.database)

    with _exit_on_value_error():
        selector_filters = parse_facet_filters(filter)
        group_filter = ExecutionGroupFilter(
            diagnostic_contains=diagnostic,
            provider_contains=provider,
            selectors=selector_filters or None,
            successful=successful,
            dirty=dirty,
            failure_reasons=failure_reason,
            created_after=_since_cutoff(since),
        )

    # Build filter options
    filters = ListGroupsFilterOptions(
        diagnostic=diagnostic,
        provider=provider,
        selectors=selector_filters or None,
        failure_reasons=failure_reason,
    )
    logger.debug(f"Applying filters: {filters}")

    # Get total count before filtering for warning messages
    total_count = session.query(ExecutionGroup).count()

    # Apply filters to query
    collection = reader.executions.groups(group_filter, limit=limit)

    # Check if any results found
    if len(collection) == 0:
        emit_no_results_warning(filters, total_count)

    results_df = collection.to_pandas()

    # Select columns to display.
    if column:
        if not results_df.empty:  # Only validate against actual data
            if not all(col in results_df.columns for col in column):
                logger.error(f"Column not found in data catalog: {column}")
                raise typer.Exit(code=1)
            results_df = results_df[column]
    elif output_format == OutputFormat.table:
        results_df = results_df[_DEFAULT_LIST_GROUPS_COLUMNS]

    # Display results
    render_dataframe(results_df, console=console, output_format=output_format)

    # Show limit warning if applicable
    filtered_count = collection.total_count or 0
    if filtered_count > limit:
        logger.warning(
            f"Displaying {limit} of {filtered_count} filtered results. "
            f"Use the `--limit` option to display more."
        )


@app.command()
def delete_groups(  # noqa: PLR0912, PLR0913
    ctx: typer.Context,
    diagnostic: _DIAGNOSTIC_ARG = None,
    provider: _PROVIDER_ARG = None,
    filter: Annotated[  # noqa: A002
        list[str] | None,
        typer.Option(
            "--filter",
            help="Filter by group selector key=value pairs (exact match). Multiple filters can be provided.",
        ),
    ] = None,
    successful: Annotated[
        bool | None,
        typer.Option(
            "--successful/--not-successful",
            help="Filter by successful or unsuccessful executions.",
        ),
    ] = None,
    dirty: Annotated[
        bool | None,
        typer.Option(
            "--dirty/--not-dirty",
            help="Filter to include only dirty or clean execution groups."
            "These execution groups will be re-computed on the next run.",
        ),
    ] = None,
    remove_outputs: bool = typer.Option(
        False,
        "--remove-outputs",
        help="Also remove output directories from the filesystem",
    ),
    force: bool = typer.Option(False, help="Skip confirmation prompt"),
) -> None:
    """
    Delete execution groups matching the specified filters.

    This command will delete execution groups and their associated executions.
    Use filters to specify which groups to delete. At least one filter must be provided
    to prevent accidental deletion of all groups.

    Filters can be combined using AND logic across filter types and OR logic within a filter type.
    """
    import pandas as pd

    session = ctx.obj.database.session
    console = ctx.obj.console

    # Parse selector filters
    with _exit_on_value_error():
        selector_filters = parse_facet_filters(filter)
        group_filter = ExecutionGroupFilter(
            diagnostic_contains=diagnostic,
            provider_contains=provider,
            selectors=selector_filters or None,
            successful=successful,
            dirty=dirty,
            include_retracted=True,
        )

    if not any([diagnostic, provider, selector_filters, successful is not None, dirty is not None]):
        logger.warning("THIS WILL DELETE ALL EXECUTION GROUPS IN THE DATABASE")
        raise typer.Exit(code=1)

    # Build filter options
    filters = ListGroupsFilterOptions(
        diagnostic=diagnostic,
        provider=provider,
        selectors=selector_filters or None,
    )
    logger.debug(f"Applying filters: {filters}")

    # Apply filters to query
    all_filtered_results = Reader(ctx.obj.database).executions.group_rows(group_filter)

    # Check if any results found
    if not all_filtered_results:
        emit_no_results_warning(filters, session.query(ExecutionGroup).count())
        return

    # Convert to DataFrame for preview
    results_df = pd.DataFrame(
        [
            {
                "id": eg.id,
                "key": eg.key,
                "provider": eg.diagnostic.provider.slug,
                "diagnostic": eg.diagnostic.slug,
                "dirty": eg.dirty,
                "successful": result.successful if result else None,
                "created_at": eg.created_at,
                "updated_at": eg.updated_at,
                "selectors": eg.selectors,
            }
            for eg, result in all_filtered_results
        ]
    )

    # Display preview
    console.print("Execution groups to be deleted:")
    pretty_print_df(results_df, console=console)

    count = len(all_filtered_results)
    console.print(f"\nWill delete {count} execution group(s).")

    # Confirm unless force is set
    if not force:
        if not typer.confirm("Do you want to proceed with deletion?"):
            console.print("Deletion cancelled.")
            return

    # Remove output directories if requested
    if remove_outputs:
        config = ctx.obj.config
        for eg, _ in all_filtered_results:
            for execution in eg.executions:
                try:
                    output_dir = safe_path(
                        execution.output_fragment, base=config.paths.results, label="results"
                    )
                except ValueError:
                    logger.error(f"Skipping unsafe output fragment: {execution.output_fragment!r}")
                    continue

                if output_dir.exists():
                    try:
                        logger.warning(f"Removing output directory: {output_dir}")
                        shutil.rmtree(output_dir)
                    except Exception as e:
                        logger.error(f"Failed to remove {output_dir}: {e}")

    # Delete execution groups and all related records
    # TODO: Add cascade delete to FK relationships and simplify this code
    with session.begin_nested() if session.in_transaction() else session.begin():
        for eg, _ in all_filtered_results:
            for execution in eg.executions:
                # Delete MetricValues first
                for metric_value in execution.values:
                    session.delete(metric_value)

                # Delete ExecutionOutputs
                for output in execution.outputs:
                    session.delete(output)

                # Delete many-to-many associations with datasets
                session.execute(
                    execution_datasets.delete().where(execution_datasets.c.execution_id == execution.id)
                )

                # Now delete the execution
                session.delete(execution)

            # Finally delete the execution group
            session.delete(eg)

    if remove_outputs:
        console.print(f"[green]Successfully deleted {count} execution group(s) and their output directories.")
    else:
        console.print(f"[green]Successfully deleted {count} execution group(s).")


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


def _execution_panel(execution_group: ExecutionGroup) -> Panel:
    if len(execution_group.executions) == 0:
        result = None
    else:
        result = execution_group.executions[-1]

    failure_reason = (result.failure_reason if result else None) or "-"
    panel = Panel(
        f"Key: [bold]{execution_group.key}[/]\n"
        f"Diagnostic: [bold]{execution_group.diagnostic.slug}[/]\n"
        f"Provider: [bold]{execution_group.diagnostic.provider.slug}[/]\n"
        f"Dirty: [bold]{execution_group.dirty}[/]\n"
        f"Successful: [bold]{result.successful if result else 'not-started'}[/]\n"
        f"Retracted: [bold]{result.retracted if result else '-'}[/]\n"
        f"Failure reason: [bold]{failure_reason}[/]\n"
        f"Created At: [bold]{execution_group.created_at}[/]\n"
        f"Updated At: [bold]{execution_group.updated_at}[/]\n"
        f"Number of attempted executions: [bold]{len(execution_group.executions)}[/]",
        title=f"Execution Details: [bold]{execution_group.id}[/]",
    )
    return panel


def _datasets_panel(result: Execution) -> Panel:
    import pandas as pd

    datasets = result.datasets

    datasets_df = pd.DataFrame(
        [
            {
                "id": dataset.id,
                "slug": dataset.slug,
                "dataset_type": dataset.dataset_type,
            }
            for dataset in datasets
        ]
    )

    return Panel(
        df_to_table(datasets_df),
        title=f"Datasets hash: {result.dataset_hash}",
    )


def _outputs_panel(outputs: tuple[OutputView, ...], output_fragment: str, reader: Reader) -> Panel:
    if not outputs:
        return Panel(Text("No registered outputs.", "bold red"), title="Outputs")

    table = Table("output_type", "short_name", "long_name", "filename")
    for out in outputs:
        if out.filename is None:
            filename_cell: Text | str = ""
        else:
            try:
                output_path = reader.artifacts.output_file(output_fragment, out.filename)
                filename_cell = Text(out.filename, style=f"link file://{output_path}")
            except ValueError:
                filename_cell = out.filename
        table.add_row(
            out.output_type,
            out.short_name or "",
            out.long_name or "",
            filename_cell,
        )

    return Panel(table, title="Outputs")


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


def _log_panel(result_directory: pathlib.Path) -> Panel | None:
    log_file = result_directory / EXECUTION_LOG_FILENAME

    if log_file.exists():
        with open(log_file) as f:
            log_content = f.read()
        log_text = Text.from_markup(f"[link file://{log_file}]{log_content}")

        return Panel(
            log_text,
            title="Execution Logs",
        )
    else:
        return Panel(
            Text("Log file not found.", "bold red"),
            title="Execution Logs",
        )


def emit_no_results_warning(
    filters: ListGroupsFilterOptions,
    total_count: int,
) -> None:
    """
    Emit informative warning when filters produce no results.
    """
    filter_parts = []
    if filters.diagnostic:
        filter_parts.append(f"diagnostic filters: {filters.diagnostic}")
    if filters.provider:
        filter_parts.append(f"provider filters: {filters.provider}")
    if filters.selectors:
        selector_strs = [
            f"{k}={','.join(values)}" if len(values) > 1 else f"{k}={values[0]}"
            for k, values in filters.selectors.items()
        ]
        filter_parts.append(f"selector filters: {selector_strs}")
    if filters.failure_reasons:
        filter_parts.append(f"failure reasons: {[reason.value for reason in filters.failure_reasons]}")

    logger.warning(
        f"No execution groups match the specified filters. "
        f"Total execution groups in database: {total_count}. "
        f"Applied filters: {', '.join(filter_parts)}"
    )


@app.command()
def inspect(
    ctx: typer.Context,
    group_id: Annotated[int, typer.Argument(help="Execution group ID to inspect.")],
) -> None:
    """
    Inspect a specific execution group by its ID

    This will display the execution details, datasets, results directory, and logs if available.
    The details are for the group's newest execution, even when it is retracted.
    """
    config: Config = ctx.obj.config
    session = ctx.obj.database.session
    console = ctx.obj.console
    reader = Reader(ctx.obj.database, results=config.paths.results)

    execution_group = session.get(ExecutionGroup, group_id)

    if not execution_group:
        logger.error(f"Execution group not found: {group_id}")
        raise typer.Exit(code=1)

    console.print(_execution_panel(execution_group))

    if not execution_group.executions:
        logger.warning(f"Execution group {group_id} has not run yet.")
        return

    result: Execution = execution_group.executions[-1]
    output_dir = reader.artifacts.output_directory(result.output_fragment)

    console.print(_datasets_panel(result))
    console.print(_outputs_panel(reader.executions.outputs(result.id), result.output_fragment, reader))
    console.print(_results_directory_panel(output_dir))
    console.print(_log_panel(output_dir))


@app.command()
def values(  # noqa: PLR0913
    ctx: typer.Context,
    group_id: Annotated[int, typer.Argument(help="Execution group ID to show metric values for.")],
    kind: Annotated[
        ValueKind,
        typer.Option(help="Which kind of value to show: scalar (default) or series."),
    ] = ValueKind.scalar,
    execution_id: Annotated[
        int | None,
        typer.Option(help="Show values for a specific execution instead of the latest one in the group."),
    ] = None,
    dimension: Annotated[
        list[str] | None,
        typer.Option(
            "--dimension",
            "-d",
            help="Filter by CV dimension key=value (exact match). Repeat for AND across keys; "
            "repeat the same key for OR within a key.",
        ),
    ] = None,
    outliers: Annotated[
        bool,
        typer.Option(
            "--outliers/--no-outliers",
            help="Run source-id-aware IQR outlier detection (scalar values only).",
        ),
    ] = False,
    include_unverified: Annotated[
        bool,
        typer.Option(
            "--include-unverified",
            help="Include values flagged as outliers (only meaningful with --outliers).",
        ),
    ] = False,
    limit: Annotated[
        int,
        typer.Option(min=0, help="Maximum number of values to display."),
    ] = 100,
    offset: Annotated[
        int,
        typer.Option(min=0, help="Number of values to skip before displaying."),
    ] = 0,
    output_format: Annotated[
        OutputFormat,
        typer.Option("--format", help="Output format: table (default) or json."),
    ] = OutputFormat.table,
) -> None:
    """
    Show computed metric values for an execution group.

    By default this shows the scalar values from the latest execution in the group.
    Use --kind series for 1-d series, --execution-id to target a specific execution,
    and --dimension key=value to filter by controlled-vocabulary dimensions.

    Values are read through the shared climate_ref.results layer, so the same filtering
    is applied here as in the API.
    """
    database = ctx.obj.database
    console = ctx.obj.console
    reader = Reader(database)

    group = database.session.get(ExecutionGroup, group_id)
    if group is None:
        logger.error(f"Execution group not found: {group_id}")
        raise typer.Exit(code=1)

    if execution_id is None:
        latest = reader.executions.latest_execution(group_id)
        if latest is None:
            logger.error(f"No executions found for group {group_id}.")
            raise typer.Exit(code=1)
        execution_id = latest.id
    else:
        execution = database.session.get(Execution, execution_id)
        if execution is None or execution.execution_group_id != group_id:
            logger.error(f"Execution {execution_id} does not belong to group {group_id}.")
            raise typer.Exit(code=1)

    try:
        dimensions = parse_facet_filters(dimension)
    except ValueError as e:
        logger.error(str(e))
        raise typer.Exit(code=1)

    # Scope to the concrete execution: promotion/retraction gating would otherwise hide the
    # values the user explicitly asked to inspect.
    filters = MetricValueFilter(
        execution_ids=[execution_id],
        dimensions=dimensions or None,
        include_superseded=True,
        include_retracted=True,
    )

    try:
        if kind == ValueKind.scalar:
            _render_scalar_values(
                reader.values,
                filters,
                console=console,
                output_format=output_format,
                outliers=outliers,
                include_unverified=include_unverified,
                offset=offset,
                limit=limit,
            )
        else:
            _render_series_values(
                reader.values,
                filters,
                console=console,
                output_format=output_format,
                offset=offset,
                limit=limit,
                outliers=outliers,
                include_unverified=include_unverified,
            )
    except KeyError as e:
        # Raised by the filter when a --dimension key is not a registered CV dimension.
        logger.error(f"Unknown dimension: {e}")
        raise typer.Exit(code=1)


def _render_scalar_values(  # noqa: PLR0913
    values: ValuesReader,
    filters: MetricValueFilter,
    *,
    console: Console,
    output_format: OutputFormat,
    outliers: bool,
    include_unverified: bool,
    offset: int,
    limit: int | None,
) -> None:
    collection = values.scalar_values(
        filters,
        outliers=OutlierPolicy() if outliers else None,
        include_unverified=include_unverified,
        offset=offset,
        limit=limit,
        with_facets=False,
    )
    if not len(collection):
        if output_format == OutputFormat.json:
            render_dataframe(collection.to_pandas(), output_format=output_format)
        elif not collection.had_outliers:
            console.print("No scalar values found.")
        if collection.had_outliers:
            logger.warning(
                f"No scalar values found. {collection.outlier_count} value(s) were flagged as "
                f"outliers and hidden. Use --include-unverified to show them."
            )
        return

    render_dataframe(collection.to_pandas(), console=console, output_format=output_format)

    if collection.had_outliers:
        state = "shown" if include_unverified else "hidden"
        logger.warning(
            f"{collection.outlier_count} value(s) flagged as outliers ({state}). "
            f"Use --include-unverified to show them."
        )
    displayed = offset + len(collection)
    if displayed < (collection.total_count or 0):
        logger.warning(
            f"Displaying {len(collection)} of {collection.total_count} values. "
            f"Use --limit / --offset to see more."
        )


def _render_series_values(  # noqa: PLR0913
    values: ValuesReader,
    filters: MetricValueFilter,
    *,
    console: Console,
    output_format: OutputFormat,
    offset: int,
    limit: int | None,
    outliers: bool = False,
    include_unverified: bool = False,
) -> None:
    # Warn if scalar-only flags are used with series values
    ignored_flags = []
    if outliers:
        ignored_flags.append("--outliers")
    if include_unverified:
        ignored_flags.append("--include-unverified")
    if ignored_flags:
        logger.warning(f"{'/'.join(ignored_flags)} only apply to scalar values and were ignored.")

    collection = values.series_values(filters, offset=offset, limit=limit, with_facets=False)
    if not len(collection):
        if output_format == OutputFormat.json:
            render_dataframe(collection.to_pandas(), output_format=output_format)
        else:
            console.print("No series values found.")
        return

    if output_format == OutputFormat.json:
        # Full long-form data for scripting.
        render_dataframe(collection.to_pandas(explode=True), output_format=output_format)
    else:
        # Compact one-row-per-series summary for the terminal; the raw arrays are too wide.
        df = collection.to_pandas(explode=False)
        df["n_points"] = df["values"].apply(len)
        df = df.drop(columns=["values", "index"])
        render_dataframe(df, console=console, output_format=output_format)

    displayed = offset + len(collection)
    if displayed < (collection.total_count or 0):
        logger.warning(
            f"Displaying {len(collection)} of {collection.total_count} series. "
            f"Use --limit / --offset to see more."
        )


@app.command()
def fail_running(
    ctx: typer.Context,
    older_than: Annotated[
        float | None,
        typer.Option(
            help="Only fail executions older than this many hours. "
            "If not specified, all running executions are failed."
        ),
    ] = None,
    diagnostic: _DIAGNOSTIC_ARG = None,
    provider: _PROVIDER_ARG = None,
    force: bool = typer.Option(False, help="Skip confirmation prompt"),
) -> None:
    """
    Mark running executions as failed.

    Running executions (those with no success/failure status) block their execution group
    from being requeued. Use this command as an escape hatch to fail stuck executions
    so they can be retried on the next solve.

    An optional age threshold can be provided with --older-than to only fail executions
    that have been running for longer than the specified number of hours.
    """
    session = ctx.obj.database.session
    console = ctx.obj.console

    # Query for running executions (successful IS NULL)
    query = session.query(Execution).filter(Execution.successful.is_(None))

    # Apply age threshold
    if older_than is not None:
        cutoff = datetime.datetime.now(tz=datetime.UTC) - datetime.timedelta(hours=older_than)
        query = query.filter(Execution.created_at < cutoff)

    if diagnostic or provider:
        scope = DiagnosticFilter(diagnostic_contains=diagnostic, provider_contains=provider)
        query = query.join(ExecutionGroup, Execution.execution_group_id == ExecutionGroup.id).filter(
            ExecutionGroup.diagnostic_id.in_(scope.diagnostic_ids())
        )

    running_executions = query.all()

    if not running_executions:
        console.print("No running executions found matching the specified criteria.")
        return

    console.print(f"Found {len(running_executions)} running execution(s) to mark as failed:")
    for execution in running_executions:
        eg = execution.execution_group
        console.print(
            f"  Execution {execution.id}: "
            f"{eg.diagnostic.provider.slug}/{eg.diagnostic.slug} "
            f"(group={eg.key}, created={execution.created_at})"
        )

    if not force:
        if not typer.confirm("Do you want to mark these executions as failed?"):
            console.print("Cancelled.")
            return

    with session.begin_nested() if session.in_transaction() else session.begin():
        for execution in running_executions:
            execution.mark_failed(ExecutionFailureReason.Cancelled)
            execution.execution_group.dirty = True
            logger.info(
                f"Marked execution {execution.id} as failed "
                f"({execution.execution_group.diagnostic.slug}/{execution.execution_group.key})"
            )

    console.print(
        f"[green]Successfully marked {len(running_executions)} execution(s) as failed "
        f"and flagged their execution groups as dirty."
    )


@app.command()
def stats(
    ctx: typer.Context,
    diagnostic: _DIAGNOSTIC_ARG = None,
    provider: _PROVIDER_ARG = None,
    since: _SINCE_ARG = None,
) -> None:
    """
    Show how many execution groups each diagnostic has, by the status of their newest execution.

    The statuses are running, failed, successful and not started, plus a dirty count that overlaps them.
    Only groups at each diagnostic's promoted version are counted, and retracted executions are skipped.
    Diagnostics with no groups are listed with zero counts, and each provider gets a ``(total)`` row.
    With ``--since``, only groups that ran since then are counted, by their newest run in that time.
    """
    import pandas as pd

    console = ctx.obj.console
    rows = ctx.obj.database.session.execute(
        select_group_counts(
            DiagnosticFilter(
                diagnostic_contains=diagnostic,
                provider_contains=provider,
                created_after=_since_cutoff(since),
            )
        )
    ).all()

    if not rows:
        console.print("No execution groups found.")
        return

    status_cols = ["running", "failed", "successful", "not_started", "dirty", "total"]
    results_df = pd.DataFrame([row._mapping for row in rows]).rename(
        columns={"provider_slug": "provider", "diagnostic_slug": "diagnostic", "group_count": "total"}
    )[["provider", "diagnostic", *status_cols]]

    # Add provider totals
    totals = results_df.groupby("provider")[status_cols].sum().reset_index()
    totals["diagnostic"] = "(total)"

    results_df = pd.concat([results_df, totals], ignore_index=True)
    results_df = results_df.sort_values(["provider", "diagnostic"]).reset_index(drop=True)

    pretty_print_df(results_df, console=console)


_SECONDS_PER_UNIT = {"h": 3600, "d": 86400, "w": 604800}


def _parse_since(value: str) -> datetime.datetime:
    """
    Parse a ``--since`` value into a naive UTC cutoff.

    Accepts a relative age with a unit suffix (``12h``, ``90d``, ``4w``)
    or an ISO date such as ``2026-07-01``.

    Parameters
    ----------
    value
        The raw option string.

    Returns
    -------
    :
        The cutoff, naive and in UTC, matching how ``created_at`` is stored.

    Raises
    ------
    ValueError
        If the string is neither a relative age nor an ISO date.
    """
    suffix = value[-1:].lower()
    if suffix in _SECONDS_PER_UNIT:
        try:
            magnitude = float(value[:-1])
        except ValueError:
            raise ValueError(f"Invalid --since value: {value!r}")
        now = datetime.datetime.now(tz=datetime.UTC).replace(tzinfo=None)
        return now - datetime.timedelta(seconds=magnitude * _SECONDS_PER_UNIT[suffix])

    try:
        return datetime.datetime.fromisoformat(value)
    except ValueError:
        raise ValueError(
            f"Invalid --since value: {value!r}. Expected a relative age (90d, 12h, 4w) or an ISO date."
        )


def _format_duration(seconds: float) -> str:
    """Render a duration compactly, in the largest unit that keeps it readable."""
    if seconds < 90:  # noqa: PLR2004
        return f"{seconds:.0f}s"
    if seconds < 5400:  # noqa: PLR2004
        return f"{seconds / 60:.0f}m"
    return f"{seconds / 3600:.1f}h"


def _resource_table_row(profile: ResourceProfile, label: str) -> dict[str, str | int]:
    limit = profile.memory_limit_seen
    headroom = profile.headroom_ratio
    return {
        # The first 8 columns are the ones an 80-column terminal keeps.
        label: profile.diagnostic_slug if profile.diagnostic_slug is not None else profile.provider_slug,
        "flag": "OVER" if profile.over_limit else "",
        "conf": profile.confidence,
        "n": profile.sample_count,
        "fail": profile.failed_execution_count,
        "rec mem": format_size(profile.recommended_memory_bytes),
        "rec cpu": profile.recommended_cpus,
        "headroom": f"{headroom:.1f}x" if headroom is not None else "-",
        "mem max": format_size(profile.peak_memory_bytes_max),
        "mem p95": format_size(profile.peak_memory_bytes_p95),
        "mem p50": format_size(profile.peak_memory_bytes_p50),
        "wall p95": _format_duration(profile.wall_seconds_p95),
        "cpu/wall": f"{profile.parallelism_p95:.1f}",
        "limit": format_size(limit) if limit is not None else "-",
    }


def _resource_json_record(profile: ResourceProfile) -> dict[str, object]:
    return {
        "provider": profile.provider_slug,
        "diagnostic": profile.diagnostic_slug,
        "sample_count": profile.sample_count,
        "excluded_execution_count": profile.excluded_execution_count,
        "failed_execution_count": profile.failed_execution_count,
        "memory_source": profile.memory_source,
        "peak_memory_bytes_p50": profile.peak_memory_bytes_p50,
        "peak_memory_bytes_p95": profile.peak_memory_bytes_p95,
        "peak_memory_bytes_max": profile.peak_memory_bytes_max,
        "wall_seconds_p95": profile.wall_seconds_p95,
        "cpu_seconds_p95": profile.cpu_seconds_p95,
        "parallelism_p95": profile.parallelism_p95,
        "memory_limit_seen": profile.memory_limit_seen,
        "headroom_ratio": profile.headroom_ratio,
        "recommended_memory_bytes": profile.recommended_memory_bytes,
        "recommended_cpus": profile.recommended_cpus,
        "confidence": profile.confidence,
        "over_limit": profile.over_limit,
    }


def _warn_about_bias(profiles: Sequence[ResourceProfile]) -> None:
    """
    Surface the reasons a recommendation may be lower than the truth.

    The table carries the same signals in its fail, conf and flag columns.
    These warnings repeat them on stderr, which is the only place a ``--format json`` caller sees them.
    """
    over = [p for p in profiles if p.over_limit]
    if over:
        logger.warning(
            f"{len(over)} row(s) flagged OVER: the p95 peak has outgrown the recorded memory limit, "
            f"so those executions are being killed by the current container size."
        )

    failed = sum(p.failed_execution_count for p in profiles)
    if failed:
        logger.warning(
            f"{failed} failed execution(s) in this window. A failure that recorded a measurement "
            f"is counted here, but a killed process records nothing at all, "
            f"so the heaviest runs may be missing and the recommendations biased low."
        )

    thin = [p for p in profiles if p.confidence != "good"]
    if thin:
        logger.warning(
            f"{len(thin)} row(s) rest on fewer than 10 samples (see the conf column). "
            f"Their p95 is the maximum by another name."
        )

    excluded = sum(p.excluded_execution_count for p in profiles)
    if excluded:
        logger.warning(
            f"{excluded} measured execution(s) were excluded as incomplete, "
            f"carrying a different memory source, or carrying a cgroup reading from a shared container. "
            f"Use --include-shared to count the shared ones."
        )


@app.command()
def resources(  # noqa: PLR0913
    ctx: typer.Context,
    diagnostic: _DIAGNOSTIC_ARG = None,
    provider: _PROVIDER_ARG = None,
    by: Annotated[
        ResourceGroupBy,
        typer.Option(help="Aggregate per diagnostic (default) or roll up per provider."),
    ] = ResourceGroupBy.diagnostic,
    since: _SINCE_ARG = None,
    safety: Annotated[
        float,
        typer.Option(help="Multiplier applied to the p95 peak when recommending a memory size."),
    ] = 1.3,
    include_shared: Annotated[
        bool,
        typer.Option(
            "--include-shared",
            help="Also count cgroup readings taken while other executions shared the worker. "
            "Those readings cover the whole worker, not the execution.",
        ),
    ] = False,
    output_format: Annotated[
        OutputFormat,
        typer.Option("--format", help="Output format: table (default) or json."),
    ] = OutputFormat.table,
) -> None:
    """
    Show measured resource usage and a recommended worker size.

    One row per diagnostic by default, sorted by recommended memory descending.
    Use --by provider for the roll-up a worker actually has to fit,
    which takes the largest memory over the provider's diagnostics.

    Three columns qualify the recommendation and should be read with it.
    The OVER flag marks a p95 peak that has outgrown the recorded memory limit.
    The fail column counts failed executions, which record nothing when they are killed,
    so a high count means the numbers here are lower than the truth.
    The conf column reports how many samples the percentiles rest on.
    """
    import pandas as pd

    console = ctx.obj.console
    reader = Reader(ctx.obj.database)

    profiles = reader.resources.profiles(
        ResourceFilter(
            diagnostic_contains=diagnostic,
            provider_contains=provider,
            created_after=_since_cutoff(since),
            # Sizing draws on every diagnostic version, not only the promoted one.
            include_superseded=True,
        ),
        group_by=by.value,
        exclusive_only=not include_shared,
        safety_factor=safety,
    )

    if not profiles:
        if output_format == OutputFormat.json:
            render_dataframe(pd.DataFrame(), output_format=output_format)
        else:
            console.print("No resource measurements found.")
        return

    ordered = sorted(
        profiles,
        key=lambda p: (-p.recommended_memory_bytes, p.provider_slug, p.diagnostic_slug or ""),
    )

    if output_format == OutputFormat.json:
        render_dataframe(
            pd.DataFrame.from_records([_resource_json_record(p) for p in ordered]),
            output_format=output_format,
        )
        _warn_about_bias(ordered)
        return

    label = "provider" if by == ResourceGroupBy.provider else "diagnostic"
    render_dataframe(
        pd.DataFrame.from_records([_resource_table_row(p, label) for p in ordered]),
        console=console,
        output_format=output_format,
    )
    _warn_about_bias(ordered)


# `get_executions_for_reingest` stays on the ORM helper by design.
# it passes`include_superseded=True` and selects `eg.executions[0]`
# (the oldest / original execution in the group),
# which is a different "latest" than `reader.executions` definition.
# Revisit once a mutable query surface exists.
@app.command()
def reingest(  # noqa: PLR0913
    ctx: typer.Context,
    group_ids: Annotated[
        list[int] | None,
        typer.Argument(help="Execution group IDs to reingest. If omitted, uses filters."),
    ] = None,
    diagnostic: _DIAGNOSTIC_ARG = None,
    provider: _PROVIDER_ARG = None,
    include_failed: Annotated[
        bool,
        typer.Option(
            "--include-failed",
            help="Also attempt reingest on failed executions.",
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Show what would be reingested without making changes.",
        ),
    ] = False,
    force: bool = typer.Option(False, help="Skip confirmation prompt"),
) -> None:
    """
    Reingest existing executions without re-running diagnostics.

    Re-runs build_execution_result() on existing output files and re-ingests
    the results into the database. Useful when new series definitions or
    metadata extraction logic has been added.

    A new Execution record is always created under the same ExecutionGroup,
    leaving the original execution untouched. Results are treated as immutable.

    The dirty flag is never modified by this command.
    """
    import pandas as pd

    from climate_ref.executor.reingest import (
        get_executions_for_reingest,
        reingest_execution,
    )
    from climate_ref.provider_registry import ProviderRegistry

    config: Config = ctx.obj.config
    db = ctx.obj.database
    console = ctx.obj.console

    if not any([group_ids, provider, diagnostic]):
        logger.error(
            "At least one filter is required (group IDs, --provider, or --diagnostic). "
            "This prevents accidental reingest of all executions."
        )
        raise typer.Exit(code=1)

    provider_registry = ProviderRegistry.build_from_config(config, db)

    results = get_executions_for_reingest(
        db,
        execution_group_ids=group_ids,
        provider_filters=provider,
        diagnostic_filters=diagnostic,
        include_failed=include_failed,
    )

    if not results:
        console.print("No executions found matching the specified criteria.")
        return

    preview_df = pd.DataFrame(
        [
            {
                "group_id": eg.id,
                "execution_id": ex.id,
                "provider": eg.diagnostic.provider.slug,
                "diagnostic": eg.diagnostic.slug,
                "key": eg.key,
                "successful": ex.successful,
            }
            for eg, ex in results
        ]
    )

    if dry_run:
        console.print(f"[bold]Dry run:[/] would reingest {len(results)} execution(s):")
        pretty_print_df(preview_df, console=console)
        return

    console.print(f"Will reingest {len(results)} execution(s):")
    pretty_print_df(preview_df, console=console)

    if not force:
        if not typer.confirm("\nProceed with reingest?"):
            console.print("Reingest cancelled.")
            return

    # Ensure any autobegun transaction from the preview queries is closed
    # so each reingest runs in its own top-level transaction (not a savepoint).
    if db.session.in_transaction():
        db.session.commit()

    # Process each execution in a separate transaction
    success_count = 0
    skip_count = 0
    for eg, ex in results:
        with db.session.begin():
            ok = reingest_execution(
                config=config,
                database=db,
                execution=ex,
                provider_registry=provider_registry,
            )

        if ok:
            success_count += 1
        else:
            skip_count += 1

    console.print(f"\n[green]Reingest complete:[/] {success_count} succeeded, {skip_count} skipped.")


@app.command()
def flag_dirty(ctx: typer.Context, execution_id: int) -> None:
    """
    Flag an execution group for recomputation
    """
    session = ctx.obj.database.session
    console = ctx.obj.console
    with session.begin():
        execution_group = session.get(ExecutionGroup, execution_id)

        if not execution_group:
            logger.error(f"Execution not found: {execution_id}")
            raise typer.Exit(code=1)

        execution_group.dirty = True

        console.print(_execution_panel(execution_group))


@app.command()
def requeue(  # noqa: PLR0913
    ctx: typer.Context,
    failure_reason: Annotated[
        list[ExecutionFailureReason] | None,
        typer.Option(
            help="Only requeue groups whose latest execution failed for this reason. "
            "Multiple values can be provided."
        ),
    ] = None,
    diagnostic: _DIAGNOSTIC_ARG = None,
    provider: _PROVIDER_ARG = None,
    since: _SINCE_ARG = None,
    filter: Annotated[  # noqa: A002
        list[str] | None,
        typer.Option(
            "--filter",
            help="Filter by group selector key=value pairs (exact match). Multiple filters can be provided.",
        ),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Show what would be requeued without making changes."),
    ] = False,
    force: bool = typer.Option(False, help="Skip confirmation prompt"),
) -> None:
    """
    Flag failed execution groups as dirty so the next solve reruns them.

    Only groups whose latest execution failed and that are not already dirty are selected.
    Use ``--failure-reason`` to requeue only one kind of failure,
    e.g. ``--failure-reason system_error`` after fixing a broken environment.
    Use ``--since`` to requeue only groups whose latest execution ran since then, e.g. ``--since 2d``.
    """
    import pandas as pd

    session = ctx.obj.database.session
    console = ctx.obj.console

    with _exit_on_value_error():
        selector_filters = parse_facet_filters(filter)
        group_filter = ExecutionGroupFilter(
            diagnostic_contains=diagnostic,
            provider_contains=provider,
            selectors=selector_filters or None,
            dirty=False,
            successful=False,
            failure_reasons=failure_reason,
            created_after=_since_cutoff(since),
            include_retracted=True,
        )

    if not any([failure_reason, diagnostic, provider, selector_filters, since]):
        logger.error(
            "At least one filter is required "
            "(--failure-reason, --diagnostic, --provider, --since or --filter). "
            "This prevents accidentally requeuing every failed execution."
        )
        raise typer.Exit(code=1)

    results = [
        (eg, execution)
        for eg, execution in Reader(ctx.obj.database).executions.group_rows(group_filter)
        # Drops in-progress and never-run groups, which `successful=False` also matches
        if execution is not None and execution.successful is False
    ]

    if not results:
        console.print("No failed execution groups match the specified filters.")
        return

    preview_df = pd.DataFrame(
        [
            {
                "id": eg.id,
                "key": eg.key,
                "provider": eg.diagnostic.provider.slug,
                "diagnostic": eg.diagnostic.slug,
                "failure_reason": execution.failure_reason,
            }
            for eg, execution in results
        ]
    )

    prefix = "[bold]Dry run:[/] would requeue" if dry_run else "Will requeue"
    console.print(f"{prefix} {len(results)} execution group(s):")
    pretty_print_df(preview_df, console=console)
    if dry_run:
        return

    if not force:
        if not typer.confirm("\nProceed with requeue?"):
            console.print("Requeue cancelled.")
            return

    # Close the selection query's autobegun transaction, so this write is a top-level commit
    if session.in_transaction():
        session.commit()
    with session.begin():
        for eg, _ in results:
            eg.dirty = True

    console.print(
        f"[green]Flagged {len(results)} execution group(s) as dirty.[/] "
        "They will be rerun on the next `ref solve`."
    )
