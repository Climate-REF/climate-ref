"""
View execution groups and their results
"""

import json
import pathlib
import shutil
from dataclasses import dataclass
from typing import Annotated
from urllib.parse import quote

import typer
from loguru import logger
from rich.console import Group
from rich.filesize import decimal
from rich.markup import escape
from rich.panel import Panel
from rich.text import Text
from rich.tree import Tree
from sqlalchemy import case, func, or_

from climate_ref.cli._utils import df_to_table, parse_facet_filters, pretty_print_df
from climate_ref.config import Config
from climate_ref.executor.reingest import ReingestMode
from climate_ref.models import Execution, ExecutionGroup
from climate_ref.models.diagnostic import Diagnostic
from climate_ref.models.execution import execution_datasets, get_execution_group_and_latest_filtered
from climate_ref.models.provider import Provider
from climate_ref_core.logging import EXECUTION_LOG_FILENAME

app = typer.Typer(help=__doc__)


@dataclass
class ListGroupsFilterOptions:
    """Options to filter execution groups in list-groups command"""

    diagnostic: list[str] | None = None
    """Filter by diagnostic slug (substring, case-insensitive)"""

    provider: list[str] | None = None
    """Filter by provider slug (substring, case-insensitive)"""

    facets: dict[str, str] | None = None
    """Filter by facet key-value pairs (exact match)"""


@app.command()
def list_groups(  # noqa: PLR0913
    ctx: typer.Context,
    column: Annotated[
        list[str] | None,
        typer.Option(help="Only include specified columns in the output"),
    ] = None,
    limit: int = typer.Option(100, help="Limit the number of rows to display"),
    diagnostic: Annotated[
        list[str] | None,
        typer.Option(
            help="Filter by diagnostic slug (substring match, case-insensitive)."
            "Multiple values can be provided."
        ),
    ] = None,
    provider: Annotated[
        list[str] | None,
        typer.Option(
            help="Filter by provider slug (substring match, case-insensitive)."
            "Multiple values can be provided."
        ),
    ] = None,
    filter: Annotated[  # noqa: A002
        list[str] | None,
        typer.Option(
            "--filter",
            help="Filter by facet key=value pairs (exact match). Multiple filters can be provided.",
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
) -> None:
    """
    List the diagnostic execution groups that have been identified

    The data catalog is sorted by the date that the execution group was created (first = newest).
    If the `--column` option is provided, only the specified columns will be displayed.

    Filters can be combined using AND logic across filter types and OR logic within a filter type.

    The output will be in a tabular format.
    """
    import pandas as pd

    session = ctx.obj.database.session
    console = ctx.obj.console

    # Parse facet filters
    try:
        facet_filters = parse_facet_filters(filter)
    except ValueError as e:
        logger.error(str(e))
        raise typer.Exit(code=1)

    # Build filter options
    filters = ListGroupsFilterOptions(
        diagnostic=diagnostic,
        provider=provider,
        facets=facet_filters if facet_filters else None,
    )
    logger.debug(f"Applying filters: {filters}")

    # Get total count before filtering for warning messages
    total_count = session.query(ExecutionGroup).count()

    # Apply filters to query
    try:
        all_filtered_results = get_execution_group_and_latest_filtered(
            session,
            diagnostic_filters=filters.diagnostic,
            provider_filters=filters.provider,
            facet_filters=filters.facets,
            successful=successful,
            dirty=dirty,
        )
        execution_groups_results = all_filtered_results[:limit]
    except Exception as e:  # pragma: no cover
        logger.error(f"Error applying filters: {e}")
        raise typer.Exit(code=1)

    # Check if any results found
    if not execution_groups_results:
        emit_no_results_warning(filters, total_count)
        results_df = pd.DataFrame(
            columns=[
                "id",
                "key",
                "provider",
                "diagnostic",
                "dirty",
                "successful",
                "created_at",
                "updated_at",
                "selectors",
            ]
        )
    else:
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
                    "selectors": json.dumps(eg.selectors),
                }
                for eg, result in execution_groups_results
            ]
        )

    # Apply column filtering
    if column and not results_df.empty:  # Only apply if df is not empty
        if not all(col in results_df.columns for col in column):
            logger.error(f"Column not found in data catalog: {column}")
            raise typer.Exit(code=1)
        results_df = results_df[column]

    # Display results
    pretty_print_df(results_df, console=console)

    # Show limit warning if applicable
    filtered_count = len(all_filtered_results)
    if filtered_count > limit:
        logger.warning(
            f"Displaying {limit} of {filtered_count} filtered results. "
            f"Use the `--limit` option to display more."
        )


@app.command()
def delete_groups(  # noqa: PLR0912, PLR0913, PLR0915
    ctx: typer.Context,
    diagnostic: Annotated[
        list[str] | None,
        typer.Option(
            help="Filter by diagnostic slug (substring match, case-insensitive)."
            "Multiple values can be provided."
        ),
    ] = None,
    provider: Annotated[
        list[str] | None,
        typer.Option(
            help="Filter by provider slug (substring match, case-insensitive)."
            "Multiple values can be provided."
        ),
    ] = None,
    filter: Annotated[  # noqa: A002
        list[str] | None,
        typer.Option(
            "--filter",
            help="Filter by facet key=value pairs (exact match). Multiple filters can be provided.",
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
        False, "--remove-outputs", help="Also remove output directories from the filesystem"
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

    # Parse facet filters
    try:
        facet_filters = parse_facet_filters(filter)
    except ValueError as e:
        logger.error(str(e))
        raise typer.Exit(code=1)

    if not any([diagnostic, provider, facet_filters, successful is not None, dirty is not None]):
        logger.warning("THIS WILL DELETE ALL EXECUTION GROUPS IN THE DATABASE")
        raise typer.Exit(code=1)

    # Build filter options
    filters = ListGroupsFilterOptions(
        diagnostic=diagnostic,
        provider=provider,
        facets=facet_filters if facet_filters else None,
    )
    logger.debug(f"Applying filters: {filters}")

    # Apply filters to query
    try:
        all_filtered_results = get_execution_group_and_latest_filtered(
            session,
            diagnostic_filters=filters.diagnostic,
            provider_filters=filters.provider,
            facet_filters=filters.facets,
            successful=successful,
            dirty=dirty,
        )
    except Exception as e:  # pragma: no cover
        logger.error(f"Error applying filters: {e}")
        raise typer.Exit(code=1)

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
                "selectors": json.dumps(eg.selectors),
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
                output_dir = config.paths.results / execution.output_fragment

                # Safety check
                if not output_dir.is_relative_to(config.paths.results):  # pragma: no cover
                    logger.error(f"Skipping unsafe path: {output_dir}")
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

    panel = Panel(
        f"Key: [bold]{execution_group.key}[/]\n"
        f"Diagnostic: [bold]{execution_group.diagnostic.slug}[/]\n"
        f"Provider: [bold]{execution_group.diagnostic.provider.slug}[/]\n"
        f"Dirty: [bold]{execution_group.dirty}[/]\n"
        f"Successful: [bold]{result.successful if result else 'not-started'}[/]\n"
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
    if filters.facets:
        facet_strs = [f"{k}={v}" for k, v in filters.facets.items()]
        filter_parts.append(f"facet filters: {facet_strs}")

    logger.warning(
        f"No execution groups match the specified filters. "
        f"Total execution groups in database: {total_count}. "
        f"Applied filters: {', '.join(filter_parts)}"
    )


@app.command()
def inspect(ctx: typer.Context, execution_id: int) -> None:
    """
    Inspect a specific execution group by its ID

    This will display the execution details, datasets, results directory, and logs if available.
    """
    config: Config = ctx.obj.config
    session = ctx.obj.database.session
    console = ctx.obj.console

    execution_group = session.get(ExecutionGroup, execution_id)

    if not execution_group:
        logger.error(f"Execution not found: {execution_id}")
        raise typer.Exit(code=1)

    console.print(_execution_panel(execution_group))

    if not execution_group.executions:
        logger.error(f"No results found for execution: {execution_id}")
        return

    result: Execution = execution_group.executions[-1]
    result_directory = config.paths.results / result.output_fragment

    console.print(_datasets_panel(result))
    console.print(_results_directory_panel(result_directory))
    console.print(_log_panel(result_directory))


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
    diagnostic: Annotated[
        list[str] | None,
        typer.Option(
            help="Filter by diagnostic slug (substring match, case-insensitive). "
            "Multiple values can be provided."
        ),
    ] = None,
    provider: Annotated[
        list[str] | None,
        typer.Option(
            help="Filter by provider slug (substring match, case-insensitive). "
            "Multiple values can be provided."
        ),
    ] = None,
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
    import datetime

    session = ctx.obj.database.session
    console = ctx.obj.console

    # Query for running executions (successful IS NULL)
    query = session.query(Execution).filter(Execution.successful.is_(None))

    # Apply age threshold
    if older_than is not None:
        cutoff = datetime.datetime.now(tz=datetime.timezone.utc) - datetime.timedelta(hours=older_than)
        query = query.filter(Execution.created_at < cutoff)

    # Apply diagnostic/provider filters via execution group -> diagnostic join
    if diagnostic or provider:
        query = query.join(ExecutionGroup, Execution.execution_group_id == ExecutionGroup.id).join(
            Diagnostic, ExecutionGroup.diagnostic_id == Diagnostic.id
        )

    if diagnostic:
        diagnostic_conditions = [
            Diagnostic.slug.ilike(f"%{filter_value.lower()}%") for filter_value in diagnostic
        ]
        query = query.filter(or_(*diagnostic_conditions))

    if provider:
        query = query.join(Provider, Diagnostic.provider_id == Provider.id)
        provider_conditions = [Provider.slug.ilike(f"%{filter_value.lower()}%") for filter_value in provider]
        query = query.filter(or_(*provider_conditions))

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
            execution.mark_failed()
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
    diagnostic: Annotated[
        list[str] | None,
        typer.Option(
            help="Filter by diagnostic slug (substring match, case-insensitive)."
            "Multiple values can be provided."
        ),
    ] = None,
    provider: Annotated[
        list[str] | None,
        typer.Option(
            help="Filter by provider slug (substring match, case-insensitive)."
            "Multiple values can be provided."
        ),
    ] = None,
) -> None:
    """
    Show summary statistics for execution groups.

    Displays counts of executions grouped by provider, broken down by status
    (running, failed, successful, not started, dirty).
    """
    import pandas as pd

    session = ctx.obj.database.session
    console = ctx.obj.console

    # Subquery to get the latest execution per execution group
    latest_exec_subquery = (
        session.query(
            Execution.execution_group_id,
            func.max(Execution.created_at).label("latest_created_at"),
        )
        .group_by(Execution.execution_group_id)
        .subquery()
    )

    # Build query: join ExecutionGroup -> Diagnostic -> Provider, and optionally latest Execution
    query = (
        session.query(
            Provider.slug.label("provider"),
            Diagnostic.slug.label("diagnostic"),
            func.count().label("total"),
            func.sum(
                case(
                    (Execution.successful.is_(None) & Execution.id.isnot(None), 1),
                    else_=0,
                )
            ).label("running"),
            func.sum(case((Execution.successful.is_(False), 1), else_=0)).label("failed"),
            func.sum(case((Execution.successful.is_(True), 1), else_=0)).label("successful"),
            func.sum(case((Execution.id.is_(None), 1), else_=0)).label("not_started"),
            func.sum(case((ExecutionGroup.dirty.is_(True), 1), else_=0)).label("dirty"),
        )
        .join(Diagnostic, ExecutionGroup.diagnostic_id == Diagnostic.id)
        .join(Provider, Diagnostic.provider_id == Provider.id)
        .outerjoin(latest_exec_subquery, ExecutionGroup.id == latest_exec_subquery.c.execution_group_id)
        .outerjoin(
            Execution,
            (Execution.execution_group_id == ExecutionGroup.id)
            & (Execution.created_at == latest_exec_subquery.c.latest_created_at),
        )
        .group_by(Provider.slug, Diagnostic.slug)
        .order_by(Provider.slug, Diagnostic.slug)
    )

    # Apply diagnostic filter
    if diagnostic:
        diagnostic_conditions = [
            Diagnostic.slug.ilike(f"%{filter_value.lower()}%") for filter_value in diagnostic
        ]
        query = query.filter(or_(*diagnostic_conditions))

    # Apply provider filter
    if provider:
        provider_conditions = [Provider.slug.ilike(f"%{filter_value.lower()}%") for filter_value in provider]
        query = query.filter(or_(*provider_conditions))

    results = query.all()

    if not results:
        console.print("No execution groups found.")
        return

    rows = [
        {
            "provider": row.provider,
            "diagnostic": row.diagnostic,
            "running": row.running,
            "failed": row.failed,
            "successful": row.successful,
            "not_started": row.not_started,
            "dirty": row.dirty,
            "total": row.total,
        }
        for row in results
    ]

    results_df = pd.DataFrame(rows)

    # Add provider totals
    status_cols = ["running", "failed", "successful", "not_started", "dirty", "total"]
    totals = results_df.groupby("provider")[status_cols].sum().reset_index()
    totals["diagnostic"] = "(total)"

    results_df = pd.concat([results_df, totals], ignore_index=True)
    results_df = results_df.sort_values(["provider", "diagnostic"]).reset_index(drop=True)

    pretty_print_df(results_df, console=console)


@app.command()
def reingest(  # noqa: PLR0913
    ctx: typer.Context,
    group_ids: Annotated[
        list[int] | None,
        typer.Argument(help="Execution group IDs to reingest. If omitted, uses filters."),
    ] = None,
    mode: Annotated[
        ReingestMode,
        typer.Option(
            help="Reingest mode: 'additive' (keep existing, add new), "
            "'replace' (delete existing, re-ingest), "
            "'versioned' (create new execution record)."
        ),
    ] = ReingestMode.additive,
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

    Three modes are available:

    - additive: Keep existing metric values, add any new ones. Outputs are replaced.

    - replace: Delete all existing metric values and outputs, then re-ingest from scratch.

    - versioned: Create a new Execution record under the same ExecutionGroup,
      leaving the original execution untouched.

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
        console.print(f"[bold]Dry run:[/] would reingest {len(results)} execution(s) in {mode.value} mode:")
        pretty_print_df(preview_df, console=console)
        return

    console.print(f"Will reingest {len(results)} execution(s) in [bold]{mode.value}[/] mode:")
    pretty_print_df(preview_df, console=console)

    if not force:
        if not typer.confirm("\nProceed with reingest?"):
            console.print("Reingest cancelled.")
            return

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
                mode=mode,
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
