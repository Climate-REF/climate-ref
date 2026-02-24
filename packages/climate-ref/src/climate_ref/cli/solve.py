from typing import Annotated

import typer

app = typer.Typer()


@app.command()
def solve(  # noqa: PLR0913
    ctx: typer.Context,
    dry_run: Annotated[
        bool,
        typer.Option(help="Do not execute any diagnostics"),
    ] = False,
    execute: Annotated[
        bool,
        typer.Option(help="Solve the newly identified executions"),
    ] = True,
    timeout: int = typer.Option(60, help="Timeout in seconds for waiting on executions to complete"),
    wait: Annotated[
        bool,
        typer.Option(
            help="Wait for executions to complete before exiting. "
            "Use --no-wait to queue executions and exit immediately."
        ),
    ] = True,
    one_per_provider: bool = typer.Option(
        False, help="Limit to one execution per provider. This is useful for testing"
    ),
    one_per_diagnostic: bool = typer.Option(
        False, help="Limit to one execution per diagnostic. This is useful for testing"
    ),
    diagnostic: Annotated[
        list[str] | None,
        typer.Option(
            help="Filters executions by the diagnostic slug. "
            "Diagnostics will be included if any of the filters match a case-insensitive subset "
            "of the diagnostic slug. "
            "Multiple values can be provided"
        ),
    ] = None,
    provider: Annotated[
        list[str] | None,
        typer.Option(
            help="Filters executions by provider slug. "
            "Providers will be included if any of the filters match a case-insensitive subset "
            "of the provider slug. "
            "Multiple values can be provided"
        ),
    ] = None,
    dataset_filter: Annotated[
        list[str] | None,
        typer.Option(
            help="Filter input datasets by facet values using key=value syntax. "
            "For example, --dataset-filter source_id=ACCESS-CM2 --dataset-filter variable_id=tas. "
            "Multiple values for the same facet are ORed (include any match), "
            "different facets are ANDed (must match all). "
            "Multiple values can be provided"
        ),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option(help="Maximum number of executions to run. If not set, all executions are run."),
    ] = None,
    rerun_failed: Annotated[
        bool,
        typer.Option(
            help="Re-run all previously failed executions, even if the execution group is not dirty. "
            "By default, failed executions are only retried if explicitly flagged dirty."
        ),
    ] = False,
) -> None:
    """
    Solve for executions that require recalculation

    This may trigger a number of additional calculations depending on what data has been ingested
    since the last solve.
    This command will block until all executions have been solved or the timeout is reached.

    Filters can be applied to limit the diagnostics and providers that are considered, see the options
    `--diagnostic` and `--provider` for more information.
    """
    from climate_ref.solver import SolveFilterOptions, solve_required_executions

    config = ctx.obj.config
    db = ctx.obj.database

    parsed_dataset_filters: dict[str, list[str]] | None = None
    if dataset_filter:
        parsed_dataset_filters = {}
        for entry in dataset_filter:
            if "=" not in entry:
                raise typer.BadParameter(
                    f"Invalid dataset filter {entry!r}. Expected key=value format.",
                    param_hint="--dataset-filter",
                )
            key, value = entry.split("=", 1)
            parsed_dataset_filters.setdefault(key, []).append(value)

    filters = SolveFilterOptions(
        diagnostic=diagnostic,
        provider=provider,
        dataset=parsed_dataset_filters,
    )

    solve_required_executions(
        config=config,
        db=db,
        dry_run=dry_run,
        execute=execute,
        timeout=timeout if wait else 0,
        one_per_provider=one_per_provider,
        one_per_diagnostic=one_per_diagnostic,
        filters=filters,
        limit=limit,
        rerun_failed=rerun_failed,
    )
