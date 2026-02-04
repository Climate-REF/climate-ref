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
    timeout: int = typer.Option(60, help="Timeout in seconds for the solve operation"),
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

    filters = SolveFilterOptions(
        diagnostic=diagnostic,
        provider=provider,
    )

    solve_required_executions(
        config=config,
        db=db,
        dry_run=dry_run,
        execute=execute,
        timeout=timeout,
        one_per_provider=one_per_provider,
        one_per_diagnostic=one_per_diagnostic,
        filters=filters,
    )
