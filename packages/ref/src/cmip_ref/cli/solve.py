import typer

from cmip_ref.solver import solve_metrics

app = typer.Typer()


@app.command()
def solve(
    ctx: typer.Context,
    dry_run: bool = typer.Option(False, help="Do not execute any metrics"),
    timeout: int = typer.Option(60, help="Timeout in seconds for the solve operation"),
) -> None:
    """
    Solve for metrics that require recalculation

    This may trigger a number of additional calculations depending on what data has been ingested
    since the last solve.
    """
    config = ctx.obj.config
    db = ctx.obj.database
    with ctx.obj.database.session.begin():
        solve_metrics(config=config, db=db, dry_run=dry_run, timeout=timeout)
