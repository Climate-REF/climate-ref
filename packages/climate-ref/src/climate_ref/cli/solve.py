from typing import TYPE_CHECKING, Annotated

import typer
from loguru import logger

from climate_ref.cli._utils import parse_facet_filters

if TYPE_CHECKING:
    from climate_ref.config import Config

app = typer.Typer()

_PROVIDER_REMEDIATION = (
    "Configure a diagnostic provider before solving:\n"
    "  - install one, e.g. `pip install 'climate-ref[aft-providers]'` or `pip install climate-ref-example`\n"
    "  - then regenerate your configuration, or add it under [[diagnostic_providers]] in ref.toml"
)


def _validate_provider_filter(config: "Config", provider: list[str] | None) -> None:
    """
    Fail loudly when no configured provider can satisfy the solve.

    Without this, ``ref solve`` (or ``ref solve --provider <typo>``) exits 0 having
    done nothing, which looks like success. Provider matching mirrors the solver:
    a filter matches a provider when it is a case-insensitive substring of the slug.
    """
    from climate_ref_core.providers import import_provider

    configured = [import_provider(p.provider).slug for p in config.diagnostic_providers]

    if not configured:
        logger.error("No diagnostic providers are configured.\n" + _PROVIDER_REMEDIATION)
        raise typer.Exit(code=1)

    if provider:
        matched = [slug for slug in configured if any(f.lower() in slug for f in provider)]
        if not matched:
            available = ", ".join(configured)
            logger.error(
                f"No configured providers match the --provider filter {provider}. "
                f"Available providers: {available}.\n" + _PROVIDER_REMEDIATION
            )
            raise typer.Exit(code=1)


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
    timeout: int = typer.Option(
        6 * 60 * 60,
        help="Timeout in seconds for waiting on executions to complete. Defaults to 6 hours. "
        "Set to 0 (or a negative value) to wait with no time limit. Ignored when --no-wait is used.",
    ),
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

    _validate_provider_filter(config, provider)

    try:
        parsed_dataset_filters = parse_facet_filters(dataset_filter) or None
    except ValueError as e:
        raise typer.BadParameter(str(e), param_hint="--dataset-filter")

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
        wait=wait,
        timeout=timeout,
        one_per_provider=one_per_provider,
        one_per_diagnostic=one_per_diagnostic,
        filters=filters,
        limit=limit,
        rerun_failed=rerun_failed,
    )
