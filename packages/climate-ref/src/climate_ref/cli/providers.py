"""
Manage the REF providers.
"""

import warnings
from enum import Enum
from typing import TYPE_CHECKING, Annotated

import typer
from loguru import logger

from climate_ref.cli._utils import pretty_print_df

if TYPE_CHECKING:
    from rich.console import Console

    from climate_ref_core.summary import ProviderSummary

app = typer.Typer(help=__doc__)


@app.command(name="list")
def list_(ctx: typer.Context) -> None:
    """
    Print the available providers.
    """
    import pandas as pd

    from climate_ref.provider_registry import ProviderRegistry
    from climate_ref_core.providers import CondaDiagnosticProvider, DiagnosticProvider

    config = ctx.obj.config
    db = ctx.obj.database
    console = ctx.obj.console
    provider_registry = ProviderRegistry.build_from_config(config, db)

    def get_env(provider: DiagnosticProvider) -> str:
        env = ""
        if isinstance(provider, CondaDiagnosticProvider):
            env = f"{provider.env_path}"
            if not provider.env_path.exists():
                env += " (not installed)"
        return env

    def get_data_path(provider: DiagnosticProvider) -> str:
        """Get the data cache path for a provider."""
        data_path = provider.get_data_path()
        if data_path is None:
            return ""
        path_str = str(data_path)
        if not data_path.exists():
            path_str += " (not fetched)"
        return path_str

    results_df = pd.DataFrame(
        [
            {
                "provider": provider.slug,
                "version": provider.version,
                "conda environment": get_env(provider),
                "data path": get_data_path(provider),
            }
            for provider in provider_registry.providers
        ]
    )
    pretty_print_df(results_df, console=console)


class ShowFormat(str, Enum):
    """Output format for the show command."""

    table = "table"
    list = "list"


@app.command()
def show(
    ctx: typer.Context,
    provider: Annotated[
        str,
        typer.Option(help="Slug of the provider to show diagnostics for."),
    ],
    output_format: Annotated[
        ShowFormat,
        typer.Option(
            "--format",
            help="Output format: 'list' for detailed per-diagnostic output, 'table' for a compact table.",
        ),
    ] = ShowFormat.list,
    columns: Annotated[
        list[str] | None,
        typer.Option(
            "--columns",
            help="Columns to include in table output (e.g. --columns diagnostic --columns variables).",
        ),
    ] = None,
) -> None:
    """
    Show diagnostics and data requirements for a provider.
    """
    from climate_ref.provider_registry import ProviderRegistry
    from climate_ref_core.summary import summarize_provider

    config = ctx.obj.config
    db = ctx.obj.database
    console = ctx.obj.console
    provider_registry = ProviderRegistry.build_from_config(config, db)

    try:
        prov = provider_registry.get(provider)
    except KeyError:
        available = ", ".join([f'"{p.slug}"' for p in provider_registry.providers])
        logger.error(f'Provider "{provider}" not available. Choose from: {available}')
        raise typer.Exit(code=1)

    summary = summarize_provider(prov)

    if not summary.diagnostics:
        console.print(f"Provider '{provider}' has no registered diagnostics.")
        return

    if output_format == ShowFormat.list:
        _show_list(summary, console)
    else:
        _show_table(summary, console, columns=columns)


def _show_table(
    summary: "ProviderSummary",
    console: "Console",
    columns: list[str] | None = None,
) -> None:
    """Display provider summary as a compact table."""
    import pandas as pd

    rows = []
    for diag in summary.diagnostics:
        for set_idx, req_set in enumerate(diag.requirement_sets):
            option_label = f"Option {set_idx + 1}" if len(diag.requirement_sets) > 1 else ""
            for req in req_set.requirements:
                rows.append(
                    {
                        "diagnostic": diag.name,
                        "slug": diag.slug,
                        "option": option_label,
                        "source_type": req.source_type,
                        "variables": ", ".join(req.variables) if req.variables else "*",
                        "experiments": ", ".join(req.experiments) if req.experiments else "*",
                        "tables": ", ".join(req.tables) if req.tables else "*",
                    }
                )

    results_df = pd.DataFrame(rows)

    if columns:
        available = list(results_df.columns)
        invalid = [c for c in columns if c not in available]
        if invalid:
            console.print(f"[red]Unknown columns: {', '.join(invalid)}[/red]")
            console.print(f"Available columns: {', '.join(available)}")
            raise typer.Exit(code=1)
        results_df = results_df[columns]

    pretty_print_df(results_df, console=console)


def _show_list(summary: "ProviderSummary", console: "Console") -> None:
    """Display provider summary as a detailed per-diagnostic list."""
    from rich.panel import Panel
    from rich.text import Text

    console.print()
    console.print(f"[bold]{summary.name}[/bold] (v{summary.version})")
    console.print()

    for diag in summary.diagnostics:
        lines = Text()
        lines.append(f"Slug: {diag.slug}\n")
        lines.append(f"Facets: {', '.join(diag.facets)}\n")

        for set_idx, req_set in enumerate(diag.requirement_sets):
            if len(diag.requirement_sets) > 1:
                lines.append(f"\nOption {set_idx + 1}:\n", style="bold")

            for req in req_set.requirements:
                lines.append(f"  Source type: {req.source_type}\n")
                lines.append(f"  Variables:   {', '.join(req.variables) if req.variables else '*'}\n")
                lines.append(f"  Experiments: {', '.join(req.experiments) if req.experiments else '*'}\n")
                if req.tables:
                    lines.append(f"  Tables:      {', '.join(req.tables)}\n")
                if req.frequencies:
                    lines.append(f"  Frequencies: {', '.join(req.frequencies)}\n")
                if req.group_by:
                    lines.append(f"  Group by:    {', '.join(req.group_by)}\n")

        console.print(Panel(lines, title=diag.name, expand=False))


@app.command(deprecated=True)
def create_env(
    ctx: typer.Context,
    provider: Annotated[
        str | None,
        typer.Option(help="Only install the environment for the named provider."),
    ] = None,
) -> None:
    """
    Create a conda environment containing the provider software.

    .. deprecated::
        Use `ref providers setup` instead, which handles both environment creation
        and data fetching in a single command.

    If no provider is specified, all providers will be installed.
    If the provider is up to date or does not use a conda environment, it will be skipped.
    """
    warnings.warn(
        "create-env is deprecated. Use 'ref providers setup' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    from climate_ref.provider_registry import ProviderRegistry
    from climate_ref_core.providers import CondaDiagnosticProvider

    config = ctx.obj.config
    db = ctx.obj.database
    providers = ProviderRegistry.build_from_config(config, db).providers

    if provider is not None:
        available = ", ".join([f'"{p.slug}"' for p in providers])
        providers = [p for p in providers if p.slug == provider]
        if not providers:
            msg = f'Provider "{provider}" not available. Choose from: {available}'
            logger.error(msg)
            raise typer.Exit(code=1)

    for provider_ in providers:
        txt = f"conda environment for provider {provider_.slug}"
        if isinstance(provider_, CondaDiagnosticProvider):
            logger.info(f"Creating {txt} in {provider_.env_path}")
            provider_.create_env()
            logger.info(f"Finished creating {txt}")
        else:
            logger.info(f"Skipping creating {txt} because it does not use conda environments.")

    list_(ctx)


@app.command()
def setup(  # noqa: PLR0913
    ctx: typer.Context,
    provider: Annotated[
        str | None,
        typer.Option(help="Only run setup for the named provider."),
    ] = None,
    skip_env: Annotated[
        bool,
        typer.Option(help="Skip environment setup (e.g., conda)."),
    ] = False,
    skip_data: Annotated[
        bool,
        typer.Option(help="Skip data fetching and ingestion."),
    ] = False,
    skip_validate: Annotated[
        bool,
        typer.Option(help="Skip validation."),
    ] = False,
    validate_only: Annotated[
        bool,
        typer.Option(help="Only validate setup, don't run it."),
    ] = False,
) -> None:
    """
    Run provider setup for offline execution.

    This command prepares all providers for offline execution by:

    1. Creating conda environments (if applicable)

    2. Fetching required reference datasets to pooch cache

    3. Ingesting provider-specific datasets into the database

    All operations are idempotent and safe to run multiple times.
    Run this on a login node with internet access before solving on compute nodes.
    """
    from climate_ref.provider_registry import ProviderRegistry

    config = ctx.obj.config
    db = ctx.obj.database
    console = ctx.obj.console
    providers = ProviderRegistry.build_from_config(config, db).providers

    if provider is not None:
        available = ", ".join([f'"{p.slug}"' for p in providers])
        providers = [p for p in providers if p.slug == provider]
        if not providers:
            msg = f'Provider "{provider}" not available. Choose from: {available}'
            logger.error(msg)
            raise typer.Exit(code=1)

    failed_providers: list[str] = []

    for provider_ in providers:
        if validate_only:
            is_valid = provider_.validate_setup(config)
            status = "[green]valid[/green]" if is_valid else "[red]invalid[/red]"
            console.print(f"Provider {provider_.slug}: {status}")
            if not is_valid:
                failed_providers.append(provider_.slug)
            continue

        logger.info(f"Setting up provider {provider_.slug}")
        try:
            provider_.setup(config, db=db, skip_env=skip_env, skip_data=skip_data)
            if not skip_validate:
                is_valid = provider_.validate_setup(config)
                if not is_valid:
                    logger.error(f"Provider {provider_.slug} setup completed but validation failed")
                    failed_providers.append(provider_.slug)
                else:
                    logger.info(f"Finished setting up provider {provider_.slug}")
            else:
                logger.info(f"Skipped validation for provider {provider_.slug}")
        except Exception as e:
            logger.opt(exception=True).error(f"Failed to setup provider {provider_.slug}: {e}")
            failed_providers.append(provider_.slug)

    if failed_providers:
        msg = f"Setup failed for providers: {', '.join(failed_providers)}"
        logger.error(msg)
        raise typer.Exit(code=1)

    if not validate_only:
        list_(ctx)
