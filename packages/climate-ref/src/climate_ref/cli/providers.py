"""
Manage the REF providers.
"""

import warnings
from typing import Annotated

import pandas as pd
import typer
from loguru import logger

from climate_ref.cli._utils import pretty_print_df
from climate_ref.provider_registry import ProviderRegistry
from climate_ref_core.providers import CondaDiagnosticProvider, DiagnosticProvider

app = typer.Typer(help=__doc__)


@app.command(name="list")
def list_(ctx: typer.Context) -> None:
    """
    Print the available providers.
    """
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
def setup(
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
        typer.Option(help="Skip data fetching."),
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

    All operations are idempotent and safe to run multiple times.
    Run this on a login node with internet access before solving on compute nodes.
    """
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
            provider_.setup(config, skip_env=skip_env, skip_data=skip_data)
            is_valid = provider_.validate_setup(config)
            if not is_valid:
                logger.error(f"Provider {provider_.slug} setup completed but validation failed")
                failed_providers.append(provider_.slug)
            else:
                logger.info(f"Finished setting up provider {provider_.slug}")
        except Exception as e:
            logger.opt(exception=True).error(f"Failed to setup provider {provider_.slug}: {e}")
            failed_providers.append(provider_.slug)

    if failed_providers:
        msg = f"Setup failed for providers: {', '.join(failed_providers)}"
        logger.error(msg)
        raise typer.Exit(code=1)

    if not validate_only:
        list_(ctx)
