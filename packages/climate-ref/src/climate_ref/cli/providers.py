"""
Manage the REF providers.
"""

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

    results_df = pd.DataFrame(
        [
            {
                "provider": provider.slug,
                "version": provider.version,
                "conda environment": get_env(provider),
            }
            for provider in provider_registry.providers
        ]
    )
    pretty_print_df(results_df, console=console)


@app.command()
def create_env(
    ctx: typer.Context,
    provider: Annotated[
        str | None,
        typer.Option(help="Only install the environment for the named provider."),
    ] = None,
) -> None:
    """
    Create a conda environment containing the provider software.

    If no provider is specified, all providers will be installed.
    If the provider is up to date or does not use a conda environment, it will be skipped.
    """
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
