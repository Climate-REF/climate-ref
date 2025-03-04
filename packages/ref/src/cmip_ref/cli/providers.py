"""
Manage the REF providers.
"""

import typer
from loguru import logger

from cmip_ref.provider_registry import ProviderRegistry
from cmip_ref_core.providers import CondaMetricsProvider

app = typer.Typer(help=__doc__)


@app.command(name="list")
def list_(ctx: typer.Context) -> None:
    """
    Print the available providers.
    """
    config = ctx.obj.config
    db = ctx.obj.database
    with db.session.begin():
        provider_registry = ProviderRegistry.build_from_config(config, db)
    providers = "\n".join([f"{p.name}: {p.version}" for p in provider_registry.providers])
    print(providers)


@app.command()
def create_env(ctx: typer.Context, provider: str | None = None) -> None:
    """
    Create a virtual environment containing the provider software.
    """
    config = ctx.obj.config
    db = ctx.obj.database
    with db.session.begin():
        providers = ProviderRegistry.build_from_config(config, db).providers

    if provider is not None:
        available = ", ".join([f'"{p.name}"' for p in providers])
        providers = [p for p in providers if p.name == provider]
        if not providers:
            msg = f'Provider "{provider}" not available. Choose from: {available}'
            logger.error(msg)
            raise typer.Exit(code=1)

    for provider_ in providers:
        if isinstance(provider_, CondaMetricsProvider):
            msg = f"Creating virtual environment for provider {provider_.name} " f"in {provider_.env_path}"
            logger.info(msg)
            provider_.create_env()
            logger.info(f"Finished creating virtual environment for {provider_.name}")
        else:
            logger.info(f"Skipping {provider_} because it does use virtual environments.")
