"""
View and update the REF configuration
"""

import json
import tomllib
from enum import StrEnum
from typing import Annotated

import typer

app = typer.Typer(help=__doc__)


class ConfigFormat(StrEnum):
    """Output format for the ``config list`` command."""

    toml = "toml"
    json = "json"


@app.command(name="list")
def list_(
    ctx: typer.Context,
    output_format: Annotated[
        ConfigFormat,
        typer.Option("--format", help="Output format: 'toml' (default) or machine-readable 'json'."),
    ] = ConfigFormat.toml,
) -> None:
    """
    Print the current climate_ref configuration

    If a configuration directory is provided,
    the configuration will attempt to load from the specified directory.
    """
    config = ctx.obj.config

    rendered = config.dumps(defaults=True)
    if output_format == ConfigFormat.json:
        print(json.dumps(tomllib.loads(rendered), indent=2))
    else:
        print(rendered)


# @app.command()
# def update() -> None:
#     """
#     Update a configuration value
#     """
#     print("config")
