"""
View and update the REF configuration
"""

from __future__ import annotations

import json
import os
import tomllib
from collections.abc import Iterator, MutableMapping
from contextlib import contextmanager
from difflib import get_close_matches
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any, cast

import attrs
import click
import tomlkit
import typer
from loguru import logger
from tomlkit import TOMLDocument

from climate_ref.cli._config_access import (
    ConfigKeyError,
    available_keys,
    coerce_value,
    default_value,
    env_var_for,
    is_structured,
    resolve_key,
)

app = typer.Typer(help=__doc__)


class ConfigFormat(StrEnum):
    """Output format for the ``config list`` command."""

    toml = "toml"
    json = "json"


class ValidationFormat(StrEnum):
    """Output format for the ``config validate`` command."""

    text = "text"
    json = "json"


def _config_file(ctx: typer.Context) -> Path:
    config_file = ctx.obj.config._config_file
    if config_file is None:  # pragma: no cover
        typer.secho("No configuration file location is configured.", fg=typer.colors.RED)
        raise typer.Exit(1)
    return cast(Path, config_file)


def _ensure_config_loaded(ctx: typer.Context) -> None:
    config_file = _config_file(ctx)
    if ctx.obj.configuration_directory is not None and not config_file.is_file():
        typer.secho("Configuration file not found", fg=typer.colors.RED)
        raise typer.Exit(1)
    if ctx.obj.config_load_error is not None:
        typer.secho(
            f"Error loading configuration from {config_file}: {ctx.obj.config_load_error}",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)


def _print_key_error(config: Any, key: str) -> None:
    typer.secho(f"Unknown configuration key: {key}", fg=typer.colors.RED, err=True)
    matches = get_close_matches(key, available_keys(config), n=3)
    if matches:
        typer.secho(f"Did you mean: {', '.join(matches)}?", err=True)


def _ensure_scalar(key: str, value: Any, field: Any) -> None:
    if is_structured(value, field):
        typer.secho(
            f"Editing structured configuration values from the CLI is not supported: {key}. "
            "Edit ref.toml directly.",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)


def _warn_if_env_shadowed(parent: object, field: Any, key: str) -> None:
    env_var = env_var_for(parent, field)
    if env_var and os.environ.get(env_var) is not None:
        typer.secho(
            f"Warning: {key} is overridden by environment variable {env_var}; "
            "the file change may not take effect until it is unset.",
            fg=typer.colors.YELLOW,
            err=True,
        )


def _stringify_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    return str(value)


def _toml_for_key(config: Any, dotted: str) -> str:
    node: Any = config.dump(defaults=True)
    for part in dotted.split("."):
        node = node[part]

    doc = TOMLDocument()
    doc[dotted.rsplit(".", maxsplit=1)[-1]] = node
    return doc.as_string()


def _validation_payload(errors: list[str]) -> dict[str, Any]:
    return {
        "valid": not errors,
        "error_count": len(errors),
        "diagnostics": [{"severity": "error", "message": error} for error in errors],
    }


def _config_env_vars(config: object) -> set[str]:
    if not attrs.has(config.__class__):
        return set()

    env_vars: set[str] = set()
    prefix = getattr(config, "_prefix", None)
    for field in attrs.fields(config.__class__):
        env_name = field.metadata.get("env")
        if prefix and env_name:
            env_vars.add(f"{prefix}_{env_name}")

        if not field.name.startswith("_"):
            value = getattr(config, field.name, None)
            env_vars.update(_config_env_vars(value))

    return env_vars


@contextmanager
def _without_config_env_overrides(config: object) -> Iterator[None]:
    saved = {name: os.environ.get(name) for name in _config_env_vars(config)}
    try:
        for name in saved:
            os.environ.pop(name, None)
        yield
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def _load_config_document(config_file: Path) -> TOMLDocument:
    if not config_file.is_file():
        return TOMLDocument()
    with config_file.open() as fh:
        return tomlkit.load(fh)


def _toml_scalar(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return value


def _set_config_document_key(doc: TOMLDocument, dotted: str, value: Any) -> None:
    node: MutableMapping[str, Any] = doc
    parts = dotted.split(".")
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, MutableMapping):
            child = tomlkit.table()
            node[part] = child
        node = child
    node[parts[-1]] = _toml_scalar(value)


def _unset_config_document_key(doc: TOMLDocument, dotted: str) -> None:
    node: MutableMapping[str, Any] = doc
    parts = dotted.split(".")
    for part in parts[:-1]:
        child = node.get(part)
        if not isinstance(child, MutableMapping):
            return
        node = child
    node.pop(parts[-1], None)


def _write_config_document(config_file: Path, doc: TOMLDocument) -> None:
    config_file.parent.mkdir(parents=True, exist_ok=True)
    config_file.write_text(doc.as_string())


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
    _ensure_config_loaded(ctx)
    config = ctx.obj.config

    rendered = config.dumps(defaults=True)
    if output_format == ConfigFormat.json:
        print(json.dumps(tomllib.loads(rendered), indent=2))
    else:
        print(rendered)


@app.command(name="init")
def init(
    ctx: typer.Context,
    force: Annotated[
        bool,
        typer.Option("--force", help="Overwrite an existing ref.toml with a fresh template."),
    ] = False,
    no_defaults: Annotated[
        bool,
        typer.Option("--no-defaults", help="Write a minimal template that omits default values."),
    ] = False,
) -> None:
    """
    Create a ref.toml configuration file for onboarding.

    By default this writes a complete template with defaults included. ``--force`` is destructive:
    it replaces any existing configuration file with a fresh template.
    """
    config = ctx.obj.config
    config_file = _config_file(ctx)

    if config_file.exists() and not force:
        typer.secho(
            f"Configuration already exists at {config_file}. Use --force to overwrite.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)

    with _without_config_env_overrides(config):
        template_config = config.__class__()
    template_config._config_file = config_file

    config_file.parent.mkdir(parents=True, exist_ok=True)
    if no_defaults:
        config_file.write_text(template_config.dumps(defaults=False))
    else:
        template_config.save(config_file)

    action = "Overwrote" if force else "Wrote"
    ctx.obj.console.print(f"{action} configuration file at {config_file}")
    ctx.obj.console.print("\nNext steps:")
    ctx.obj.console.print("  1. Set REF_CONFIGURATION to this directory if needed.")
    ctx.obj.console.print("  2. Edit ref.toml for your paths, database, executor and providers.")
    ctx.obj.console.print("  3. Run `ref config validate`.")
    ctx.obj.console.print("  4. Run `ref providers list`, then `ref solve` when data is available.")


@app.command(name="get")
def get(ctx: typer.Context, key: Annotated[str, typer.Argument(help="Dotted configuration key")]) -> None:
    """
    Print one effective configuration value.

    Environment variables are already applied, so this shows the value the REF will use at runtime.
    Scalar values are printed alone on stdout for scripts; structured sections are printed as TOML.
    """
    _ensure_config_loaded(ctx)
    config = ctx.obj.config
    try:
        parent, field, value = resolve_key(config, key)
    except ConfigKeyError:
        _print_key_error(config, key)
        raise typer.Exit(1)

    env_var = env_var_for(parent, field)
    if env_var and os.environ.get(env_var) is not None:
        logger.info(f"{key} is overridden by environment variable {env_var}; showing the effective value.")

    if is_structured(value, field):
        print(_toml_for_key(config, key))
    else:
        print(_stringify_scalar(value))


@app.command(name="set")
def set_(
    ctx: typer.Context,
    key: Annotated[str, typer.Argument(help="Dotted scalar configuration key")],
    value: Annotated[str, typer.Argument(help="New value")],
) -> None:
    """
    Set one scalar configuration value and persist it to ref.toml.

    The value is written to the file-backed configuration, even when environment variables shadow
    the effective runtime value.
    """
    _ensure_config_loaded(ctx)
    config = ctx.obj.config
    try:
        parent, field, current_value = resolve_key(config, key)
    except ConfigKeyError:
        _print_key_error(config, key)
        raise typer.Exit(1)

    _ensure_scalar(key, current_value, field)
    try:
        coerced = coerce_value(field, value)
    except Exception as exc:
        typer.secho(f"Invalid value for {key}: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    setattr(parent, field.name, coerced)
    config_file = _config_file(ctx)
    doc = _load_config_document(config_file)
    _set_config_document_key(doc, key, coerced)
    _write_config_document(config_file, doc)
    _warn_if_env_shadowed(parent, field, key)
    typer.echo(f"{key} = {_stringify_scalar(coerced)}")
    typer.echo(f"Saved to {config._config_file}")


@app.command(name="unset")
def unset(
    ctx: typer.Context,
    key: Annotated[str, typer.Argument(help="Dotted scalar configuration key")],
) -> None:
    """
    Reset one scalar configuration value to its default and persist it to ref.toml.

    Structured values such as provider lists and executor config dictionaries must be edited directly.
    """
    _ensure_config_loaded(ctx)
    config = ctx.obj.config
    try:
        parent, field, current_value = resolve_key(config, key)
    except ConfigKeyError:
        _print_key_error(config, key)
        raise typer.Exit(1)

    _ensure_scalar(key, current_value, field)
    try:
        reset_value = default_value(parent, field)
    except ValueError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(1)

    setattr(parent, field.name, reset_value)
    config_file = _config_file(ctx)
    doc = _load_config_document(config_file)
    _unset_config_document_key(doc, key)
    _write_config_document(config_file, doc)
    _warn_if_env_shadowed(parent, field, key)
    typer.echo(f"{key} reset to default {_stringify_scalar(reset_value)}")
    typer.echo(f"Saved to {config._config_file}")


@app.command(name="edit")
def edit(ctx: typer.Context) -> None:
    """
    Open ref.toml in the user's editor and warn if the result is invalid.

    The editor is selected by Click from ``$VISUAL`` / ``$EDITOR``. If no config file exists yet,
    run ``ref config init`` first.
    """
    config_file = _config_file(ctx)
    if not config_file.is_file():
        typer.secho(
            f"No configuration file found at {config_file}. Run `ref config init` first.",
            fg=typer.colors.RED,
        )
        raise typer.Exit(1)

    click.edit(filename=str(config_file))
    errors = ctx.obj.config.collect_validation_errors(config_file)
    if errors:
        typer.secho(
            f"Configuration at {config_file} is invalid after editing:",
            fg=typer.colors.YELLOW,
            err=True,
        )
        for error in errors:
            typer.secho(f"  - {error}", fg=typer.colors.YELLOW, err=True)
    else:
        typer.echo(f"Edited {config_file}")


@app.command(name="validate")
def validate(
    ctx: typer.Context,
    output_format: Annotated[
        ValidationFormat,
        typer.Option("--format", help="Output format: 'text' (default) or machine-readable 'json'."),
    ] = ValidationFormat.text,
) -> None:
    """
    Validate that ref.toml parses and matches the REF configuration schema.

    The exit code is the CI contract: 0 means valid and 1 means invalid. JSON output includes a
    Terraform-style ``valid`` flag, error count and diagnostics list.
    """
    config_file = _config_file(ctx)
    if not config_file.is_file():
        if output_format == ValidationFormat.json:
            payload = _validation_payload([f"No configuration file found at {config_file}."])
            print(json.dumps(payload, indent=2))
        else:
            typer.secho(f"No configuration file found at {config_file}.", fg=typer.colors.RED)
        raise typer.Exit(1)

    errors = ctx.obj.config.collect_validation_errors(config_file)
    if output_format == ValidationFormat.json:
        print(json.dumps(_validation_payload(errors), indent=2))
    elif errors:
        typer.secho(f"Configuration at {config_file} is invalid:", fg=typer.colors.RED)
        for error in errors:
            typer.echo(f"  - {error}")
    else:
        typer.secho(f"Configuration at {config_file} is valid.", fg=typer.colors.GREEN)

    if errors:
        raise typer.Exit(1)
