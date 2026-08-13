"""
A description of the deployment, to accompany the findings.

A finding on its own rarely says enough to act on: the same warning means different things
on a laptop and on an HPC deployment with a shared software prefix. This collects what a
maintainer would otherwise have to ask for, in a form that can be pasted into an issue.

Values that could carry a credential are redacted here rather than at the point of display,
so no output format can leak one.
"""

import importlib.metadata
import os
import platform
import sys

from loguru import logger

from climate_ref.config import env_prefix
from climate_ref.database import REDACTED, redact_url
from climate_ref.doctor.context import DoctorContext
from climate_ref.doctor.registry import iter_checks
from climate_ref_core.source_types import SourceDatasetType

_SECRET_MARKERS = ("SECRET", "TOKEN", "PASSWORD", "PASSWD", "ACCESS_KEY", "API_KEY")
"""Substrings that make an environment variable's value too risky to print."""

_URL_MARKERS = ("URL", "URI", "DSN", "CONNECTION_STRING")
"""Substrings that mark a value as a URL, whose own password must be hidden."""

_ENV_PREFIXES = (f"{env_prefix}_", "DASK_", "ESMVALTOOL_")
"""Environment variable prefixes that change how a deployment behaves."""

_UNAVAILABLE = "unavailable"
"""Stands in for a section that could not be collected."""


def _redact_env_value(name: str, value: str) -> str:
    """Hide a credential a variable holds, whether as the whole value or inside a URL."""
    upper = name.upper()
    if any(marker in upper for marker in _SECRET_MARKERS):
        return REDACTED
    if any(marker in upper for marker in _URL_MARKERS):
        return redact_url(value)
    return value


def _versions() -> dict[str, str]:
    """Report the installed REF packages, plus the interpreter running them."""
    versions = {
        distribution.metadata["Name"]: distribution.version
        for distribution in importlib.metadata.distributions()
        if (distribution.metadata["Name"] or "").startswith("climate-ref")
    }
    return {**dict(sorted(versions.items())), "python": platform.python_version()}


def _platform() -> dict[str, str]:
    """Report the machine the deployment runs on."""
    return {
        "system": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "cpu_count": str(os.cpu_count()),
        "executable": sys.executable,
    }


def _environment_variables() -> dict[str, str]:
    """Report the environment variables that change how a deployment behaves, secrets hidden."""
    return {
        name: _redact_env_value(name, value)
        for name, value in sorted(os.environ.items())
        if name.startswith(_ENV_PREFIXES)
    }


def _configuration(context: DoctorContext) -> dict[str, str]:
    """Report the configuration values that most often explain a finding."""
    config = context.config
    if config is None:
        return {"config_file": _UNAVAILABLE}
    return {
        "config_file": str(config._config_file),
        "log_level": config.log_level,
        "n_jobs": str(config.n_jobs),
        "cmip6_parser": config.cmip6_parser,
        "cmip7_parser": config.cmip7_parser,
        "executor": config.executor.executor,
        "database_url": redact_url(config.db.database_url),
        "ignore_datasets_file": str(config.ignore_datasets_file),
    }


def _paths(context: DoctorContext) -> dict[str, str]:
    """Report the configured paths, and whether they are actually there."""
    config = context.config
    if config is None:
        return {}
    paths = {
        "log": config.paths.log,
        "scratch": config.paths.scratch,
        "software": config.paths.software,
        "results": config.paths.results,
    }
    return {name: f"{path} ({'exists' if path.exists() else 'missing'})" for name, path in paths.items()}


def _providers(context: DoctorContext) -> dict[str, str]:
    """Report the enabled providers and their versions, or why they could not be listed."""
    try:
        providers = context.providers
    except Exception as exc:
        logger.debug(f"Could not load providers for the environment report: {exc}")
        return {"error": f"{type(exc).__name__}: {exc}"}
    return {provider.slug: provider.version for provider in providers}


def _ingested(context: DoctorContext) -> dict[str, str]:
    """Report how many datasets and files are ingested, per source type."""
    counts = {}
    for source_type in SourceDatasetType:
        try:
            catalog = context.catalog(source_type)
        except Exception as exc:
            logger.debug(f"Could not load the {source_type.value} catalog: {exc}")
            counts[source_type.value] = _UNAVAILABLE
            continue
        if not len(catalog) or "instance_id" not in catalog:
            continue
        counts[source_type.value] = f"{catalog['instance_id'].nunique()} datasets, {len(catalog)} files"
    return counts


def _checks() -> dict[str, str]:
    """Report every check that ran, and whether it is built in or came from a plugin."""
    return {registered.slug: registered.source for registered in iter_checks()}


def collect_environment(context: DoctorContext) -> dict[str, dict[str, str]]:
    """
    Describe the deployment being checked.

    A section that cannot be collected is reported as such rather than raising.
    An environment report is only useful if it survives the deployment being broken.

    Parameters
    ----------
    context
        The deployment to describe.

    Returns
    -------
    :
        One entry per area, each holding that area's ``name: value`` pairs.
    """
    sections = {
        "versions": _versions,
        "platform": _platform,
        "configuration": lambda: _configuration(context),
        "paths": lambda: _paths(context),
        "providers": lambda: _providers(context),
        "ingested": lambda: _ingested(context),
        "environment_variables": _environment_variables,
        "checks": _checks,
    }

    collected = {}
    for name, collect in sections.items():
        try:
            collected[name] = collect()
        except Exception as exc:
            logger.exception(f"Could not collect the '{name}' section of the environment report")
            collected[name] = {"error": f"{type(exc).__name__}: {exc}"}

    return collected
