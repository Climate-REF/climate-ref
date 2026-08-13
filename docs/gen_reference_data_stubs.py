"""
Generate the reference data page.

Lists every observational dataset the diagnostics require and which collection supplies it,
from the same function `ref doctor` uses, so the published table and the local check cannot
disagree.

This script follows the pattern of gen_diagnostic_stubs.py.
"""

from __future__ import annotations

import mkdocs_gen_files
from loguru import logger

import climate_ref  # noqa: F401  (registers the obs4REF and quickstart registries)
from climate_ref_core.providers import import_provider
from climate_ref_core.reference_data import (
    collect_required_reference_data,
    format_reference_data_markdown,
)

# Providers whose requirements make up the reference data list. The example provider is
# excluded: it is a tutorial, and its requirements are not data anyone needs to fetch.
PROVIDERS = [
    ("esmvaltool", "climate_ref_esmvaltool:provider"),
    ("pmp", "climate_ref_pmp:provider"),
    ("ilamb", "climate_ref_ilamb:provider"),
]


def _load_providers():
    """Load all available providers, skipping those with missing dependencies."""
    loaded = []
    for slug, fqn in PROVIDERS:
        try:
            loaded.append(import_provider(fqn))
        except Exception:
            logger.warning(f"Could not load provider '{slug}' ({fqn}), skipping.")
    return loaded


providers = _load_providers()
datasets = collect_required_reference_data(providers)

with mkdocs_gen_files.open("reference-data.md", "w") as fh:
    fh.write(format_reference_data_markdown(datasets))
