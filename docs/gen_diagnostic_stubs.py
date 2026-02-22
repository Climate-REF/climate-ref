"""
Generate diagnostic summary documentation pages.

Produces per-provider pages and a cross-provider overview of variables
by experiment and source type.

This script follows the pattern of gen_doc_stubs.py and gen_config_stubs.py.
"""

from __future__ import annotations

import mkdocs_gen_files
from loguru import logger

from climate_ref_core.providers import import_provider
from climate_ref_core.summary import (
    collect_by_source_type,
    format_overview_markdown,
    format_provider_markdown,
    summarize_provider,
)

# Provider entry points: (slug, fully-qualified name)
PROVIDERS = [
    ("example", "climate_ref_example:provider"),
    ("esmvaltool", "climate_ref_esmvaltool:provider"),
    ("pmp", "climate_ref_pmp:provider"),
    ("ilamb", "climate_ref_ilamb:provider"),
]


def _load_providers():
    """Load all available providers, skipping those with missing dependencies."""
    loaded = []
    for slug, fqn in PROVIDERS:
        try:
            provider = import_provider(fqn)
            loaded.append(provider)
        except Exception:
            logger.warning(f"Could not load provider '{slug}' ({fqn}), skipping.")
    return loaded


def write_provider_pages(providers):
    """Write a markdown page for each provider."""
    for provider in providers:
        summary = summarize_provider(provider)
        md = format_provider_markdown(summary)
        path = f"diagnostics/{provider.slug}.md"

        with mkdocs_gen_files.open(path, "w") as fh:
            fh.write(md)


def write_index_page(providers):
    """Write the diagnostics index page combining provider table and overview."""
    lines = [
        "# Diagnostics",
        "",
        "Summary of diagnostic providers and their available diagnostics.",
        "",
        "## Providers",
        "",
        "| Provider | Slug | Version | Diagnostics |",
        "| --- | --- | --- | --- |",
    ]

    for provider in providers:
        summary = summarize_provider(provider)
        link = f"[{summary.name}]({summary.slug}.md)"
        lines.append(f"| {link} | `{summary.slug}` | `{summary.version}` | {len(summary.diagnostics)} |")

    lines.append("")

    # Append the overview tables
    tables = collect_by_source_type(providers)
    overview_md = format_overview_markdown(tables)
    # Skip the "# Diagnostics Overview" heading since we're embedding in the index page
    overview_lines = overview_md.split("\n")
    # Find the first "##" heading and include from there
    for i, line in enumerate(overview_lines):
        if line.startswith("## "):
            lines.extend(overview_lines[i:])
            break

    with mkdocs_gen_files.open("diagnostics/index.md", "w") as fh:
        fh.write("\n".join(lines))


providers = _load_providers()
write_provider_pages(providers)
write_index_page(providers)
