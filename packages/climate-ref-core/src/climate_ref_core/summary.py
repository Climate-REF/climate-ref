"""
Utilities for extracting and formatting diagnostic/provider summaries.

This module provides structured summary data classes and functions for
introspecting diagnostic providers, their diagnostics, and data requirements.
Both CLI and documentation generation can reuse these utilities.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING

from attrs import frozen

from climate_ref_core.datasets import FacetFilter
from climate_ref_core.diagnostics import DataRequirement

if TYPE_CHECKING:
    from climate_ref_core.diagnostics import Diagnostic
    from climate_ref_core.providers import DiagnosticProvider


@frozen
class DataRequirementSummary:
    """
    Summary of a single DataRequirement's filters and grouping.
    """

    source_type: str
    variables: tuple[str, ...]
    experiments: tuple[str, ...]
    tables: tuple[str, ...]
    frequencies: tuple[str, ...]
    group_by: tuple[str, ...] | None


@frozen
class RequirementSetSummary:
    """
    Summary of one OR-alternative set of DataRequirements.

    A diagnostic may have multiple requirement sets (OR-logic).
    Each set contains one or more DataRequirementSummary entries
    that must all be satisfied together (AND-logic within a set).
    """

    requirements: tuple[DataRequirementSummary, ...]


@frozen
class DiagnosticSummary:
    """
    Summary of a single diagnostic and its data requirements.
    """

    name: str
    slug: str
    provider_name: str
    provider_slug: str
    facets: tuple[str, ...]
    requirement_sets: tuple[RequirementSetSummary, ...]


@frozen
class ProviderSummary:
    """
    Summary of a diagnostic provider and all its diagnostics.
    """

    name: str
    slug: str
    version: str
    diagnostics: tuple[DiagnosticSummary, ...]


def _extract_facet_values(filters: tuple[FacetFilter, ...], facet_name: str) -> tuple[str, ...]:
    """
    Extract unique values for a facet across all FacetFilters.

    Parameters
    ----------
    filters
        Tuple of FacetFilter instances to search.
    facet_name
        The facet key to extract values for.

    Returns
    -------
    :
        Sorted tuple of unique values found for the facet.
    """
    values: set[str] = set()
    for f in filters:
        if facet_name in f.facets:
            values.update(f.facets[facet_name])
    return tuple(sorted(values))


def _normalize_requirement_sets(
    data_requirements: Sequence[DataRequirement] | Sequence[Sequence[DataRequirement]],
) -> list[Sequence[DataRequirement]]:
    """
    Normalize data_requirements into a list of requirement sets.

    Handles both flat (single set) and nested (OR-logic) formats,
    mirroring the solver logic.

    Parameters
    ----------
    data_requirements
        Either a flat sequence of DataRequirement or a nested sequence
        of sequences (OR-logic).

    Returns
    -------
    :
        List of requirement sets, where each set is a sequence of DataRequirement.
    """
    if not data_requirements:
        return []

    first_item = next(iter(data_requirements))

    if isinstance(first_item, DataRequirement):
        # Single flat collection
        return [data_requirements]  # type: ignore[list-item]
    elif isinstance(first_item, Sequence):
        # Nested OR-logic collections
        return list(data_requirements)  # type: ignore[arg-type]
    else:
        raise TypeError(f"Expected DataRequirement or Sequence[DataRequirement], got {type(first_item)}")


def summarize_data_requirement(req: DataRequirement) -> DataRequirementSummary:
    """
    Create a summary of a single DataRequirement.

    Parameters
    ----------
    req
        The DataRequirement to summarize.

    Returns
    -------
    :
        A DataRequirementSummary with extracted facet values.
    """
    return DataRequirementSummary(
        source_type=req.source_type.value,
        variables=_extract_facet_values(req.filters, "variable_id"),
        experiments=_extract_facet_values(req.filters, "experiment_id"),
        tables=_extract_facet_values(req.filters, "table_id"),
        frequencies=_extract_facet_values(req.filters, "frequency"),
        group_by=req.group_by,
    )


def summarize_diagnostic(diagnostic: Diagnostic) -> DiagnosticSummary:
    """
    Create a summary of a diagnostic and its data requirements.

    Parameters
    ----------
    diagnostic
        The Diagnostic to summarize.

    Returns
    -------
    :
        A DiagnosticSummary with all requirement sets summarized.
    """
    requirement_sets = _normalize_requirement_sets(diagnostic.data_requirements)

    set_summaries = []
    for req_set in requirement_sets:
        req_summaries = tuple(summarize_data_requirement(req) for req in req_set)
        set_summaries.append(RequirementSetSummary(requirements=req_summaries))

    return DiagnosticSummary(
        name=diagnostic.name,
        slug=diagnostic.slug,
        provider_name=diagnostic.provider.name,
        provider_slug=diagnostic.provider.slug,
        facets=diagnostic.facets,
        requirement_sets=tuple(set_summaries),
    )


def summarize_provider(provider: DiagnosticProvider) -> ProviderSummary:
    """
    Create a summary of a provider and all its diagnostics.

    Parameters
    ----------
    provider
        The DiagnosticProvider to summarize.

    Returns
    -------
    :
        A ProviderSummary with all diagnostics summarized.
    """
    diag_summaries = tuple(summarize_diagnostic(d) for d in provider.diagnostics())
    return ProviderSummary(
        name=provider.name,
        slug=provider.slug,
        version=provider.version,
        diagnostics=diag_summaries,
    )


def collect_variables_by_experiment(
    providers: Iterable[DiagnosticProvider],
) -> dict[str, dict[str, set[str]]]:
    """
    Collect variables grouped by experiment and source type across providers.

    Parameters
    ----------
    providers
        Iterable of DiagnosticProvider instances to aggregate.

    Returns
    -------
    :
        Nested dict: experiment -> source_type -> set of variable_ids.
        Experiments with value ``"*"`` represent diagnostics that accept
        any experiment (no experiment_id filter).
    """
    result: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))

    for provider in providers:
        provider_summary = summarize_provider(provider)
        for diag in provider_summary.diagnostics:
            for req_set in diag.requirement_sets:
                for req in req_set.requirements:
                    experiments = req.experiments if req.experiments else ("*",)
                    for exp in experiments:
                        result[exp][req.source_type].update(req.variables)

    return dict(result)


def format_diagnostic_markdown(summary: DiagnosticSummary) -> str:
    """
    Format a single diagnostic summary as a markdown section.

    Parameters
    ----------
    summary
        The DiagnosticSummary to format.

    Returns
    -------
    :
        Markdown string describing the diagnostic.
    """
    lines = [
        f"### {summary.name}",
        "",
        f"**Slug**: `{summary.slug}`",
        "",
        f"**Facets**: {', '.join(f'`{f}`' for f in summary.facets)}",
        "",
    ]

    for i, req_set in enumerate(summary.requirement_sets):
        if len(summary.requirement_sets) > 1:
            lines.append(f"#### Option {i + 1}")
            lines.append("")

        for req in req_set.requirements:
            lines.append(f"**Source type**: `{req.source_type}`")
            lines.append("")
            if req.variables:
                lines.append(f"- **Variables**: {', '.join(f'`{v}`' for v in req.variables)}")
            if req.experiments:
                lines.append(f"- **Experiments**: {', '.join(f'`{e}`' for e in req.experiments)}")
            if req.tables:
                lines.append(f"- **Tables**: {', '.join(f'`{t}`' for t in req.tables)}")
            if req.frequencies:
                lines.append(f"- **Frequencies**: {', '.join(f'`{f}`' for f in req.frequencies)}")
            if req.group_by:
                lines.append(f"- **Group by**: {', '.join(f'`{g}`' for g in req.group_by)}")
            lines.append("")

    return "\n".join(lines)


def format_provider_markdown(summary: ProviderSummary) -> str:
    """
    Format a full provider summary as a markdown page.

    Parameters
    ----------
    summary
        The ProviderSummary to format.

    Returns
    -------
    :
        Markdown string for the full provider page.
    """
    lines = [
        f"# {summary.name}",
        "",
        f"**Slug**: `{summary.slug}`",
        "",
        f"**Version**: `{summary.version}`",
        "",
        f"**Diagnostics**: {len(summary.diagnostics)}",
        "",
        "---",
        "",
    ]

    for diag in summary.diagnostics:
        lines.append(format_diagnostic_markdown(diag))
        lines.append("---")
        lines.append("")

    return "\n".join(lines)
