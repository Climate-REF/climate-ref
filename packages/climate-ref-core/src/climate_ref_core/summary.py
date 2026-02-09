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


@frozen
class DiagnosticReference:
    """
    A reference to a diagnostic, linking name and slug to its provider.
    """

    name: str
    slug: str
    provider_slug: str

    @property
    def label(self) -> str:
        """Human-readable label: 'Name (provider)'."""
        return f"{self.name} ({self.provider_slug})"

    @property
    def sort_key(self) -> str:
        """Sort by provider then name."""
        return f"{self.provider_slug}/{self.name}"


@frozen
class VariableEntry:
    """
    Summary of a variable's usage across diagnostics.
    """

    variable_id: str
    diagnostics: tuple[DiagnosticReference, ...]


@frozen
class ExperimentSummaryTable:
    """
    Summary of all variables required for a specific experiment.
    """

    experiment: str
    variables: tuple[VariableEntry, ...]


@frozen
class SourceTypeSummaryTable:
    """
    Summary of all experiments and variables for a source type across providers.
    """

    source_type: str
    experiments: tuple[ExperimentSummaryTable, ...]


def collect_by_source_type(
    providers: Iterable[DiagnosticProvider],
) -> list[SourceTypeSummaryTable]:
    """
    Collect variables grouped by source type and experiment across all providers.

    For each source type, produces experiments with their required variables
    and the diagnostics that need them.

    Parameters
    ----------
    providers
        Iterable of DiagnosticProvider instances to aggregate.

    Returns
    -------
    :
        List of SourceTypeSummaryTable, sorted by source type name.
        Within each source type, experiments are sorted alphabetically
        (with ``"*"`` sorted first).
    """
    # source_type -> experiment -> variable -> set of DiagnosticReference
    data: dict[str, dict[str, dict[str, set[DiagnosticReference]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(set))
    )

    for provider in providers:
        provider_summary = summarize_provider(provider)
        for diag in provider_summary.diagnostics:
            ref = DiagnosticReference(
                name=diag.name,
                slug=diag.slug,
                provider_slug=diag.provider_slug,
            )
            for req_set in diag.requirement_sets:
                for req in req_set.requirements:
                    experiments = req.experiments if req.experiments else ("*",)
                    for exp in experiments:
                        for var in req.variables:
                            data[req.source_type][exp][var].add(ref)

    tables = []
    for source_type in sorted(data.keys()):
        exp_tables = []
        for experiment in sorted(data[source_type].keys()):
            var_entries = []
            for var_id in sorted(data[source_type][experiment].keys()):
                refs = data[source_type][experiment][var_id]
                var_entries.append(
                    VariableEntry(
                        variable_id=var_id,
                        diagnostics=tuple(sorted(refs, key=lambda r: r.sort_key)),
                    )
                )
            exp_tables.append(ExperimentSummaryTable(experiment=experiment, variables=tuple(var_entries)))
        tables.append(SourceTypeSummaryTable(source_type=source_type, experiments=tuple(exp_tables)))

    return tables


def format_overview_markdown(tables: list[SourceTypeSummaryTable]) -> str:
    """
    Format source-type summary tables as a markdown page.

    Produces a section per source type with sub-sections per experiment,
    each containing a table of variables and the diagnostics that need them.

    Parameters
    ----------
    tables
        List of SourceTypeSummaryTable to format.

    Returns
    -------
    :
        Markdown string with tables grouped by source type and experiment.
    """
    lines = [
        "# Diagnostics Overview",
        "",
        "Variables required across all diagnostic providers, grouped by source type and experiment.",
        "",
    ]

    for table in tables:
        lines.append(f"## {table.source_type}")
        lines.append("")

        for exp_table in table.experiments:
            exp_label = exp_table.experiment if exp_table.experiment != "*" else "Any experiment"
            lines.append(f"### {exp_label}")
            lines.append("")
            lines.append("| Variable | Diagnostics |")
            lines.append("| --- | --- |")

            for var in exp_table.variables:
                links = []
                for ref in var.diagnostics:
                    anchor = ref.slug
                    links.append(f"[{ref.name}]({ref.provider_slug}.md#{anchor})")
                lines.append(f"| `{var.variable_id}` | {', '.join(links)} |")

            lines.append("")

    return "\n".join(lines)


def _format_requirement_table(req: DataRequirementSummary) -> list[str]:
    """
    Format a single DataRequirementSummary as a markdown table.

    Parameters
    ----------
    req
        The DataRequirementSummary to format.

    Returns
    -------
    :
        List of markdown lines for the requirement table.
    """
    lines = [
        "| | |",
        "| --- | --- |",
        f"| **Source type** | `{req.source_type}` |",
    ]
    if req.variables:
        lines.append(f"| **Variables** | {', '.join(f'`{v}`' for v in req.variables)} |")
    if req.experiments:
        lines.append(f"| **Experiments** | {', '.join(f'`{e}`' for e in req.experiments)} |")
    if req.tables:
        lines.append(f"| **Tables** | {', '.join(f'`{t}`' for t in req.tables)} |")
    if req.frequencies:
        lines.append(f"| **Frequencies** | {', '.join(f'`{f}`' for f in req.frequencies)} |")
    if req.group_by:
        lines.append(f"| **Group by** | {', '.join(f'`{g}`' for g in req.group_by)} |")
    return lines


def format_diagnostic_markdown(summary: DiagnosticSummary) -> str:
    """
    Format a single diagnostic summary as a markdown section.

    Uses admonitions for metadata and tabs for OR-logic options.

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
        f"### {summary.name} {{ #{summary.slug} }}",
        "",
    ]

    # Aggregate across all requirement sets for the summary card
    all_source_types: set[str] = set()
    all_variables: set[str] = set()
    all_experiments: set[str] = set()
    for req_set in summary.requirement_sets:
        for req in req_set.requirements:
            all_source_types.add(req.source_type)
            all_variables.update(req.variables)
            all_experiments.update(req.experiments)

    source_types = ", ".join(f"`{s}`" for s in sorted(all_source_types)) if all_source_types else "None"
    variables = ", ".join(f"`{v}`" for v in sorted(all_variables)) if all_variables else "any"
    experiments = ", ".join(f"`{e}`" for e in sorted(all_experiments)) if all_experiments else "any"
    facets = ", ".join(f"`{f}`" for f in summary.facets) if summary.facets else "None"

    # Metadata admonition
    lines.extend(
        [
            f"/// admonition | `{summary.slug}`",
            "    type: info",
            "",
            f"**Source types**: {source_types}  ",
            f"**Variables**: {variables}  ",
            f"**Experiments**: {experiments}  ",
            f"**Facets**: {facets}",
            "",
            "///",
            "",
        ]
    )

    # Data requirements: tabs for OR-logic, admonition for single option
    if len(summary.requirement_sets) > 1:
        for i, req_set in enumerate(summary.requirement_sets):
            # Build a tab label from the source types in this option
            tab_source_types = sorted({req.source_type for req in req_set.requirements})
            tab_label = " + ".join(tab_source_types) if tab_source_types else f"Option {i + 1}"
            lines.append(f'=== "{tab_label}"')
            lines.append("")
            for req in req_set.requirements:
                for table_line in _format_requirement_table(req):
                    lines.append(f"    {table_line}")
                lines.append("")
    else:
        for req_set in summary.requirement_sets:
            for req in req_set.requirements:
                lines.extend(_format_requirement_table(req))
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
        "/// admonition | Provider details",
        "    type: info",
        "",
        f"**Slug**: `{summary.slug}`  ",
        f"**Version**: `{summary.version}`  ",
        f"**Diagnostics**: {len(summary.diagnostics)}",
        "",
        "///",
        "",
    ]

    for diag in summary.diagnostics:
        lines.append(format_diagnostic_markdown(diag))
        lines.append("---")
        lines.append("")

    return "\n".join(lines)
