"""
What reference data the diagnostics need, and which collection supplies it.

A deployment draws its observational data from several places:
the obs4REF registry, the obs4MIPs archive on ESGF, and the provider registries (PMP climatology, ILAMB).
Which of them supplies a given dataset is not stated anywhere the user can read,
so the answer has had to be reconstructed by hand from the providers' data requirements.
This module works it out once, so that the documentation and ``ref doctor`` agree.

Provenance is resolved per ``source_id`` rather than per variable. A registry either carries
a dataset or it does not, and asking whether a specific (``source_id``, ``variable_id``) pair
exists is a question only the archive can answer: ESGF intersects its facets, so a requirement
naming four sources and eight variables does not imply all thirty-two combinations exist.
"""

from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping
from typing import Any

from attrs import frozen

from climate_ref_core.dataset_registry import (
    DatasetRegistryManager,
    RegistryUseCase,
    dataset_registry_manager,
)
from climate_ref_core.providers import DiagnosticProvider
from climate_ref_core.source_types import SourceDatasetType
from climate_ref_core.summary import DiagnosticReference, summarize_provider

ESGF_OBS4MIPS = "ESGF obs4MIPs"
"""Supplier used for reference data that no registry carries."""

UNKNOWN_SUPPLIER = "unknown"
"""Supplier used when a required dataset is in no registry and no archive is known to hold it."""

# The source types whose data is observational reference data rather than model output.
REFERENCE_SOURCE_TYPES = frozenset(
    {
        SourceDatasetType.obs4MIPs,
        SourceDatasetType.obs4REF,
        SourceDatasetType.PMPClimatology,
        SourceDatasetType.ESMValToolReference,
    }
)


@frozen
class ReferenceDataset:
    """
    A reference dataset the enabled diagnostics require, and where it comes from.
    """

    source_type: str
    """Value of the source type the requirement asks for, e.g. ``obs4mips``."""

    source_id: str
    """The dataset's ``source_id``."""

    variable_ids: tuple[str, ...]
    """
    Every variable the diagnostics ask this dataset for.

    This is the union across requirements, not a claim that the dataset publishes each one.
    """

    registry_name: str | None
    """The registry that carries it, or ``None`` when it has to be fetched from ESGF."""

    diagnostics: tuple[DiagnosticReference, ...]
    """The diagnostics that require it, sorted by provider then name."""

    @property
    def is_from_registry(self) -> bool:
        """Whether a `ref datasets fetch-data` call can retrieve this dataset."""
        return self.registry_name is not None

    @property
    def supplier(self) -> str:
        """Where the dataset comes from: a registry name, `ESGF_OBS4MIPS`, or `UNKNOWN_SUPPLIER`."""
        if self.registry_name is not None:
            return self.registry_name
        if self.source_type == SourceDatasetType.obs4MIPs.value:
            return ESGF_OBS4MIPS
        return UNKNOWN_SUPPLIER


def _registry_key_parser(source_type: SourceDatasetType) -> Callable[[str], dict[str, Any]] | None:
    """
    Get the function that reads a registry's keys, or ``None`` if its keys carry no source_id.

    A registry's key format follows from the source type it supplies, not from its name, so
    every registry declaring that source type is read the same way.

    Imported lazily: the parsers live alongside the ESGF request machinery, which pulls in
    pandas, and this module is imported by the documentation build.
    """
    from climate_ref_core.esgf.registry import (  # noqa: PLC0415
        _parse_obs4ref_key,
        _parse_pmp_climatology_key,
    )

    parsers = {
        SourceDatasetType.obs4REF: _parse_obs4ref_key,
        SourceDatasetType.PMPClimatology: _parse_pmp_climatology_key,
    }
    return parsers.get(source_type)


def source_ids_by_registry(
    manager: DatasetRegistryManager | None = None,
) -> dict[tuple[str, str], list[str]]:
    """
    Map each dataset a reference registry carries to the registries that carry it.

    Keyed on (source type, ``source_id``) rather than ``source_id`` alone: a registry only
    supplies a dataset for the source type it is registered against, so the ERA-5 climatology
    in the PMP registry does not satisfy a requirement for obs4MIPs ERA-5.

    Registries whose keys do not encode a ``source_id`` are skipped: their contents cannot be
    matched to a data requirement this way, and are fetched by the provider instead.

    Parameters
    ----------
    manager
        Registry manager to read. Defaults to the process-wide one.

    Returns
    -------
    :
        Mapping of (source type value, ``source_id``) to the names of the registries carrying
        it, in registration order. More than one name means the collections overlap.
    """
    manager = manager if manager is not None else dataset_registry_manager

    found: dict[tuple[str, str], list[str]] = defaultdict(list)
    for name in manager.keys():
        entry = manager.entry(name)
        if entry.use_case is not RegistryUseCase.reference or entry.source_type is None:
            continue
        parser = _registry_key_parser(entry.source_type)
        if parser is None:
            continue
        # The obs4REF registry carries what obs4MIPs has not published yet,
        # so it supplies requirements written against either.
        source_types = {entry.source_type.value}
        if entry.source_type is SourceDatasetType.obs4REF:
            source_types.add(SourceDatasetType.obs4MIPs.value)

        for key in entry.registry.registry:
            metadata: Mapping[str, Any] = parser(key)
            source_id = metadata.get("source_id")
            if not source_id:
                continue
            for source_type in source_types:
                names = found[(source_type, source_id)]
                if name not in names:
                    names.append(name)
    return dict(found)


def collect_required_reference_data(
    providers: Iterable[DiagnosticProvider],
    manager: DatasetRegistryManager | None = None,
) -> list[ReferenceDataset]:
    """
    Work out every reference dataset the given providers' diagnostics require.

    Parameters
    ----------
    providers
        The diagnostic providers to inspect.
    manager
        Registry manager used to resolve where each dataset comes from.

    Returns
    -------
    :
        One entry per (source type, source_id), sorted by source type then source_id.
    """
    registry_of = source_ids_by_registry(manager)

    variables: dict[tuple[str, str], set[str]] = defaultdict(set)
    diagnostics: dict[tuple[str, str], set[DiagnosticReference]] = defaultdict(set)
    # Source types a requirement may be met from, its own first
    suppliers: dict[tuple[str, str], list[str]] = defaultdict(list)

    for provider in providers:
        for diagnostic in summarize_provider(provider).diagnostics:
            reference = DiagnosticReference(
                name=diagnostic.name,
                slug=diagnostic.slug,
                provider_slug=diagnostic.provider_slug,
            )
            for requirement_set in diagnostic.requirement_sets:
                for requirement in requirement_set.requirements:
                    if requirement.source_type not in {t.value for t in REFERENCE_SOURCE_TYPES}:
                        continue
                    for source_id in requirement.source_ids:
                        key = (requirement.source_type, source_id)
                        variables[key].update(requirement.variables)
                        diagnostics[key].add(reference)
                        for source_type in (requirement.source_type, *requirement.fallback_source_types):
                            if source_type not in suppliers[key]:
                                suppliers[key].append(source_type)

    datasets = []
    for (source_type, source_id), variable_ids in variables.items():
        registry_name = next(
            (
                registry_of[(supplier, source_id)][0]
                for supplier in suppliers[(source_type, source_id)]
                if registry_of.get((supplier, source_id))
            ),
            None,
        )
        datasets.append(
            ReferenceDataset(
                source_type=source_type,
                source_id=source_id,
                variable_ids=tuple(sorted(variable_ids)),
                registry_name=registry_name,
                diagnostics=tuple(sorted(diagnostics[(source_type, source_id)], key=lambda r: r.sort_key)),
            )
        )

    return sorted(datasets, key=lambda d: (d.source_type, d.source_id))


def format_reference_data_markdown(datasets: Iterable[ReferenceDataset]) -> str:
    """
    Render the required reference data as a markdown page.

    Returns
    -------
    :
        Markdown, one table per supplier.
    """
    by_supplier: dict[str, list[ReferenceDataset]] = defaultdict(list)
    for dataset in datasets:
        by_supplier[dataset.supplier].append(dataset)

    lines = [
        "# Reference data",
        "",
        "Every observational dataset the diagnostics require, and where it comes from.",
        "This page is generated from the providers' data requirements, so it cannot drift.",
        "",
        "Variables are the union of everything the diagnostics ask a dataset for;",
        "a dataset does not necessarily publish all of them.",
        "Run `ref doctor` to check which of these a deployment is missing.",
        "",
    ]

    for supplier in sorted(by_supplier):
        entries = by_supplier[supplier]
        registry_name = entries[0].registry_name
        lines.append(f"## {supplier}")
        lines.append("")
        if registry_name is not None:
            lines.append("Fetch with:")
            lines.append("")
            lines.append("```bash")
            lines.append(f"ref datasets fetch-data --registry {registry_name} --output-directory <dir>")
            lines.append("```")
        else:
            lines.append(
                "Not carried by any registry, so it has to be fetched from the ESGF archive "
                "(see [Download required datasets](getting-started/02-download-datasets.md))."
            )
        lines.append("")
        lines.append("| `source_id` | Source type | Variables | Required by |")
        lines.append("| --- | --- | --- | --- |")
        for entry in entries:
            variables = ", ".join(f"`{v}`" for v in entry.variable_ids)
            ordered = sorted(entry.diagnostics, key=lambda d: (d.provider_slug, d.slug))
            diagnostics = ", ".join(f"`{d.provider_slug}/{d.slug}`" for d in ordered)
            lines.append(f"| `{entry.source_id}` | `{entry.source_type}` | {variables} | {diagnostics} |")
        lines.append("")

    return "\n".join(lines)
