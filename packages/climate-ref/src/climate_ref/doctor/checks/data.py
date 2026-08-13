"""
Checks over the data a deployment has ingested.

These look for the conditions that make a solve quietly do the wrong thing rather than fail:
reference data that no diagnostic can reach,
reference data that is missing so its diagnostics never run,
and datasets whose files cover the same period twice.
"""

from collections import defaultdict

from climate_ref.doctor.context import DoctorContext
from climate_ref.doctor.findings import Finding, Severity
from climate_ref.doctor.registry import check
from climate_ref.text import pluralise
from climate_ref_core.reference_data import (
    ESGF_OBS4MIPS,
    ReferenceDataset,
    collect_required_reference_data,
    source_ids_by_registry,
)
from climate_ref_core.source_types import SourceDatasetType
from climate_ref_core.summary import summarize_provider


@check(
    "duplicate-coverage",
    "Datasets holding more than one file for the same period",
)
def check_duplicate_coverage(context: DoctorContext) -> list[Finding]:
    """
    Find datasets holding more than one file for the same period.

    This happens when the same dataset is ingested from two collections at the same version:
    the files merge into one dataset because they share an ``instance_id``,
    and every diagnostic reading it then sees the overlapping period twice.

    The obs4REF registry and the obs4MIPs archive both carry several datasets, so ingesting both triggers it.

    Parameters
    ----------
    context
        The deployment to check.

    Returns
    -------
    :
        One finding per dataset with overlapping files.
    """
    findings = []

    for source_type in SourceDatasetType:
        catalog = context.catalog(source_type)
        if not len(catalog) or "start_time" not in catalog:
            continue

        for instance_id, group in catalog.groupby("instance_id"):
            spans = group[["start_time", "end_time", "path"]].dropna(subset=["start_time", "end_time"])
            if len(spans) < 2:  # noqa: PLR2004
                continue

            spans = spans.sort_values("start_time")
            starts = spans["start_time"].to_numpy()
            ends = spans["end_time"].to_numpy()
            # an overlap is one that starts before the latest end seen so far.
            running_end = ends[0]
            overlaps: list[tuple[str, str]] = []
            paths = spans["path"].tolist()
            for index in range(1, len(spans)):
                if starts[index] < running_end:
                    overlaps.append((paths[index - 1], paths[index]))
                running_end = max(running_end, ends[index])

            if not overlaps:
                continue

            roots = sorted({_collection_root(path) for pair in overlaps for path in pair})
            findings.append(
                Finding(
                    severity=Severity.ERROR,
                    summary=f"{instance_id} holds {len(spans)} files covering overlapping periods",
                    detail=f"{pluralise(len(overlaps), 'overlapping pair')}, across: {', '.join(roots)}.",
                    remedy=(
                        "A diagnostic reading one of these sees the overlapping period more than once. "
                        "This usually means the same dataset was ingested from two collections. "
                        "Re-ingest from one of them only."
                    ),
                )
            )

    return findings


def _collection_root(path: str, depth: int = 4) -> str:
    """Shorten a file path to the directory that identifies which collection it came from."""
    parts = str(path).split("/")
    return "/".join(parts[:depth]) if len(parts) > depth else str(path)


@check(
    "missing-reference-data",
    "Reference data the enabled diagnostics require but which is not ingested",
)
def check_missing_reference_data(context: DoctorContext) -> list[Finding]:
    """
    Find reference datasets the enabled diagnostics require but which are not ingested.

    An unmet reference requirement is silent: the diagnostic simply plans no executions,
    so a deployment can look healthy while producing nothing for whole diagnostics.

    Parameters
    ----------
    context
        The deployment to check.

    Returns
    -------
    :
        One finding per required dataset that is not ingested.
    """
    required = collect_required_reference_data(context.providers)
    if not required:
        return []

    ingested: dict[str, set[str]] = defaultdict(set)
    for source_type in SourceDatasetType:
        catalog = context.catalog(source_type)
        if len(catalog) and "source_id" in catalog:
            ingested[source_type.value].update(catalog["source_id"].unique())

    findings = []
    for dataset in sorted(required, key=lambda d: (d.supplier, d.source_id)):
        if dataset.source_id in ingested[dataset.source_type]:
            continue
        diagnostics = ", ".join(f"{d.provider_slug}/{d.slug}" for d in dataset.diagnostics)
        remedy, command = _how_to_obtain(dataset)
        findings.append(
            Finding(
                severity=Severity.WARNING,
                summary=(
                    f"{dataset.source_id} ({dataset.source_type}) is not ingested, so "
                    f"{pluralise(len(dataset.diagnostics), 'diagnostic')} will not run"
                ),
                detail=f"Needed for {', '.join(dataset.variable_ids)} by {diagnostics}.",
                remedy=remedy,
                command=command,
            )
        )
    return findings


def _how_to_obtain(dataset: ReferenceDataset) -> tuple[str, str]:
    """
    Explain where a required reference dataset comes from.

    The wording holds for every dataset obtained the same way, so the report can state it once
    for all of them.
    """
    if dataset.registry_name is not None:
        return (
            "Fetch these, then ingest the directory they land in.",
            f"ref datasets fetch-data --registry {dataset.registry_name} --output-directory <dir>",
        )
    if dataset.supplier == ESGF_OBS4MIPS:
        return (
            "These are published to obs4MIPs on ESGF, and `scripts/fetch-esgf.py` has a request "
            "for each. See the 'Download required datasets' guide.",
            "",
        )
    return ("No registry carries these and they are not known to be on ESGF.", "")


@check(
    "unreachable-source-type",
    "Data ingested under a source type that no enabled diagnostic asks for",
)
def check_unreachable_source_types(context: DoctorContext) -> list[Finding]:
    """
    Find data ingested under a source type that no enabled diagnostic asks for.

    The clearest case is obs4REF.
    The ``obs4ref`` source type exists and can be ingested,
    but no diagnostic declares an obs4REF data requirement,
    and the solver only matches a requirement against its own source type.
    Data ingested that way is never selected.

    Parameters
    ----------
    context
        The deployment to check.

    Returns
    -------
    :
        One finding per source type that holds data nothing asks for.
    """
    requested: set[str] = set()
    for provider in context.providers:
        for diagnostic in summarize_provider(provider).diagnostics:
            for requirement_set in diagnostic.requirement_sets:
                for requirement in requirement_set.requirements:
                    requested.add(requirement.source_type)

    findings = []
    for source_type in SourceDatasetType:
        if source_type.value in requested:
            continue
        catalog = context.catalog(source_type)
        if not len(catalog):
            continue
        count = catalog["instance_id"].nunique()
        findings.append(
            Finding(
                severity=Severity.WARNING,
                summary=(
                    f"{pluralise(count, 'dataset')} ingested as '{source_type.value}' "
                    "that no enabled diagnostic requires"
                ),
                detail=(
                    "The solver matches a data requirement against its own source type only, "
                    "so nothing will select these datasets."
                ),
                remedy=(
                    "If this is obs4REF data, re-ingest it with "
                    f"`--source-type {SourceDatasetType.obs4MIPs.value}`."
                ),
            )
        )
    return findings


@check(
    "overlapping-registries",
    "Reference datasets that more than one registry carries",
)
def check_overlapping_registries(context: DoctorContext) -> list[Finding]:
    """
    Report datasets that more than one registry carries.

    Fetching both copies is what produces the duplicate coverage that `check_duplicate_coverage` finds,
    so this is the warning before the error.

    Parameters
    ----------
    context
        The deployment to check.

    Returns
    -------
    :
        One finding per dataset carried by more than one registry.
    """
    findings = []
    # obs4REF registries answer for two source types, so the same overlap is reported under both.
    # Collapse to one finding per dataset and set of registries.
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for (_, source_id), registries in sorted(source_ids_by_registry().items()):
        if len(registries) < 2:  # noqa: PLR2004
            continue
        key = (source_id, tuple(registries))
        if key in seen:
            continue
        seen.add(key)
        findings.append(
            Finding(
                severity=Severity.INFO,
                summary=f"{source_id} is carried by {pluralise(len(registries), 'registry', 'registries')}",
                detail=f"Carried by: {', '.join(registries)}.",
                remedy=(
                    "Fetch each of these from one registry only. Ingesting two copies of the same "
                    "version gives one dataset holding both sets of files."
                ),
            )
        )
    return findings
