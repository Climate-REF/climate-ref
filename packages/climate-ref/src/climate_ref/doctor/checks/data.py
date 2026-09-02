"""
Checks over the data a deployment has ingested.

These look for the conditions that make a solve quietly do the wrong thing rather than fail:
reference data that no diagnostic can reach,
reference data that is missing so its diagnostics never run,
obs4REF data ingested under the obs4MIPs source type,
obs4REF data that obs4MIPs has since published,
datasets whose files cover the same period twice,
and diagnostics the ingested data cannot solve at all.
"""

from collections import defaultdict
from collections.abc import Mapping

import pandas as pd

from climate_ref.data_catalog import DataCatalog
from climate_ref.datasets.obs4mips import in_collection_directory
from climate_ref.doctor.context import DoctorContext
from climate_ref.doctor.findings import Finding, Severity
from climate_ref.doctor.registry import check
from climate_ref.solver import (
    apply_obs4ref_fallback,
    as_frame,
    extract_covered_datasets,
    obs_dataset_key,
    solve_executions,
)
from climate_ref.text import pluralise
from climate_ref_core.diagnostics import Diagnostic
from climate_ref_core.exceptions import InvalidDiagnosticException
from climate_ref_core.reference_data import (
    ESGF_OBS4MIPS,
    ReferenceDataset,
    collect_required_reference_data,
    source_ids_by_registry,
)
from climate_ref_core.source_types import SourceDatasetType
from climate_ref_core.summary import normalize_requirement_sets, summarize_provider


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
        required = {"instance_id", "start_time", "end_time", "path"}
        if not len(catalog) or not required.issubset(catalog.columns):
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

    # obs4REF fills in whatever obs4MIPs lacks, so either satisfies an obs4MIPs requirement.
    ingested[SourceDatasetType.obs4MIPs.value] |= ingested[SourceDatasetType.obs4REF.value]

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

    The solver only matches a requirement against its own source type,
    so data ingested under a type nothing asks for is never selected.
    The one exception is obs4REF data, which fills in for obs4MIPs requirements.

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
    if SourceDatasetType.obs4MIPs.value in requested:
        requested.add(SourceDatasetType.obs4REF.value)

    findings = []
    for source_type in SourceDatasetType:
        if source_type.value in requested:
            continue
        catalog = context.catalog(source_type)
        if not len(catalog) or "instance_id" not in catalog.columns:
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
                    "Re-ingest the data under the source type the diagnostics ask for, "
                    "or enable a provider that uses it."
                ),
            )
        )
    return findings


@check(
    "misfiled-obs4ref",
    "obs4REF data ingested under the obs4MIPs source type",
)
def check_misfiled_obs4ref(context: DoctorContext) -> list[Finding]:
    """
    Find obs4REF data that was ingested as obs4MIPs.

    Earlier releases ingested the obs4REF collection this way, and it still solves.
    The cost is that the catalog no longer shows which datasets came from the registry
    and which from the archive, and a later obs4MIPs publication cannot take over from it.

    A dataset counts as obs4REF when its files sit under an ``obs4REF`` directory,
    which is how the registry lays them out.
    Carrying a ``source_id`` the registry also carries is not enough,
    because the four datasets published to both archives are legitimately ingested as obs4MIPs.

    Parameters
    ----------
    context
        The deployment to check.

    Returns
    -------
    :
        One finding when any such data is present.
    """
    catalog = context.catalog(SourceDatasetType.obs4MIPs)
    if not len(catalog) or not {"instance_id", "path"}.issubset(catalog.columns):
        return []

    misfiled = in_collection_directory(catalog["path"], "obs4REF")
    if not misfiled.any():
        return []

    instance_ids = sorted(catalog.loc[misfiled, "instance_id"].unique())
    return [
        Finding(
            severity=Severity.WARNING,
            summary=f"{pluralise(len(instance_ids), 'obs4REF dataset')} ingested as obs4mips",
            detail="Affected: " + ", ".join(instance_ids) + ".",
            remedy=(
                "Re-ingest the obs4REF collection under its own source type, "
                "then retract each of the obs4mips rows above with `ref datasets retract <instance_id>`."
            ),
            command="ref datasets ingest --source-type obs4ref <dir>",
        )
    ]


@check(
    "superseded-obs4ref",
    "obs4REF datasets that the obs4MIPs archive has since published",
)
def check_superseded_obs4ref(context: DoctorContext) -> list[Finding]:
    """
    Find obs4REF datasets for which an obs4MIPs copy is also ingested.

    The solver takes the obs4MIPs copy, so the obs4REF one is no longer used.
    This is the signal that a dataset can be dropped from the obs4REF registry.

    Parameters
    ----------
    context
        The deployment to check.

    Returns
    -------
    :
        One finding per superseded obs4REF dataset.
    """
    obs4mips = context.catalog(SourceDatasetType.obs4MIPs)
    obs4ref = context.catalog(SourceDatasetType.obs4REF)
    if not len(obs4mips) or not len(obs4ref) or "instance_id" not in obs4mips or "instance_id" not in obs4ref:
        return []

    published = set(obs_dataset_key(obs4mips["instance_id"]))
    superseded = obs4ref[obs_dataset_key(obs4ref["instance_id"]).isin(published)]
    return [
        Finding(
            severity=Severity.INFO,
            summary=f"{instance_id} is superseded by the obs4MIPs copy",
            remedy=(
                "The obs4MIPs copy is used instead. "
                "These can be retracted, and dropped from the obs4REF registry."
            ),
        )
        for instance_id in sorted(superseded["instance_id"].unique())
    ]


@check(
    "unsolvable-diagnostics",
    "Enabled diagnostics for which the ingested data produces no executions",
)
def check_unsolvable_diagnostics(context: DoctorContext) -> list[Finding]:
    """
    Find enabled diagnostics that the ingested data cannot solve at all.

    This runs the solver against the ingested catalogs, diagnostic by diagnostic,
    so it catches everything the narrower checks do not:
    a filter no dataset matches, a constraint no group satisfies, a source type nothing was ingested under.

    Parameters
    ----------
    context
        The deployment to check.

    Returns
    -------
    :
        One finding per diagnostic with no executions.
    """
    catalogs: dict[SourceDatasetType, DataCatalog] = {
        source_type: context.data_catalog(source_type) for source_type in SourceDatasetType
    }

    available = apply_obs4ref_fallback(catalogs)

    findings = []
    for provider in context.providers:
        for diagnostic in provider.diagnostics():
            try:
                solvable = any(True for _ in solve_executions(catalogs, diagnostic, provider))
            except InvalidDiagnosticException:
                solvable = False
            if solvable:
                continue
            findings.append(
                Finding(
                    severity=Severity.WARNING,
                    summary=f"{provider.slug}/{diagnostic.slug} has no executions",
                    detail=_why_unsolvable(diagnostic, available),
                    remedy="Ingest the data the unmet requirement names, then run the solver again.",
                )
            )
    return findings


def _why_unsolvable(
    diagnostic: Diagnostic,
    available: Mapping[SourceDatasetType, pd.DataFrame | DataCatalog],
) -> str:
    """
    Explain which requirement the ingested data fails to meet.

    Each requirement is checked on its own, so the first one with no matching group names
    the data to fetch. When every requirement matches something, the failure lies in how
    they combine, which is reported as such.
    """
    reasons = []
    for requirements in normalize_requirement_sets(diagnostic.data_requirements):
        for requirement in requirements:
            catalog = available[requirement.source_type]
            frame = as_frame(catalog)
            if not len(frame):
                reasons.append(f"nothing is ingested as {requirement.source_type.value}")
                break
            if not extract_covered_datasets(catalog, requirement):
                facets = "; ".join(
                    ", ".join(f"{k}={'|'.join(v)}" for k, v in sorted(f.facets.items()))
                    for f in requirement.filters
                )
                reasons.append(f"no {requirement.source_type.value} datasets match {facets}")
                break
        else:
            reasons.append("every requirement matches data, but no complete group satisfies the constraints")
    return "Unmet: " + ". ".join(dict.fromkeys(reasons)) + "."


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
