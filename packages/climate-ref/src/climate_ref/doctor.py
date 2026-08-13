"""
Health checks for a Climate-REF deployment.

These look for the conditions that make a solve quietly do the wrong thing rather than fail:
reference data that no diagnostic can reach, reference data that is missing so its diagnostics
never run, and datasets whose files cover the same period twice.

Each check is a function taking a `DoctorContext` and returning `Finding`s. Checks read the
database and the configuration; none of them write.
"""

from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Sequence

import pandas as pd
from attrs import define, field, frozen
from loguru import logger

from climate_ref.config import Config
from climate_ref.database import Database
from climate_ref.datasets import get_dataset_adapter
from climate_ref_core.diagnostics import DataRequirement, Diagnostic
from climate_ref_core.providers import DiagnosticProvider
from climate_ref_core.reference_data import (
    ESGF_OBS4MIPS,
    ReferenceDataset,
    collect_required_reference_data,
    source_ids_by_registry,
)
from climate_ref_core.source_types import SourceDatasetType


class Severity:
    """How much a finding matters. Ordered worst-first for reporting."""

    ERROR = "error"
    """Results computed in this state are wrong."""

    WARNING = "warning"
    """Something the deployment probably did not intend, but results remain valid."""

    INFO = "info"
    """Worth knowing; no action required."""


SEVERITY_ORDER = (Severity.ERROR, Severity.WARNING, Severity.INFO)

EMPTY_CATALOG = pd.DataFrame()
"""Stands in for a source type with nothing ingested."""


@frozen
class Finding:
    """
    One problem found by a check.
    """

    check: str
    """Slug of the check that produced it, e.g. ``duplicate-coverage``."""

    severity: str
    """One of `Severity`."""

    summary: str
    """One line stating what is wrong."""

    detail: str = ""
    """Optional further explanation, including what to do about it."""


@define
class DoctorContext:
    """
    The deployment being checked.

    Providers and catalogs are loaded lazily so a check that does not need them does not
    pay for them, and so a failure to load one provider does not stop the other checks.
    """

    config: Config | None
    database: Database | None
    _providers: list[DiagnosticProvider] | None = field(default=None, alias="_providers")
    _catalogs: dict[SourceDatasetType, pd.DataFrame] = field(factory=dict, alias="_catalogs")

    @classmethod
    def from_catalogs(
        cls,
        catalogs: dict[SourceDatasetType, pd.DataFrame],
        providers: Iterable[DiagnosticProvider],
    ) -> "DoctorContext":
        """
        Build a context from catalogs already in hand, with no database behind it.

        Source types absent from ``catalogs`` are treated as having nothing ingested, so
        every check can run without reaching for a database that is not there.
        """
        complete = {
            source_type: catalogs.get(source_type, EMPTY_CATALOG) for source_type in SourceDatasetType
        }
        return cls(config=None, database=None, _providers=list(providers), _catalogs=complete)

    @property
    def providers(self) -> list[DiagnosticProvider]:
        """The diagnostic providers this deployment has enabled."""
        if self._providers is None:
            from climate_ref.provider_registry import ProviderRegistry  # noqa: PLC0415

            if self.config is None or self.database is None:
                raise ValueError("This context has no configuration to load providers from")
            self._providers = list(ProviderRegistry.build_from_config(self.config, self.database).providers)
        return self._providers

    def catalog(self, source_type: SourceDatasetType) -> pd.DataFrame:
        """
        Load the ingested catalog for a source type, one row per file.

        Returns an empty frame when nothing of that type has been ingested.
        """
        if source_type not in self._catalogs:
            if self.database is None:
                raise ValueError("This context has no database to load a catalog from")
            adapter = get_dataset_adapter(source_type.value)
            self._catalogs[source_type] = adapter.load_catalog(self.database)
        return self._catalogs[source_type]


def check_duplicate_coverage(context: DoctorContext) -> list[Finding]:
    """
    Find datasets holding more than one file for the same period.

    This happens when the same dataset is ingested from two collections at the same version:
    the files merge into one dataset because they share an ``instance_id``, and every diagnostic
    reading it then sees the overlapping period twice. The obs4REF registry and the obs4MIPs
    archive both carry several datasets, so ingesting both triggers it.
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
            # Walking the files in start order, an overlap is one that starts before the
            # latest end seen so far.
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
                    check="duplicate-coverage",
                    severity=Severity.ERROR,
                    summary=f"{instance_id} holds {len(spans)} files covering overlapping periods",
                    detail=(
                        f"{len(overlaps)} overlapping pair(s), across: {', '.join(roots)}. "
                        "A diagnostic reading this dataset sees the overlapping period more than "
                        "once. This usually means the same dataset was ingested from two "
                        "collections; re-ingest from one of them only."
                    ),
                )
            )

    return findings


def _collection_root(path: str, depth: int = 4) -> str:
    """Shorten a file path to the directory that identifies which collection it came from."""
    parts = str(path).split("/")
    return "/".join(parts[:depth]) if len(parts) > depth else str(path)


def check_missing_reference_data(context: DoctorContext) -> list[Finding]:
    """
    Find reference datasets the enabled diagnostics require but which are not ingested.

    An unmet reference requirement is silent: the diagnostic simply plans no executions,
    so a deployment can look healthy while producing nothing for whole diagnostics.
    """
    required = collect_required_reference_data(context.providers)
    if not required:
        return []

    ingested: dict[str, set[str]] = defaultdict(set)
    for source_type in SourceDatasetType:
        catalog = context.catalog(source_type)
        if len(catalog) and "source_id" in catalog:
            ingested[source_type.value].update(catalog["source_id"].unique())
    # Deliberately no fallback between source types. A requirement is only satisfied by data
    # ingested under its own source type, so obs4REF data sitting in the obs4ref table does
    # not count towards an obs4MIPs requirement -- that is the situation this check exists to
    # surface, and `check_unreachable_source_types` names the cause.
    findings = []
    for dataset in sorted(required, key=lambda d: (d.supplier, d.source_id)):
        if dataset.source_id in ingested[dataset.source_type]:
            continue
        findings.append(
            Finding(
                check="missing-reference-data",
                severity=Severity.WARNING,
                summary=(
                    f"{dataset.source_id} ({dataset.source_type}) is not ingested, "
                    f"so {len(dataset.diagnostics)} diagnostic(s) will not run"
                ),
                detail=_how_to_obtain(dataset),
            )
        )
    return findings


def _how_to_obtain(dataset: ReferenceDataset) -> str:
    """Explain where a required reference dataset comes from."""
    diagnostics = ", ".join(f"{d.provider_slug}/{d.slug}" for d in dataset.diagnostics)
    variables = ", ".join(dataset.variable_ids)
    if dataset.registry_name is not None:
        how = (
            f"Fetch with `ref datasets fetch-data --registry {dataset.registry_name} "
            "--output-directory <dir>`, then ingest that directory."
        )
    elif dataset.supplier == ESGF_OBS4MIPS:
        how = (
            "Published to obs4MIPs on ESGF; `scripts/fetch-esgf.py` has a request for it. "
            "See the 'Download required datasets' guide."
        )
    else:
        how = "No registry carries this dataset and it is not known to be on ESGF."
    return f"Needed for {variables} by {diagnostics}. {how}"


def check_unreachable_source_types(context: DoctorContext) -> list[Finding]:
    """
    Find data ingested under a source type that no enabled diagnostic asks for.

    The clearest case is obs4REF: the ``obs4ref`` source type exists and can be ingested,
    but no diagnostic declares an obs4REF data requirement, and the solver only matches a
    requirement against its own source type. Data ingested that way is never selected.
    """
    requested: set[SourceDatasetType] = set()
    for provider in context.providers:
        for diagnostic in provider.diagnostics():
            for requirement in _iter_requirements(diagnostic):
                requested.add(requirement.source_type)

    findings = []
    for source_type in SourceDatasetType:
        if source_type in requested:
            continue
        catalog = context.catalog(source_type)
        if not len(catalog):
            continue
        count = catalog["instance_id"].nunique()
        findings.append(
            Finding(
                check="unreachable-source-type",
                severity=Severity.WARNING,
                summary=(
                    f"{count} dataset(s) are ingested as '{source_type.value}', "
                    "which no enabled diagnostic requires"
                ),
                detail=(
                    "The solver matches a data requirement against its own source type only, "
                    f"so nothing will select these datasets. If this is obs4REF data, re-ingest "
                    f"it with `--source-type {SourceDatasetType.obs4MIPs.value}`."
                ),
            )
        )
    return findings


def _iter_requirements(diagnostic: Diagnostic) -> Iterator[DataRequirement]:
    """
    Yield every data requirement a diagnostic declares.

    A diagnostic declares either a flat sequence of requirements, or a sequence of
    alternative branches of them. Data for any branch may end up being selected, so every
    branch counts as requested.
    """
    for item in diagnostic.data_requirements:
        if isinstance(item, DataRequirement):
            yield item
        else:
            yield from item


def check_overlapping_registries(context: DoctorContext) -> list[Finding]:
    """
    Report datasets that more than one registry carries.

    Fetching both copies is what produces the duplicate coverage that
    `check_duplicate_coverage` finds, so this is the warning before the error.
    """
    findings = []
    # obs4REF registries answer for two source types, so the same overlap is reported under
    # both. Collapse to one finding per dataset and set of registries.
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
                check="overlapping-registries",
                severity=Severity.INFO,
                summary=f"{source_id} is carried by {len(registries)} registries",
                detail=(
                    f"Carried by: {', '.join(registries)}. Fetch it from one of them only; "
                    "ingesting two copies of the same version gives one dataset holding both "
                    "sets of files."
                ),
            )
        )
    return findings


CHECKS: tuple[Callable[[DoctorContext], list[Finding]], ...] = (
    check_duplicate_coverage,
    check_missing_reference_data,
    check_unreachable_source_types,
    check_overlapping_registries,
)


def run_checks(
    context: DoctorContext,
    checks: Iterable[Callable[[DoctorContext], list[Finding]]] = CHECKS,
) -> list[Finding]:
    """
    Run the checks and collect their findings, worst first.

    A check that raises is reported as a finding rather than stopping the run, so one broken
    check cannot hide the others.

    Parameters
    ----------
    context
        The deployment to check.
    checks
        The checks to run. Defaults to all of them.

    Returns
    -------
    :
        Findings ordered by severity, then by the check that produced them.
    """
    findings: list[Finding] = []
    for check in checks:
        try:
            findings.extend(check(context))
        except Exception as exc:
            logger.exception(f"Check '{check.__name__}' failed")
            findings.append(
                Finding(
                    check=check.__name__,
                    severity=Severity.ERROR,
                    summary=f"Check '{check.__name__}' could not run",
                    detail=f"{type(exc).__name__}: {exc}",
                )
            )

    return sorted(findings, key=lambda f: (SEVERITY_ORDER.index(f.severity), f.check, f.summary))


def worst_severity(findings: Sequence[Finding]) -> str | None:
    """Return the most serious severity present, or ``None`` when there are no findings."""
    for severity in SEVERITY_ORDER:
        if any(finding.severity == severity for finding in findings):
            return severity
    return None
