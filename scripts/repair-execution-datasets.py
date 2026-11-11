"""
Rewrite the dataset links that the obs4REF fallback merge broke.

From 2026-09-02 that merge stored row positions instead of dataset ids
for executions of diagnostics that declare a fallback source type.
This fixes the latest execution of each affected group when it succeeded,
by matching it to a fresh solve on `dataset_hash`, and only counts the other affected executions.

Groups are keyed by diagnostic version,
so run it again with the release that produced each unmatched version it reports.
Dry run by default. Pass `--apply` to rewrite the links.
Solving may write dataset metadata even in a dry run, so run it while nothing else writes the database, e.g.

    kubectl exec -i <pod running climate-ref> -- \
        python - --provider pmp < scripts/repair-execution-datasets.py
"""

import collections
import datetime
import hashlib
from collections.abc import Iterable
from typing import Annotated

import attrs
import typer
from loguru import logger
from sqlalchemy import column, delete, select, table, text

from climate_ref.config import Config
from climate_ref.database import Database
from climate_ref.models import Dataset, Execution, ExecutionGroup
from climate_ref.models.diagnostic import Diagnostic as DiagnosticModel
from climate_ref.models.execution import execution_datasets
from climate_ref.models.provider import Provider as ProviderModel
from climate_ref.solver import DiagnosticExecution, ExecutionSolver, SolveFilterOptions, matches_filter
from climate_ref_core.datasets import ExecutionDatasetCollection
from climate_ref_core.diagnostics import DataRequirement, Diagnostic

app = typer.Typer()

BATCH_SIZE = 500

# obs4REF datasets legitimately stand in for obs4MIPs.
_MISMATCHED_TYPES_SQL = """
SELECT DISTINCT ed.execution_id
FROM execution_dataset ed
JOIN execution e ON e.id = ed.execution_id
JOIN dataset d ON d.id = ed.dataset_id
WHERE NOT EXISTS (
    SELECT 1 FROM execution_group_source_type s
    WHERE s.execution_group_id = e.execution_group_id
    AND (lower(s.source_type) = lower(d.dataset_type)
         OR (lower(s.source_type) = 'obs4mips' AND lower(d.dataset_type) = 'obs4ref'))
)
"""


@attrs.define
class Planner:
    """Matches solved executions to existing ones and collects the links to rewrite."""

    show: int
    diagnostic_ids: dict[tuple[str, str], int] = attrs.field(factory=dict)
    group_ids: dict[tuple[str, int, int], int] = attrs.field(factory=dict)
    group_versions: dict[int, tuple[int, int]] = attrs.field(factory=dict)
    executions: dict[tuple[int, str], list[int]] = attrs.field(factory=lambda: collections.defaultdict(list))
    execution_groups: dict[int, int] = attrs.field(factory=dict)
    successful: set[int] = attrs.field(factory=set)
    latest: set[int] = attrs.field(factory=set)
    dataset_ids: dict[str, int] = attrs.field(factory=dict)
    legacy_slugs: dict[str, str] = attrs.field(factory=dict)
    links: dict[int, list[int]] = attrs.field(factory=lambda: collections.defaultdict(list))
    matched: set[int] = attrs.field(factory=set)
    rewrites: dict[int, list[int]] = attrs.field(factory=dict)
    counts: collections.Counter[str] = attrs.field(factory=collections.Counter)
    examples: list[str] = attrs.field(factory=list)

    def load(self, db: Database) -> None:
        """Snapshot the ids needed to match solved executions."""
        session = db.session
        for diagnostic_id, diagnostic_slug, provider_slug in session.execute(
            select(DiagnosticModel.id, DiagnosticModel.slug, ProviderModel.slug).join(
                DiagnosticModel.provider
            )
        ):
            self.diagnostic_ids[(provider_slug, diagnostic_slug)] = diagnostic_id
        for group_id, key, diagnostic_id, diagnostic_version in session.execute(
            select(
                ExecutionGroup.id,
                ExecutionGroup.key,
                ExecutionGroup.diagnostic_id,
                ExecutionGroup.diagnostic_version,
            )
        ):
            self.group_ids[(key, diagnostic_id, diagnostic_version)] = group_id
            self.group_versions[group_id] = (diagnostic_id, diagnostic_version)
        # The latest execution of a group is the one the API serves, as `select_latest_execution` picks it.
        newest: dict[int, tuple[datetime.datetime, int]] = {}
        for execution_id, group_id, dataset_hash, created_at, successful in session.execute(
            select(
                Execution.id,
                Execution.execution_group_id,
                Execution.dataset_hash,
                Execution.created_at,
                Execution.successful,
            )
        ):
            self.executions[(group_id, dataset_hash)].append(execution_id)
            self.execution_groups[execution_id] = group_id
            if successful:
                self.successful.add(execution_id)
            order = (created_at, execution_id)
            newest[group_id] = max(newest.get(group_id, order), order)
        self.latest = {execution_id for _, execution_id in newest.values()}
        for dataset_id, slug in session.execute(select(Dataset.id, Dataset.slug)):
            self.dataset_ids[slug] = dataset_id
        for execution_id, dataset_id in session.execute(
            select(execution_datasets.c.execution_id, execution_datasets.c.dataset_id)
        ):
            self.links[execution_id].append(dataset_id)
        self.legacy_slugs = _legacy_slugs(db)

    def buckets(self, diagnostic_ids: set[int]) -> dict[str, set[int]]:
        """Split the executions of these diagnostics by whether they are the latest and succeeded."""
        buckets: dict[str, set[int]] = collections.defaultdict(set)
        for execution_id, group_id in self.execution_groups.items():
            if self.group_versions[group_id][0] not in diagnostic_ids:
                continue
            position = "latest" if execution_id in self.latest else "superseded"
            outcome = "successful" if execution_id in self.successful else "failed"
            buckets[f"{position} {outcome}"].add(execution_id)
        return buckets

    def legacy_hash(self, datasets: ExecutionDatasetCollection) -> str:
        """Compute the ``dataset_hash`` these datasets had before migration ``d6e7f8a9b0c1``."""
        parts = []
        for source_type, collection in datasets.items():
            values = sorted(
                self.legacy_slugs.get(str(slug), str(slug))
                for slug in collection.datasets[collection.slug_column]
            )
            digest = hashlib.sha1("\n".join(values).encode("utf-8")).hexdigest()  # noqa: S324
            parts.append(f"{source_type.value}:{digest}")
        return hashlib.sha1("\n".join(sorted(parts)).encode("utf-8")).hexdigest()  # noqa: S324

    def consider(self, potentials: Iterable[DiagnosticExecution], only: set[int]) -> None:
        """Plan rewrites for the executions in ``only`` that match ``potentials``."""
        for potential in potentials:
            self.counts["solved"] += 1
            diagnostic_id = self.diagnostic_ids.get((potential.provider.slug, potential.diagnostic.slug))
            if diagnostic_id is None:
                self.counts["solved, diagnostic not registered yet"] += 1
                continue
            group_id = self.group_ids.get(
                (potential.dataset_key, diagnostic_id, potential.diagnostic.version)
            )
            if group_id is None:
                self.counts["solved, no group yet"] += 1
                continue
            candidates = {
                *self.executions.get((group_id, potential.datasets.hash), []),
                *self.executions.get((group_id, self.legacy_hash(potential.datasets)), []),
            }
            matching = sorted(candidates & only)
            if not matching:
                self.counts["solved, no latest successful execution with this dataset hash"] += 1
                continue

            # One row per file, as `register_datasets` writes them.
            slugs = [
                str(slug)
                for _, collection in potential.datasets.items()
                for slug in collection.datasets[collection.slug_column]
            ]
            missing = sorted({slug for slug in slugs if slug not in self.dataset_ids})
            if missing:
                self.counts["skipped, slug not in dataset table"] += len(matching)
                logger.warning(f"{potential.execution_slug()}: unknown slugs {missing[:3]}")
                continue
            target = sorted(self.dataset_ids[slug] for slug in slugs)
            distinct = sorted(set(target))

            for execution_id in matching:
                self.matched.add(execution_id)
                current = self.links.get(execution_id, [])
                if sorted(set(current)) != distinct:
                    self.counts["wrong links"] += 1
                elif sorted(current) != target:
                    self.counts["wrong per-file counts"] += 1
                else:
                    self.counts["already correct"] += 1
                    continue
                self.rewrites[execution_id] = target
                if len(self.examples) < self.show:
                    self.examples.append(
                        f"execution {execution_id} ({potential.execution_slug()}): "
                        f"{sorted(set(current))} -> {distinct}"
                    )


_CMIP6_ID_COLUMNS = (
    "activity_id",
    "institution_id",
    "source_id",
    "experiment_id",
    "member_id",
    "table_id",
    "variable_id",
    "grid_label",
    "version",
)
_REFERENCE_ID_COLUMNS = (
    "activity_id",
    "activity_id",
    "institution_id",
    "source_id",
    "frequency",
    "variable_id",
    "nominal_resolution",
    "grid_label",
    "version",
)


def _legacy_slugs(db: Database) -> dict[str, str]:
    """Map each rewritten ``instance_id`` to the one it had before migration ``d6e7f8a9b0c1``."""
    sources = [("cmip6_dataset", ("CMIP6",), _CMIP6_ID_COLUMNS)] + [
        (name, (), _REFERENCE_ID_COLUMNS)
        for name in ("obs4mips_dataset", "obs4ref_dataset", "pmp_climatology_dataset")
    ]
    legacy = {}
    for name, prefix, columns in sources:
        unique = sorted(set(columns))
        sub = table(name, column("instance_id"), *(column(c) for c in unique))
        for row in db.session.execute(select(sub)).mappings():
            values = dict(row)
            if values.get("nominal_resolution"):
                values["nominal_resolution"] = values["nominal_resolution"].replace(" ", "")
            parts = [*prefix, *(values[c] for c in columns)]
            if None not in parts:
                legacy[values["instance_id"]] = ".".join(parts)
    return legacy


def _uses_fallback(diagnostic: Diagnostic) -> bool:
    requirements = [
        requirement
        for item in diagnostic.data_requirements
        for requirement in ((item,) if isinstance(item, DataRequirement) else item)
    ]
    return any(requirement.fallback_source_types for requirement in requirements)


def _affected_diagnostics(planner: Planner, solver: ExecutionSolver, filters: SolveFilterOptions) -> set[int]:
    """Find the ids of the registered diagnostics the filters select that declare a fallback source type."""
    return {
        planner.diagnostic_ids[(provider.slug, diagnostic.slug)]
        for provider in solver.provider_registry.providers
        for diagnostic in provider.diagnostics()
        if matches_filter(diagnostic, filters)
        and _uses_fallback(diagnostic)
        and (provider.slug, diagnostic.slug) in planner.diagnostic_ids
    }


@app.command(help=__doc__)
def main(
    apply: Annotated[bool, typer.Option(help="Rewrite the links instead of reporting.")] = False,
    provider: Annotated[
        list[str] | None, typer.Option(help="Only solve providers matching this slug.")
    ] = None,
    diagnostic: Annotated[
        list[str] | None, typer.Option(help="Only solve diagnostics matching this slug.")
    ] = None,
    show: Annotated[int, typer.Option(help="Number of example changes and ids to print.")] = 20,
) -> None:
    """Plan, and with ``--apply`` write, the corrected execution dataset links."""
    filters = SolveFilterOptions(provider=provider, diagnostic=diagnostic)

    config = Config.default()
    db = Database.from_config(config, run_migrations=False)

    planner = Planner(show=show)
    with db.session.begin():
        planner.load(db)
    logger.info(f"Loaded {len(planner.links)} executions with links and {len(planner.dataset_ids)} datasets")

    solver = ExecutionSolver.build_from_db(config, db)
    buckets = planner.buckets(_affected_diagnostics(planner, solver, filters))
    targets = buckets["latest successful"]
    planner.consider(solver.solve(filters), only=targets)

    # Executions linking a dataset outside their group's source types are wrong for certain.
    with db.session.begin():
        flagged = {row[0] for row in db.session.execute(text(_MISMATCHED_TYPES_SQL))}
    unrepaired = sorted((flagged & targets) - planner.rewrites.keys())
    unmatched = sorted(targets - planner.matched)
    slugs = {diagnostic_id: slug for (_, slug), diagnostic_id in planner.diagnostic_ids.items()}
    unmatched_versions = collections.Counter(
        (slugs[diagnostic_id], version)
        for diagnostic_id, version in (
            planner.group_versions[planner.execution_groups[execution_id]] for execution_id in unmatched
        )
    )

    for line in planner.examples:
        print(line)
    print()
    for name, value in planner.counts.items():
        print(f"{name}: {value}")
    print("executions of diagnostics that declare a fallback source type:")
    for name, ids in sorted(buckets.items()):
        print(f"  {name}: {len(ids)} e.g. {sorted(ids)[:show]}")
    print(f"latest successful executions matched: {len(planner.matched)}")
    print(f"executions to rewrite: {len(planner.rewrites)}")
    print(f"latest successful executions not matched: {len(unmatched)}, by diagnostic version:")
    for (slug, version), value in unmatched_versions.most_common():
        print(f"  {slug} v{version}: {value}")
    print(f"executions linking a dataset of the wrong source type: {len(flagged)}")
    print(f"of those, latest successful and not repaired: {len(unrepaired)} e.g. {unrepaired[:show]}")

    if not apply:
        print("Dry run, nothing written. Pass --apply to rewrite.")
        return

    pending = sorted(planner.rewrites.items())
    for start in range(0, len(pending), BATCH_SIZE):
        batch = pending[start : start + BATCH_SIZE]
        with db.session.begin():
            db.session.execute(
                delete(execution_datasets).where(
                    execution_datasets.c.execution_id.in_([execution_id for execution_id, _ in batch])
                )
            )
            db.session.execute(
                execution_datasets.insert(),
                [
                    {"execution_id": execution_id, "dataset_id": dataset_id}
                    for execution_id, dataset_ids in batch
                    for dataset_id in dataset_ids
                ],
            )
        logger.info(f"Rewrote {start + len(batch)}/{len(pending)} executions")


if __name__ == "__main__":
    app()
