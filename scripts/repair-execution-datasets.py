"""
Rebuild the dataset links of executions from a fresh solve.

From 2026-09-02 the obs4REF fallback merge dropped the catalog index,
so `execution_dataset` stored row positions instead of dataset ids
and an execution could link datasets it never used.

This solves against the current database, matches each solved execution to the existing executions
in its group that share its `dataset_hash`, and resolves the dataset ids from the `instance_id` of every row.
It never trusts the catalog index, so it also runs on an image without the fix.

Migration `d6e7f8a9b0c1` rewrote obs4MIPs-style and multi-activity CMIP6 `instance_id`s
but not the hashes stored on executions, so a hash is also computed from the ids as they were before it.

With `--as-of-run`, executions left unmatched are solved again against the datasets available
when their solve run started: created by then, not yet retracted, and the latest version at that time.
The grey list, diagnostic code and dataset files are today's, so an execution they have since changed
stays unmatched.

Dry run by default. Pass `--apply` to rewrite the links. Run it with the deployment's environment, e.g.

    kubectl exec -i <pod running climate-ref> -- \
        python - --provider pmp < scripts/repair-execution-datasets.py

Solving the current catalog may finalise unfinalised datasets, which writes their metadata even in a dry run.
"""

import collections
import datetime
import hashlib
from collections.abc import Iterable
from typing import Annotated

import attrs
import pandas as pd
import typer
from loguru import logger
from sqlalchemy import column, delete, select, table, text

from climate_ref.config import Config
from climate_ref.data_catalog import DataCatalog
from climate_ref.database import Database
from climate_ref.models import Dataset, Execution, ExecutionGroup
from climate_ref.models.diagnostic import Diagnostic as DiagnosticModel
from climate_ref.models.execution import execution_datasets
from climate_ref.models.provider import Provider as ProviderModel
from climate_ref.query.datasets import select_datasets
from climate_ref.query.filters import DatasetFilter
from climate_ref.solver import DiagnosticExecution, ExecutionSolver, SolveFilterOptions, matches_filter
from climate_ref_core.datasets import ExecutionDatasetCollection, SourceDatasetType

app = typer.Typer()

BATCH_SIZE = 500

# Executions are created in bursts by each ingest-solve run, so a longer gap starts a new run.
RUN_GAP = datetime.timedelta(minutes=30)

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

    dedupe: bool
    show: int
    diagnostic_ids: dict[tuple[str, str], int] = attrs.field(factory=dict)
    group_ids: dict[tuple[str, int, int], int] = attrs.field(factory=dict)
    executions: dict[tuple[int, str], list[int]] = attrs.field(factory=lambda: collections.defaultdict(list))
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
        for execution_id, group_id, dataset_hash in session.execute(
            select(Execution.id, Execution.execution_group_id, Execution.dataset_hash)
        ):
            self.executions[(group_id, dataset_hash)].append(execution_id)
        for dataset_id, slug in session.execute(select(Dataset.id, Dataset.slug)):
            self.dataset_ids[slug] = dataset_id
        for execution_id, dataset_id in session.execute(
            select(execution_datasets.c.execution_id, execution_datasets.c.dataset_id)
        ):
            self.links[execution_id].append(dataset_id)
        self.legacy_slugs = _legacy_slugs(db)

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

    def consider(self, potentials: Iterable[DiagnosticExecution], only: set[int] | None = None) -> None:
        """Plan rewrites for the executions matching ``potentials``, limited to ``only`` when given."""
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
            matching = sorted(
                execution_id for execution_id in candidates if only is None or execution_id in only
            )
            if not matching:
                self.counts["solved, no execution with this dataset hash"] += 1
                continue

            slugs = {
                str(slug)
                for _, collection in potential.datasets.items()
                for slug in collection.datasets[collection.slug_column]
            }
            missing = sorted(slug for slug in slugs if slug not in self.dataset_ids)
            if missing:
                self.counts["skipped, slug not in dataset table"] += len(matching)
                logger.warning(f"{potential.execution_slug()}: unknown slugs {missing[:3]}")
                continue
            expected = sorted(self.dataset_ids[slug] for slug in slugs)

            for execution_id in matching:
                self.matched.add(execution_id)
                current = self.links.get(execution_id, [])
                if sorted(set(current)) != expected:
                    self.counts["wrong links"] += 1
                elif self.dedupe and len(current) != len(expected):
                    self.counts["duplicate links"] += 1
                else:
                    self.counts["already correct"] += 1
                    continue
                self.rewrites[execution_id] = expected
                if len(self.examples) < self.show:
                    self.examples.append(
                        f"execution {execution_id} ({potential.execution_slug()}): "
                        f"{sorted(set(current))} -> {expected}"
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


def _solve_runs(db: Database) -> list[tuple[datetime.datetime, list[tuple[int, int]]]]:
    """Group every execution into solve runs, as ``(run start, [(execution id, diagnostic id)])``."""
    rows = db.session.execute(
        select(Execution.created_at, Execution.id, ExecutionGroup.diagnostic_id)
        .join(ExecutionGroup, ExecutionGroup.id == Execution.execution_group_id)
        .order_by(Execution.created_at)
    ).all()
    runs: list[tuple[datetime.datetime, list[tuple[int, int]]]] = []
    previous = None
    for created_at, execution_id, diagnostic_id in rows:
        if previous is None or created_at - previous > RUN_GAP:
            runs.append((created_at, []))
        runs[-1][1].append((execution_id, diagnostic_id))
        previous = created_at
    return runs


def _all_versions(solver: ExecutionSolver, db: Database) -> dict[SourceDatasetType, pd.DataFrame]:
    """Load every version of every dataset, retracted ones included, keyed by dataset id."""
    frames = {}
    for source_type, catalog in solver.data_catalog.items():
        adapter = catalog.adapter
        if adapter is None:
            continue
        query = select_datasets(
            DatasetFilter(source_type=source_type, latest_only=False, include_retracted=True)
        )
        adapter._dataset_query = lambda query=query: query  # type: ignore[method-assign]
        frames[source_type] = adapter.load_catalog(db)
    return frames


def _repair_as_of_run(
    planner: Planner, solver: ExecutionSolver, db: Database, filters: SolveFilterOptions
) -> None:
    in_scope = {
        planner.diagnostic_ids[(provider.slug, diagnostic.slug)]: diagnostic.slug
        for provider in solver.provider_registry.providers
        for diagnostic in provider.diagnostics()
        if matches_filter(diagnostic, filters) and (provider.slug, diagnostic.slug) in planner.diagnostic_ids
    }
    with db.session.begin():
        runs = _solve_runs(db)
        availability = db.session.execute(select(Dataset.id, Dataset.created_at, Dataset.retracted_at)).all()
    frames = _all_versions(solver, db)
    created = pd.to_datetime(pd.Series({row.id: row.created_at for row in availability}))
    retracted = pd.to_datetime(pd.Series({row.id: row.retracted_at for row in availability}))

    for start, members in runs:
        unmatched = {
            execution_id: diagnostic_id
            for execution_id, diagnostic_id in members
            if diagnostic_id in in_scope and execution_id not in planner.matched
        }
        if not unmatched:
            continue
        available = created.index[(created <= start) & (retracted.isna() | (retracted > start))]
        catalogs = {}
        for source_type, frame in frames.items():
            adapter = solver.data_catalog[source_type].adapter
            assert adapter is not None
            catalogs[source_type] = DataCatalog.from_frame(
                adapter.filter_latest_versions(frame[frame.index.isin(available)])
            )
        diagnostics = sorted({in_scope[diagnostic_id] for diagnostic_id in unmatched.values()})
        logger.info(
            f"Run at {start}: {len(unmatched)} unmatched executions across {len(diagnostics)} diagnostics"
        )
        as_of = ExecutionSolver(provider_registry=solver.provider_registry, data_catalog=catalogs)
        planner.consider(
            as_of.solve(SolveFilterOptions(provider=filters.provider, diagnostic=diagnostics)),
            only=set(unmatched),
        )


@app.command(help=__doc__)
def main(  # noqa: PLR0913
    apply: Annotated[bool, typer.Option(help="Rewrite the links instead of reporting.")] = False,
    provider: Annotated[
        list[str] | None, typer.Option(help="Only solve providers matching this slug.")
    ] = None,
    diagnostic: Annotated[
        list[str] | None, typer.Option(help="Only solve diagnostics matching this slug.")
    ] = None,
    as_of_run: Annotated[
        bool,
        typer.Option(
            help="Solve unmatched executions again against the datasets available when their run started."
        ),
    ] = False,
    dedupe: Annotated[
        bool, typer.Option(help="Also rewrite executions whose links are right but repeated once per file.")
    ] = False,
    show: Annotated[int, typer.Option(help="Number of example changes to print.")] = 20,
) -> None:
    """Plan, and with ``--apply`` write, the corrected execution dataset links."""
    filters = SolveFilterOptions(provider=provider, diagnostic=diagnostic)

    config = Config.default()
    db = Database.from_config(config, run_migrations=False)

    planner = Planner(dedupe=dedupe, show=show)
    with db.session.begin():
        planner.load(db)
    logger.info(f"Loaded {len(planner.links)} executions with links and {len(planner.dataset_ids)} datasets")

    solver = ExecutionSolver.build_from_db(config, db)
    planner.consider(solver.solve(filters))
    current_matched = len(planner.matched)
    if as_of_run:
        _repair_as_of_run(planner, solver, db, filters)

    # Executions linking a dataset outside their group's source types are wrong for certain.
    with db.session.begin():
        flagged = {row[0] for row in db.session.execute(text(_MISMATCHED_TYPES_SQL))}
    unrepaired = sorted(flagged - planner.rewrites.keys())

    for line in planner.examples:
        print(line)
    print()
    for name, value in planner.counts.items():
        print(f"{name}: {value}")
    print(f"executions matched by the current solve: {current_matched}")
    if as_of_run:
        print(f"executions matched by an as-of-run solve: {len(planner.matched) - current_matched}")
    print(f"executions to rewrite: {len(planner.rewrites)}")
    print(f"linking a dataset of the wrong source type: {len(flagged)}")
    print(f"of those, not repaired by this run: {len(unrepaired)} e.g. {unrepaired[:show]}")

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
