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

With `--retract-orphans`, groups the current solve no longer produces have their latest execution retracted
when it is unmatched and its links may be wrong:
it links a dataset of the wrong source type, or its diagnostic declares a fallback source type.

Dry run by default. Pass `--apply` to rewrite the links.
Run it with the climate-ref version that solves the database,
because groups are keyed by diagnostic version, e.g.

    kubectl exec -i <pod running climate-ref> -- \
        python - --provider pmp < scripts/repair-execution-datasets.py

Solving the current catalog may finalise unfinalised datasets, which writes their metadata even in a dry run.
So every run holds the lock that `ref solve --lock` takes, and exits 75 when another process holds it.
Pass `--lock-file` when the database's writers use another lock file.
"""

import collections
import contextlib
import datetime
import fcntl
import hashlib
import os
from collections.abc import Generator, Iterable
from pathlib import Path
from typing import Annotated

import attrs
import pandas as pd
import typer
from loguru import logger
from sqlalchemy import column, delete, func, select, table, text, update

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
from climate_ref_core.diagnostics import DataRequirement, Diagnostic
from climate_ref_core.env import env

app = typer.Typer()

# Matches `ref solve --lock`, so a scheduled solve can tell a held lock from a failure.
LOCK_HELD_EXIT_CODE = 75

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
    produced: set[int] = attrs.field(factory=set)
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
            if only is None:
                self.produced.add(group_id)
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
            per_file = sorted(self.dataset_ids[slug] for slug in slugs)
            distinct = sorted(set(per_file))
            target = distinct if self.dedupe else per_file

            for execution_id in matching:
                self.matched.add(execution_id)
                current = self.links.get(execution_id, [])
                if sorted(set(current)) != distinct:
                    self.counts["wrong links"] += 1
                elif sorted(current) != target:
                    self.counts["duplicate links" if self.dedupe else "wrong per-file counts"] += 1
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


def _in_scope(
    planner: Planner, solver: ExecutionSolver, filters: SolveFilterOptions
) -> dict[int, Diagnostic]:
    """Map the id of each registered diagnostic the filters select to the diagnostic."""
    return {
        planner.diagnostic_ids[(provider.slug, diagnostic.slug)]: diagnostic
        for provider in solver.provider_registry.providers
        for diagnostic in provider.diagnostics()
        if matches_filter(diagnostic, filters) and (provider.slug, diagnostic.slug) in planner.diagnostic_ids
    }


def _uses_fallback(diagnostic: Diagnostic) -> bool:
    requirements = [
        requirement
        for item in diagnostic.data_requirements
        for requirement in ((item,) if isinstance(item, DataRequirement) else item)
    ]
    return any(requirement.fallback_source_types for requirement in requirements)


def _orphans(
    planner: Planner, solver: ExecutionSolver, db: Database, filters: SolveFilterOptions, flagged: set[int]
) -> list[int]:
    """Find the latest executions to retract in groups the current solve no longer produces."""
    in_scope = _in_scope(planner, solver, filters)
    with db.session.begin():
        latest = db.session.execute(
            select(
                ExecutionGroup.id,
                ExecutionGroup.diagnostic_id,
                ExecutionGroup.diagnostic_version,
                func.max(Execution.id),
            )
            .join(Execution, Execution.execution_group_id == ExecutionGroup.id)
            .group_by(ExecutionGroup.id)
        ).all()
    orphans = []
    for group_id, diagnostic_id, diagnostic_version, execution_id in latest:
        diagnostic = in_scope.get(diagnostic_id)
        # Groups of an older diagnostic version are never solved again, so they are not orphans.
        if diagnostic is None or diagnostic_version != diagnostic.version:
            continue
        if group_id in planner.produced or execution_id in planner.matched:
            continue
        if execution_id in flagged or _uses_fallback(diagnostic):
            orphans.append(execution_id)
    return sorted(orphans)


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
        diagnostic_id: diagnostic.slug
        for diagnostic_id, diagnostic in _in_scope(planner, solver, filters).items()
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


@contextlib.contextmanager
def _exclusive_lock(path: Path) -> Generator[None]:
    """Hold the POSIX lock `ref solve --lock` takes, or exit 75 when another process holds it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+") as handle:
        try:
            fcntl.lockf(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, PermissionError):
            handle.seek(0)
            logger.error(f"Another process holds {path} (PID {handle.read().strip() or 'unknown'})")
            raise typer.Exit(code=LOCK_HELD_EXIT_CODE)
        try:
            handle.truncate(0)
            handle.write(f"{os.getpid()}\n")
            handle.flush()
            yield
        finally:
            fcntl.lockf(handle, fcntl.LOCK_UN)


def _repair(  # noqa: PLR0913
    filters: SolveFilterOptions,
    *,
    apply: bool,
    as_of_run: bool,
    dedupe: bool,
    retract_orphans: bool,
    show: int,
) -> None:
    """Plan, and with ``apply`` write, the corrected execution dataset links."""
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
    orphans = _orphans(planner, solver, db, filters, flagged) if retract_orphans else []

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
    if retract_orphans:
        print(f"executions to retract in groups no longer solved: {len(orphans)} e.g. {orphans[:show]}")

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

    for start in range(0, len(orphans), BATCH_SIZE):
        with db.session.begin():
            db.session.execute(
                update(Execution)
                .where(Execution.id.in_(orphans[start : start + BATCH_SIZE]))
                .values(retracted=True)
            )
        logger.info(f"Retracted {min(start + BATCH_SIZE, len(orphans))}/{len(orphans)} executions")


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
    retract_orphans: Annotated[
        bool,
        typer.Option(
            help="Retract the latest execution of groups no longer solved when its links may be wrong."
        ),
    ] = False,
    show: Annotated[int, typer.Option(help="Number of example changes to print.")] = 20,
    lock_file: Annotated[
        Path | None,
        typer.Option(
            help="Lock file shared with the database's other writers. "
            "Defaults to the one `ref solve --lock` uses.",
            envvar="REF_LOCK_FILE",
        ),
    ] = None,
) -> None:
    """Plan, and with ``--apply`` write, the corrected execution dataset links."""
    filters = SolveFilterOptions(provider=provider, diagnostic=diagnostic)
    with _exclusive_lock(lock_file or env.path("REF_CONFIGURATION").resolve() / "ref.lock"):
        _repair(
            filters,
            apply=apply,
            as_of_run=as_of_run,
            dedupe=dedupe,
            retract_orphans=retract_orphans,
            show=show,
        )


if __name__ == "__main__":
    app()
