"""
Rebuild the dataset links of executions from a fresh solve.

From 2026-09-02 the obs4REF fallback merge dropped the catalog index,
so `execution_dataset` stored row positions instead of dataset ids
and an execution could link datasets it never used.

This solves against the current database, matches each solved execution to the existing executions
in its group that share its `dataset_hash`, and resolves the dataset ids from the `instance_id` of every row.
It never trusts the catalog index, so it also runs on an image without the fix.

With `--as-of-run`, executions left unmatched are solved again against the datasets available
when their solve run started: created by then, not yet retracted, and the latest version at that time.
Production changed its hash function around 2026-10-14, so older executions cannot match by hash.
Their stored links are checked instead: everything outside obs4MIPs must agree with the solve,
and the rest must be obs4MIPs ids or row positions, which is what the bug wrote.
The grey list and diagnostic code are today's, so an execution they have since changed stays unmatched.

Dry run by default. Pass `--apply` to rewrite the links. Run it with the deployment's environment, e.g.

    kubectl exec -i <pod running climate-ref> -- \
        python - --provider pmp < scripts/repair-execution-datasets.py

Solving the current catalog may finalise unfinalised datasets, which writes their metadata even in a dry run.
"""

import argparse
import collections
import datetime
import sys
from collections.abc import Iterable

import attrs
import pandas as pd
from loguru import logger
from sqlalchemy import delete, select, text

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
from climate_ref_core.datasets import SourceDatasetType

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
    hashes: dict[int, str] = attrs.field(factory=dict)
    by_group: dict[int, list[int]] = attrs.field(factory=lambda: collections.defaultdict(list))
    dataset_ids: dict[str, int] = attrs.field(factory=dict)
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
            self.hashes[execution_id] = dataset_hash
            self.by_group[group_id].append(execution_id)
        for dataset_id, slug in session.execute(select(Dataset.id, Dataset.slug)):
            self.dataset_ids[slug] = dataset_id
        for execution_id, dataset_id in session.execute(
            select(execution_datasets.c.execution_id, execution_datasets.c.dataset_id)
        ):
            self.links[execution_id].append(dataset_id)

    def consider(  # noqa: PLR0912
        self,
        potentials: Iterable[DiagnosticExecution],
        only: set[int] | None = None,
        position_limit: int | None = None,
    ) -> None:
        """
        Plan rewrites for the executions matching ``potentials``, limited to ``only`` when given.

        With ``position_limit``, an execution in the group whose hash differs is still accepted
        when its stored links agree with the solve outside obs4MIPs,
        and its other links are obs4MIPs ids or row positions below the limit.
        """
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
            if position_limit is None:
                candidates = self.executions.get((group_id, potential.datasets.hash), [])
            else:
                candidates = self.by_group.get(group_id, [])
            matching = [execution_id for execution_id in candidates if only is None or execution_id in only]
            if not matching:
                self.counts["solved, no execution with this dataset hash"] += 1
                continue

            slugs_by_type = {
                source_type: {str(slug) for slug in collection.datasets[collection.slug_column]}
                for source_type, collection in potential.datasets.items()
            }
            missing = sorted(
                slug for slugs in slugs_by_type.values() for slug in slugs if slug not in self.dataset_ids
            )
            if missing:
                self.counts["skipped, slug not in dataset table"] += len(matching)
                logger.warning(f"{potential.execution_slug()}: unknown slugs {missing[:3]}")
                continue
            ids_by_type = {
                source_type: {self.dataset_ids[slug] for slug in slugs}
                for source_type, slugs in slugs_by_type.items()
            }
            expected = sorted(set().union(*ids_by_type.values()))

            for execution_id in matching:
                if position_limit is not None:
                    if self.hashes[execution_id] == potential.datasets.hash:
                        self.counts["as-of hash match"] += 1
                    elif _consistent(self.links.get(execution_id, []), ids_by_type, position_limit):
                        self.counts["as-of relaxed match"] += 1
                    else:
                        self.counts["as-of relaxed match rejected"] += 1
                        continue
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


def _consistent(
    current: list[int], ids_by_type: dict[SourceDatasetType, set[int]], position_limit: int
) -> bool:
    """Whether stored links could have come from these datasets, given only obs4MIPs ids were corrupted."""
    obs = ids_by_type.get(SourceDatasetType.obs4MIPs, set())
    trusted = set().union(
        *(ids for source_type, ids in ids_by_type.items() if source_type != SourceDatasetType.obs4MIPs)
    )
    rest = set(current) - trusted
    return (
        trusted <= set(current)
        and bool(rest) == bool(obs)
        and all(dataset_id in obs or dataset_id < position_limit for dataset_id in rest)
    )


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
    planner: Planner, solver: ExecutionSolver, db: Database, args: argparse.Namespace
) -> None:
    in_scope = {
        planner.diagnostic_ids[(provider.slug, diagnostic.slug)]: diagnostic.slug
        for provider in solver.provider_registry.providers
        for diagnostic in provider.diagnostics()
        if matches_filter(diagnostic, SolveFilterOptions(provider=args.provider, diagnostic=args.diagnostic))
        and (provider.slug, diagnostic.slug) in planner.diagnostic_ids
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
        # A fallback merge renumbered rows from zero, so a corrupted id is below the obs4 row count.
        position_limit = sum(
            len(catalogs[source_type].to_frame())
            for source_type in (SourceDatasetType.obs4MIPs, SourceDatasetType.obs4REF)
            if source_type in catalogs
        )
        as_of = ExecutionSolver(provider_registry=solver.provider_registry, data_catalog=catalogs)
        planner.consider(
            as_of.solve(SolveFilterOptions(provider=args.provider, diagnostic=diagnostics)),
            only=set(unmatched),
            position_limit=position_limit,
        )


def main() -> int:
    """Plan, and with ``--apply`` write, the corrected execution dataset links."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="Rewrite the links instead of reporting.")
    parser.add_argument("--provider", action="append", help="Only solve providers matching this slug.")
    parser.add_argument("--diagnostic", action="append", help="Only solve diagnostics matching this slug.")
    parser.add_argument(
        "--as-of-run",
        action="store_true",
        help="Solve unmatched executions again against the datasets available when their run started.",
    )
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="Also rewrite executions whose links are right but repeated once per file.",
    )
    parser.add_argument("--show", type=int, default=20, help="Number of example changes to print.")
    args = parser.parse_args()

    config = Config.default()
    db = Database.from_config(config, run_migrations=False)

    planner = Planner(dedupe=args.dedupe, show=args.show)
    with db.session.begin():
        planner.load(db)
    logger.info(f"Loaded {len(planner.links)} executions with links and {len(planner.dataset_ids)} datasets")

    solver = ExecutionSolver.build_from_db(config, db)
    planner.consider(solver.solve(SolveFilterOptions(provider=args.provider, diagnostic=args.diagnostic)))
    current_matched = len(planner.matched)
    if args.as_of_run:
        _repair_as_of_run(planner, solver, db, args)

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
    if args.as_of_run:
        print(f"executions matched by an as-of-run solve: {len(planner.matched) - current_matched}")
    print(f"executions to rewrite: {len(planner.rewrites)}")
    print(f"linking a dataset of the wrong source type: {len(flagged)}")
    print(f"of those, not repaired by this run: {len(unrepaired)} e.g. {unrepaired[: args.show]}")

    if not args.apply:
        print("Dry run, nothing written. Pass --apply to rewrite.")
        return 0

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
    return 0


if __name__ == "__main__":
    sys.exit(main())
