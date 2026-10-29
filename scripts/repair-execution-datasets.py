"""
Rebuild the dataset links of executions from a fresh solve.

From 2026-09-02 the obs4REF fallback merge dropped the catalog index,
so `execution_dataset` stored row positions instead of dataset ids
and an execution could link datasets it never used.

This solves against the current database, matches each solved execution to the existing executions
in its group that share its `dataset_hash`, and resolves the dataset ids from the `instance_id` of every row.
It never trusts the catalog index, so it also runs on an image without the fix.
An execution whose datasets have changed since it ran no longer matches and is reported, not repaired.

Dry run by default. Pass `--apply` to rewrite the links. Run it with the deployment's environment, e.g.

    kubectl exec -i <pod running climate-ref> -- \
        python - --provider pmp < scripts/repair-execution-datasets.py

Solving may finalise unfinalised datasets, which writes their metadata even in a dry run.
"""

import argparse
import collections
import sys

from loguru import logger
from sqlalchemy import delete, select, text

from climate_ref.config import Config
from climate_ref.database import Database
from climate_ref.models import Dataset, Execution, ExecutionGroup
from climate_ref.models.diagnostic import Diagnostic as DiagnosticModel
from climate_ref.models.execution import execution_datasets
from climate_ref.models.provider import Provider as ProviderModel
from climate_ref.solver import ExecutionSolver, SolveFilterOptions

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


def _load_state(
    db: Database,
) -> tuple[
    dict[tuple[str, str], int],
    dict[tuple[str, int, int], int],
    dict[tuple[int, str], list[int]],
    dict[str, int],
]:
    session = db.session
    diagnostic_ids = {
        (provider_slug, diagnostic_slug): diagnostic_id
        for diagnostic_id, diagnostic_slug, provider_slug in session.execute(
            select(DiagnosticModel.id, DiagnosticModel.slug, ProviderModel.slug).join(
                DiagnosticModel.provider
            )
        )
    }
    group_ids = {
        (key, diagnostic_id, diagnostic_version): group_id
        for group_id, key, diagnostic_id, diagnostic_version in session.execute(
            select(
                ExecutionGroup.id,
                ExecutionGroup.key,
                ExecutionGroup.diagnostic_id,
                ExecutionGroup.diagnostic_version,
            )
        )
    }
    executions: dict[tuple[int, str], list[int]] = collections.defaultdict(list)
    for execution_id, group_id, dataset_hash in session.execute(
        select(Execution.id, Execution.execution_group_id, Execution.dataset_hash)
    ):
        executions[(group_id, dataset_hash)].append(execution_id)
    dataset_ids = {slug: dataset_id for dataset_id, slug in session.execute(select(Dataset.id, Dataset.slug))}
    return diagnostic_ids, group_ids, executions, dataset_ids


def _load_links(db: Database) -> dict[int, list[int]]:
    links: dict[int, list[int]] = collections.defaultdict(list)
    for execution_id, dataset_id in db.session.execute(
        select(execution_datasets.c.execution_id, execution_datasets.c.dataset_id)
    ):
        links[execution_id].append(dataset_id)
    return links


def main() -> int:  # noqa: PLR0912, PLR0915
    """Plan, and with ``--apply`` write, the corrected execution dataset links."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--apply", action="store_true", help="Rewrite the links instead of reporting.")
    parser.add_argument("--provider", action="append", help="Only solve providers matching this slug.")
    parser.add_argument("--diagnostic", action="append", help="Only solve diagnostics matching this slug.")
    parser.add_argument(
        "--dedupe",
        action="store_true",
        help="Also rewrite executions whose links are right but repeated once per file.",
    )
    parser.add_argument("--show", type=int, default=20, help="Number of example changes to print.")
    args = parser.parse_args()

    config = Config.default()
    db = Database.from_config(config, run_migrations=False)

    with db.session.begin():
        diagnostic_ids, group_ids, executions, dataset_ids = _load_state(db)
        links = _load_links(db)
    logger.info(f"Loaded {sum(map(len, executions.values()))} executions and {len(dataset_ids)} datasets")

    solver = ExecutionSolver.build_from_db(config, db)
    filters = SolveFilterOptions(provider=args.provider, diagnostic=args.diagnostic)

    rewrites: dict[int, list[int]] = {}
    counts: collections.Counter[str] = collections.Counter()
    examples: list[str] = []
    for potential in solver.solve(filters):
        counts["solved"] += 1
        diagnostic_id = diagnostic_ids[(potential.provider.slug, potential.diagnostic.slug)]
        group_id = group_ids.get((potential.dataset_key, diagnostic_id, potential.diagnostic.version))
        if group_id is None:
            counts["solved, no group yet"] += 1
            continue
        matching = executions.get((group_id, potential.datasets.hash), [])
        if not matching:
            counts["solved, no execution with this dataset hash"] += 1
            continue

        slugs = {
            str(slug)
            for _, collection in potential.datasets.items()
            for slug in collection.datasets[collection.slug_column]
        }
        missing = sorted(slug for slug in slugs if slug not in dataset_ids)
        if missing:
            counts["skipped, slug not in dataset table"] += len(matching)
            logger.warning(f"{potential.execution_slug()}: unknown slugs {missing[:3]}")
            continue
        expected = sorted(dataset_ids[slug] for slug in slugs)

        for execution_id in matching:
            current = links.get(execution_id, [])
            if sorted(set(current)) != expected:
                counts["wrong links"] += 1
            elif args.dedupe and len(current) != len(expected):
                counts["duplicate links"] += 1
            else:
                counts["already correct"] += 1
                continue
            rewrites[execution_id] = expected
            if len(examples) < args.show:
                examples.append(
                    f"execution {execution_id} ({potential.execution_slug()}): "
                    f"{sorted(set(current))} -> {expected}"
                )

    # Executions linking a dataset outside their group's source types are wrong for certain.
    with db.session.begin():
        flagged = {row[0] for row in db.session.execute(text(_MISMATCHED_TYPES_SQL))}
    unrepaired = sorted(flagged - rewrites.keys())

    for line in examples:
        print(line)
    print()
    for name, value in counts.items():
        print(f"{name}: {value}")
    print(f"executions to rewrite: {len(rewrites)}")
    print(f"linking a dataset of the wrong source type: {len(flagged)}")
    print(f"of those, not repaired by this run: {len(unrepaired)} e.g. {unrepaired[: args.show]}")

    if not args.apply:
        print("Dry run, nothing written. Pass --apply to rewrite.")
        return 0

    pending = sorted(rewrites.items())
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
                    for execution_id, dataset_ids_ in batch
                    for dataset_id in dataset_ids_
                ],
            )
        logger.info(f"Rewrote {start + len(batch)}/{len(pending)} executions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
