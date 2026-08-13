"""
End-to-end plumbing of the resource columns on the ``Execution`` row.

The accuracy of the measurement itself is proven in
``climate-ref-core/tests/integration/test_resource_measurement.py``.
These tests only check that a measurement taken inside an executor
survives the trip through result handling into the database.
They assert only lower bounds, which hold under any load,
so they can run in parallel with other tests.

The diagnostics are defined at module level because the process pool has to pickle them.
"""

import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pytest

from climate_ref.executor import LocalExecutor, SynchronousExecutor
from climate_ref.models import Diagnostic as DiagnosticModel
from climate_ref.models import Execution, ExecutionGroup
from climate_ref.models import Provider as ProviderModel
from climate_ref.provider_registry import _register_provider
from climate_ref.results import Reader
from climate_ref_core.datasets import ExecutionDatasetCollection, SourceDatasetType
from climate_ref_core.diagnostics import (
    DataRequirement,
    Diagnostic,
    ExecutionDefinition,
    ExecutionResult,
)
from climate_ref_core.providers import DiagnosticProvider
from climate_ref_core.pycmec.metric import CMECMetric
from climate_ref_core.pycmec.output import CMECOutput

CPU_SECONDS = 0.1
"""CPU the diagnostic spends. Enough to read back as a nonzero count, cheap enough to run anywhere."""


def _metric_bundle() -> dict:
    return {
        "DIMENSIONS": {
            "json_structure": ["region", "metric", "statistic"],
            "region": {"global": {}},
            "metric": {"synthetic": {}},
            "statistic": {"value": {}},
        },
        "RESULTS": {"global": {"synthetic": {"value": 1.0}}},
    }


class SpendingDiagnostic(Diagnostic):
    """Burns a little CPU, then succeeds."""

    name = "spender"
    slug = "spender"
    data_requirements = (DataRequirement(source_type=SourceDatasetType.CMIP6, filters=(), group_by=None),)
    facets = ("region", "metric", "statistic")
    fails = False

    def execute(self, definition: ExecutionDefinition) -> None:
        """Spend a little CPU so the recorded numbers are nonzero rather than merely present."""
        start = time.process_time()
        x = 0.0
        while time.process_time() - start < CPU_SECONDS:
            for i in range(10000):
                x += i * 1.000001

        if self.fails:
            raise RuntimeError("deliberate failure after spending the budget")

    def build_execution_result(self, definition: ExecutionDefinition) -> ExecutionResult:
        """Write a minimal but valid pair of CMEC bundles."""
        return ExecutionResult.build_from_output_bundle(
            definition,
            cmec_output_bundle=CMECOutput.create_template(),
            cmec_metric_bundle=CMECMetric(**_metric_bundle()),
        )


class FailingSpendingDiagnostic(SpendingDiagnostic):
    """Spends the same budget and then fails, which is the run worth measuring."""

    name = "failing-spender"
    slug = "failing-spender"
    fails = True


@pytest.fixture
def spending_provider(config):
    provider = DiagnosticProvider("spending", "v0.1.0")
    provider.register(SpendingDiagnostic())
    provider.register(FailingSpendingDiagnostic())
    provider.configure(config)
    return provider


@pytest.fixture
def db_with_spending_provider(db, spending_provider):
    with db.session.begin():
        _register_provider(db, spending_provider)
    return db


def seed_execution(db, provider, diagnostic_slug: str, config, *, key: str) -> tuple[int, Path]:
    """Persist a pending execution and return its id and the scratch directory it writes to."""
    fragment = f"{provider.slug}/{diagnostic_slug}/{key}"
    scratch_dir = config.paths.scratch / fragment
    scratch_dir.mkdir(parents=True, exist_ok=True)

    with db.session.begin():
        diagnostic_row = (
            db.session.query(DiagnosticModel)
            .join(DiagnosticModel.provider)
            .filter(
                ProviderModel.slug == provider.slug,
                DiagnosticModel.slug == diagnostic_slug,
            )
            .one()
        )
        execution_group = ExecutionGroup(key=key, diagnostic_id=diagnostic_row.id, dirty=True, selectors={})
        db.session.add(execution_group)
        db.session.flush()
        execution = Execution(execution_group=execution_group, dataset_hash=key, output_fragment=fragment)
        db.session.add(execution)
        db.session.flush()
        execution_id = execution.id
        db.session.expunge(execution)

    return execution_id, scratch_dir


def build_definition(config, diagnostic, scratch_dir: Path, key: str) -> ExecutionDefinition:
    return ExecutionDefinition(
        diagnostic=diagnostic,
        key=key,
        datasets=ExecutionDatasetCollection({}),
        root_directory=config.paths.scratch,
        output_directory=scratch_dir,
    )


def run_through(executor, db, config, provider, diagnostic_slug: str, key: str) -> Execution:
    """Run one execution to completion and return the row it left behind."""
    execution_id, scratch_dir = seed_execution(db, provider, diagnostic_slug, config, key=key)
    definition = build_definition(config, provider.get(diagnostic_slug), scratch_dir, key)

    with db.session.begin():
        execution = db.session.get(Execution, execution_id)
        db.session.expunge(execution)

    executor.run(definition, execution)
    executor.join(timeout=120)

    with db.session.begin():
        return db.session.get(Execution, execution_id)


class TestSynchronousExecutor:
    """The in-process path, which is also the shape of a diagnostic that spawns nothing."""

    def test_a_successful_execution_records_what_it_spent(
        self, db_with_spending_provider, config, spending_provider
    ):
        executor = SynchronousExecutor(database=db_with_spending_provider, config=config)

        execution = run_through(
            executor,
            db_with_spending_provider,
            config,
            spending_provider,
            "spender",
            key="sync-ok",
        )

        assert execution.successful is True
        assert execution.wall_seconds >= CPU_SECONDS
        assert execution.cpu_seconds >= 0.5 * CPU_SECONDS
        assert execution.peak_memory_bytes > 0
        assert execution.memory_source in {"cgroup", "proc_tree", "rusage"}
        assert execution.resource_context["host"]

    def test_a_failed_execution_records_what_it_spent(
        self, db_with_spending_provider, config, spending_provider
    ):
        """A run that died is the strongest evidence there is about what a diagnostic needs."""
        executor = SynchronousExecutor(database=db_with_spending_provider, config=config)

        execution = run_through(
            executor,
            db_with_spending_provider,
            config,
            spending_provider,
            "failing-spender",
            key="sync-fail",
        )

        assert execution.successful is False
        assert execution.cpu_seconds >= 0.5 * CPU_SECONDS
        assert execution.peak_memory_bytes > 0


class TestLocalExecutor:
    """The process pool path, which is how the REF runs diagnostics by default."""

    @pytest.fixture
    def pool(self):
        pool = ProcessPoolExecutor(max_workers=1)
        yield pool
        pool.shutdown(wait=False)

    def test_a_worker_process_reports_back_what_it_spent(
        self, db_with_spending_provider, config, spending_provider, pool
    ):
        """The measurement is taken in the worker and has to survive the trip home."""
        executor = LocalExecutor(database=db_with_spending_provider, config=config, pool=pool)

        execution = run_through(
            executor,
            db_with_spending_provider,
            config,
            spending_provider,
            "spender",
            key="pool-ok",
        )

        assert execution.successful is True
        assert execution.cpu_seconds >= 0.5 * CPU_SECONDS
        assert execution.peak_memory_bytes > 0
        assert execution.queue_seconds is not None
        assert execution.queue_seconds >= 0.0


class TestAggregation:
    """
    A real run produces rows that ``ref executions resources`` can aggregate.

    The aggregation logic itself is unit tested with seeded rows in
    ``tests/unit/results/test_resources.py``.
    """

    def test_measurements_become_a_sizing_recommendation(
        self, db_with_spending_provider, config, spending_provider
    ):
        executor = SynchronousExecutor(database=db_with_spending_provider, config=config)
        for repeat in range(2):
            run_through(
                executor,
                db_with_spending_provider,
                config,
                spending_provider,
                "spender",
                key=f"agg-{repeat}",
            )

        profiles = Reader(db_with_spending_provider).resources.profiles(provider_contains=["spending"])

        assert len(profiles) == 1
        profile = profiles[0]
        assert profile.n_samples == 2
        assert profile.n_excluded == 0
        assert profile.peak_memory_p95 > 0
        assert profile.recommended_memory_bytes >= profile.peak_memory_p95
