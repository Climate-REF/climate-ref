"""
Integration tests for reingest functionality
"""

import pytest

from climate_ref.executor.reingest import (
    ReingestMode,
    reconstruct_execution_definition,
    reingest_execution,
)
from climate_ref.models import ScalarMetricValue, SeriesMetricValue
from climate_ref.models.dataset import CMIP6Dataset
from climate_ref.models.diagnostic import Diagnostic as DiagnosticModel
from climate_ref.models.execution import Execution, ExecutionGroup, ExecutionOutput, execution_datasets
from climate_ref.models.metric_value import MetricValue
from climate_ref.provider_registry import ProviderRegistry
from climate_ref.solver import solve_required_executions
from climate_ref_core.datasets import SourceDatasetType


def test_definition_round_trip(config, db_seeded, provider):
    with db_seeded.session.begin():
        datasets = db_seeded.session.query(CMIP6Dataset).limit(2).all()
        assert len(datasets) >= 1

        selector = (("source_id", datasets[0].source_id),)

        diag = db_seeded.session.query(DiagnosticModel).first()
        eg = ExecutionGroup(
            key="round-trip-test",
            diagnostic_id=diag.id,
            selectors={SourceDatasetType.CMIP6.value: [list(pair) for pair in selector]},
        )
        db_seeded.session.add(eg)
        db_seeded.session.flush()

        ex = Execution(
            execution_group_id=eg.id,
            successful=True,
            output_fragment="test/round-trip/abc",
            dataset_hash="h1",
        )
        db_seeded.session.add(ex)
        db_seeded.session.flush()

        for dataset in datasets:
            db_seeded.session.execute(
                execution_datasets.insert().values(
                    execution_id=ex.id,
                    dataset_id=dataset.id,
                )
            )

    diagnostic = provider.get("mock")
    definition = reconstruct_execution_definition(config, ex, diagnostic)

    assert definition.key == "round-trip-test"
    assert definition.diagnostic is diagnostic
    assert SourceDatasetType.CMIP6 in definition.datasets

    collection = definition.datasets[SourceDatasetType.CMIP6]

    # Dataset IDs should match what was linked
    expected_ids = sorted(d.id for d in datasets)
    actual_ids = sorted(collection.datasets.index.unique())
    assert expected_ids == actual_ids, f"Dataset IDs: expected {expected_ids}, got {actual_ids}"

    # File paths should be present for all linked datasets
    expected_paths = sorted(f.path for d in datasets for f in d.files)
    actual_paths = sorted(collection.datasets["path"].tolist())
    assert expected_paths == actual_paths, "File paths not preserved through round-trip"

    # Key facets should survive the round-trip
    for facet in ("variable_id", "source_id", "experiment_id"):
        assert facet in collection.datasets.columns, f"Missing facet column: {facet}"

    # Selector should match what was stored on the execution group
    assert collection.selector == selector


def _snapshot_scalars(db, execution):
    """Snapshot scalar metrics as a set of (value, dimensions) for comparison."""
    values = db.session.query(ScalarMetricValue).filter_by(execution_id=execution.id).all()
    return {(v.value, tuple(sorted(v.dimensions.items()))) for v in values}


def _snapshot_series(db, execution):
    """Snapshot series metrics as a set of (values_tuple, dimensions) for comparison."""
    values = db.session.query(SeriesMetricValue).filter_by(execution_id=execution.id).all()
    return {(tuple(v.values), tuple(sorted(v.dimensions.items()))) for v in values}


def _snapshot_outputs(db, execution):
    """Snapshot outputs as a set of (short_name, output_type, filename) for comparison."""
    outputs = db.session.query(ExecutionOutput).filter_by(execution_id=execution.id).all()
    return {(o.short_name, o.output_type, o.filename) for o in outputs}


@pytest.fixture
def _solved_example(config, db_seeded):
    """Run the example provider's diagnostics via solve, producing real executions."""
    solve_required_executions(db=db_seeded, config=config, one_per_diagnostic=True)

    # Verify at least one successful execution was produced
    successful = db_seeded.session.query(Execution).filter_by(successful=True).all()
    assert len(successful) >= 1, "solve_required_executions should produce at least one successful execution"

    # Close any lingering transaction so test methods start clean
    if db_seeded.session.in_transaction():
        db_seeded.session.commit()


@pytest.fixture
def provider_registry(config, db_seeded):
    """Build a ProviderRegistry from the seeded database."""
    return ProviderRegistry.build_from_config(config, db_seeded)


@pytest.mark.usefixtures("_solved_example")
class TestReingestAfterSolve:
    """End-to-end: solve with example provider, then reingest in each mode."""

    def _get_successful_execution(self, db_seeded):
        """Get the first successful execution with metric values."""
        executions = db_seeded.session.query(Execution).filter_by(successful=True).all()
        for ex in executions:
            if db_seeded.session.query(MetricValue).filter_by(execution_id=ex.id).count() > 0:
                return ex
        pytest.skip("No successful execution with metric values found")

    def test_reingest_replace_preserves_metric_count(self, config, db_seeded, provider_registry):
        """Replace reingest should produce the same number of metrics as the original."""
        execution = self._get_successful_execution(db_seeded)

        original_scalars = _snapshot_scalars(db_seeded, execution)
        original_series = _snapshot_series(db_seeded, execution)
        assert original_scalars, "Execution should have scalar metric values"

        if db_seeded.session.in_transaction():
            db_seeded.session.commit()

        with db_seeded.session.begin():
            ok = reingest_execution(
                config=config,
                database=db_seeded,
                execution=execution,
                provider_registry=provider_registry,
                mode=ReingestMode.replace,
            )
        assert ok is True

        replaced_scalars = _snapshot_scalars(db_seeded, execution)
        replaced_series = _snapshot_series(db_seeded, execution)

        assert original_scalars == replaced_scalars, "Replace reingest should produce identical scalars"
        assert original_series == replaced_series, "Replace reingest should produce identical series"

    def test_reingest_additive_is_idempotent(self, config, db_seeded, provider_registry):
        """Running additive reingest twice should not create duplicate metrics."""
        execution = self._get_successful_execution(db_seeded)

        count_before = db_seeded.session.query(MetricValue).filter_by(execution_id=execution.id).count()

        if db_seeded.session.in_transaction():
            db_seeded.session.commit()

        with db_seeded.session.begin():
            ok = reingest_execution(
                config=config,
                database=db_seeded,
                execution=execution,
                provider_registry=provider_registry,
                mode=ReingestMode.additive,
            )
        assert ok is True

        count_after = db_seeded.session.query(MetricValue).filter_by(execution_id=execution.id).count()
        assert count_before == count_after, (
            f"Additive reingest created duplicates: {count_before} -> {count_after}"
        )

    def test_reingest_versioned_creates_equivalent_execution(self, config, db_seeded, provider_registry):
        """Versioned reingest should create a new execution with equivalent metrics and outputs."""
        execution = self._get_successful_execution(db_seeded)

        original_scalars = _snapshot_scalars(db_seeded, execution)
        original_series = _snapshot_series(db_seeded, execution)
        original_outputs = _snapshot_outputs(db_seeded, execution)

        execution_count_before = db_seeded.session.query(Execution).count()

        if db_seeded.session.in_transaction():
            db_seeded.session.commit()

        with db_seeded.session.begin():
            ok = reingest_execution(
                config=config,
                database=db_seeded,
                execution=execution,
                provider_registry=provider_registry,
                mode=ReingestMode.versioned,
            )
        assert ok is True

        # A new execution should exist
        execution_count_after = db_seeded.session.query(Execution).count()
        assert execution_count_after == execution_count_before + 1

        # Find the versioned execution
        eg = execution.execution_group
        new_execution = (
            db_seeded.session.query(Execution)
            .filter(
                Execution.execution_group_id == eg.id,
                Execution.id != execution.id,
                Execution.successful.is_(True),
            )
            .order_by(Execution.id.desc())
            .first()
        )
        assert new_execution is not None

        versioned_scalars = _snapshot_scalars(db_seeded, new_execution)
        versioned_series = _snapshot_series(db_seeded, new_execution)
        versioned_outputs = _snapshot_outputs(db_seeded, new_execution)

        assert original_scalars == versioned_scalars, "Versioned scalars should match original"
        assert original_series == versioned_series, "Versioned series should match original"
        assert original_outputs == versioned_outputs, "Versioned outputs should match original"

        # Original execution should be untouched
        assert _snapshot_scalars(db_seeded, execution) == original_scalars
