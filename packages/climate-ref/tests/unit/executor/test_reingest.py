"""Tests for the reingest module."""

import json
import pathlib

import pytest
from climate_ref_esmvaltool import provider as esmvaltool_provider
from climate_ref_pmp import provider as pmp_provider

from climate_ref.executor.reingest import (
    ReingestMode,
    _copy_results_to_scratch,
    _delete_execution_results,
    _extract_dataset_attributes,
    _get_existing_metric_dimensions,
    _handle_reingest_output_bundle,
    _ingest_metrics,
    get_executions_for_reingest,
    reconstruct_execution_definition,
    reingest_execution,
)
from climate_ref.models import ScalarMetricValue
from climate_ref.models.dataset import CMIP6Dataset
from climate_ref.models.diagnostic import Diagnostic as DiagnosticModel
from climate_ref.models.execution import (
    Execution,
    ExecutionGroup,
    ExecutionOutput,
    ResultOutputType,
    execution_datasets,
)
from climate_ref.models.metric_value import MetricValue
from climate_ref.models.provider import Provider as ProviderModel
from climate_ref.provider_registry import ProviderRegistry, _register_provider
from climate_ref_core.datasets import SourceDatasetType
from climate_ref_core.diagnostics import ExecutionResult
from climate_ref_core.metric_values import SeriesMetricValue as TSeries
from climate_ref_core.pycmec.controlled_vocabulary import CV
from climate_ref_core.pycmec.metric import CMECMetric
from climate_ref_core.pycmec.output import CMECOutput


@pytest.fixture
def reingest_db(db, config):
    """Set up a database with an execution group, execution, and result files on disk."""
    with db.session.begin():
        provider_model = ProviderModel(name="mock_provider", slug="mock_provider", version="v0.1.0")
        db.session.add(provider_model)
        db.session.flush()

        diag_model = DiagnosticModel(
            name="mock",
            slug="mock",
            provider_id=provider_model.id,
        )
        db.session.add(diag_model)
        db.session.flush()

        eg = ExecutionGroup(
            key="test-key",
            diagnostic_id=diag_model.id,
            selectors={"cmip6": [["source_id", "ACCESS-ESM1-5"], ["variable_id", "tas"]]},
            dirty=False,
        )
        db.session.add(eg)
        db.session.flush()

        execution = Execution(
            execution_group_id=eg.id,
            successful=True,
            output_fragment="mock_provider/mock/abc123",
            dataset_hash="hash1",
        )
        db.session.add(execution)
        db.session.flush()

    return db


@pytest.fixture
def reingest_execution_obj(reingest_db):
    """Get the execution object from the reingest_db fixture."""
    return reingest_db.session.query(Execution).one()


@pytest.fixture
def mock_provider_registry(provider):
    """Create a mock ProviderRegistry that returns the test provider."""
    return ProviderRegistry(providers=[provider])


@pytest.fixture
def output_dir_with_results(config, reingest_execution_obj):
    """Create a results directory with empty CMEC template files."""
    results_dir = config.paths.results / reingest_execution_obj.output_fragment
    results_dir.mkdir(parents=True, exist_ok=True)

    CMECMetric(**CMECMetric.create_template()).dump_to_json(results_dir / "diagnostic.json")
    CMECOutput(**CMECOutput.create_template()).dump_to_json(results_dir / "output.json")

    series_data = [
        TSeries(
            dimensions={"source_id": "test-model"},
            values=[1.0, 2.0, 3.0],
            index=[0, 1, 2],
            index_name="time",
            attributes={"units": "K"},
        )
    ]
    TSeries.dump_to_json(results_dir / "series.json", series_data)

    return results_dir


@pytest.fixture
def output_dir_with_data(config, reingest_execution_obj):
    """Create a results directory with CMEC files containing actual metric/output data."""
    results_dir = config.paths.results / reingest_execution_obj.output_fragment
    results_dir.mkdir(parents=True, exist_ok=True)

    # 2-level nesting required by MetricResults schema
    (results_dir / "diagnostic.json").write_text(
        json.dumps(
            {
                "DIMENSIONS": {
                    "json_structure": ["source_id", "metric"],
                    "source_id": {"test-model": {}},
                    "metric": {"rmse": {}},
                },
                "RESULTS": {"test-model": {"rmse": 42.0}},
            }
        )
    )

    (results_dir / "plot.png").write_bytes(b"fake png")
    (results_dir / "output.json").write_text(
        json.dumps(
            {
                "index": "index.html",
                "provenance": {"environment": {}, "modeldata": [], "obsdata": {}, "log": "cmec_output.log"},
                "data": {},
                "plots": {"test_plot": {"filename": "plot.png", "long_name": "Test Plot", "description": ""}},
                "html": {},
                "metrics": None,
                "diagnostics": {},
            }
        )
    )

    TSeries.dump_to_json(
        results_dir / "series.json",
        [
            TSeries(
                dimensions={"source_id": "test-model"},
                values=[1.0, 2.0, 3.0],
                index=[0, 1, 2],
                index_name="time",
                attributes={"units": "K"},
            )
        ],
    )

    return results_dir


class TestReingestMode:
    def test_enum_values(self):
        assert ReingestMode.additive.value == "additive"
        assert ReingestMode.replace.value == "replace"
        assert ReingestMode.versioned.value == "versioned"

    def test_from_string(self):
        assert ReingestMode("additive") == ReingestMode.additive
        assert ReingestMode("replace") == ReingestMode.replace
        assert ReingestMode("versioned") == ReingestMode.versioned


class TestExtractDatasetAttributes:
    def test_extracts_cmip6_attributes(self, db_seeded):
        dataset = db_seeded.session.query(CMIP6Dataset).first()
        attrs = _extract_dataset_attributes(dataset)

        assert "variable_id" in attrs
        assert "source_id" in attrs
        assert "instance_id" in attrs
        assert "id" not in attrs
        assert "dataset_type" not in attrs


class TestReconstructExecutionDefinition:
    def test_reconstruct_basic(self, config, reingest_db, reingest_execution_obj, provider):
        """Test that a basic execution definition can be reconstructed."""
        diagnostic = provider.get("mock")
        definition = reconstruct_execution_definition(config, reingest_execution_obj, diagnostic)

        assert definition.key == "test-key"
        assert definition.diagnostic is diagnostic
        # Definition points at scratch (not results) so build_execution_result
        # writes to a safe location.
        assert definition.output_directory == config.paths.scratch / "mock_provider/mock/abc123"

    def test_reconstruct_output_directory_under_scratch(
        self, config, reingest_db, reingest_execution_obj, provider
    ):
        """Output directory should be under scratch for safe re-extraction."""
        diagnostic = provider.get("mock")
        definition = reconstruct_execution_definition(config, reingest_execution_obj, diagnostic)

        assert str(definition.output_directory).startswith(str(config.paths.scratch))


class TestDeleteExecutionResults:
    def test_deletes_metric_values_and_outputs(self, reingest_db, reingest_execution_obj):
        """Deleting results should remove all metric values and outputs."""
        execution = reingest_execution_obj

        # Add some metric values and outputs
        reingest_db.session.add(
            ScalarMetricValue(
                execution_id=execution.id,
                value=42.0,
                attributes={"test": True},
            )
        )
        reingest_db.session.add(
            ExecutionOutput(
                execution_id=execution.id,
                output_type=ResultOutputType.Plot,
                filename="test.png",
                short_name="test",
            )
        )
        reingest_db.session.commit()

        assert reingest_db.session.query(MetricValue).filter_by(execution_id=execution.id).count() == 1
        assert reingest_db.session.query(ExecutionOutput).filter_by(execution_id=execution.id).count() == 1

        _delete_execution_results(reingest_db, execution)
        reingest_db.session.commit()

        assert reingest_db.session.query(MetricValue).filter_by(execution_id=execution.id).count() == 0
        assert reingest_db.session.query(ExecutionOutput).filter_by(execution_id=execution.id).count() == 0


class TestReingestExecution:
    def test_reingest_missing_output_dir(
        self, config, reingest_db, reingest_execution_obj, mock_provider_registry
    ):
        """Should return False when output directory doesn't exist."""
        result = reingest_execution(
            config=config,
            database=reingest_db,
            execution=reingest_execution_obj,
            provider_registry=mock_provider_registry,
            mode=ReingestMode.replace,
        )
        assert result is False

    def test_reingest_unresolvable_diagnostic(self, config, reingest_db, reingest_execution_obj):
        """Should return False when provider registry can't resolve diagnostic."""
        empty_registry = ProviderRegistry(providers=[])
        result = reingest_execution(
            config=config,
            database=reingest_db,
            execution=reingest_execution_obj,
            provider_registry=empty_registry,
            mode=ReingestMode.replace,
        )
        assert result is False

    @pytest.mark.filterwarnings("ignore:Unknown dimension values.*:UserWarning")
    def test_reingest_replace_mode(
        self,
        config,
        reingest_db,
        reingest_execution_obj,
        mock_provider_registry,
        output_dir_with_results,
        mocker,
    ):
        """Replace mode should delete existing values and re-ingest."""
        execution = reingest_execution_obj

        # Pre-populate some metric values
        reingest_db.session.add(
            ScalarMetricValue(
                execution_id=execution.id,
                value=99.0,
                attributes={"old": True},
            )
        )
        reingest_db.session.commit()

        old_count = reingest_db.session.query(ScalarMetricValue).filter_by(execution_id=execution.id).count()
        assert old_count == 1

        # Mock build_execution_result to return a controlled result
        mock_diagnostic = mock_provider_registry.get_metric("mock_provider", "mock")
        mock_result = mocker.Mock(spec=ExecutionResult)
        mock_result.successful = True
        mock_result.metric_bundle_filename = pathlib.Path("diagnostic.json")
        mock_result.output_bundle_filename = pathlib.Path("output.json")
        mock_result.series_filename = pathlib.Path("series.json")
        mock_result.retryable = False
        mock_result.to_output_path = lambda f: output_dir_with_results / f if f else output_dir_with_results
        mock_result.as_relative_path = pathlib.Path
        mocker.patch.object(mock_diagnostic, "build_execution_result", return_value=mock_result)

        ok = reingest_execution(
            config=config,
            database=reingest_db,
            execution=execution,
            provider_registry=mock_provider_registry,
            mode=ReingestMode.replace,
        )
        reingest_db.session.commit()

        assert ok is True
        # Old value (99.0) should be gone
        old_vals = (
            reingest_db.session.query(ScalarMetricValue)
            .filter_by(execution_id=execution.id)
            .filter(ScalarMetricValue.value == 99.0)
            .all()
        )
        assert len(old_vals) == 0

    @pytest.mark.filterwarnings("ignore:Unknown dimension values.*:UserWarning")
    def test_reingest_versioned_creates_new_execution(
        self,
        config,
        reingest_db,
        reingest_execution_obj,
        mock_provider_registry,
        output_dir_with_results,
        mocker,
    ):
        """Versioned mode should create a new Execution record."""
        original_id = reingest_execution_obj.id
        original_count = reingest_db.session.query(Execution).count()

        mock_diagnostic = mock_provider_registry.get_metric("mock_provider", "mock")
        mock_result = mocker.Mock(spec=ExecutionResult)
        mock_result.successful = True
        mock_result.metric_bundle_filename = pathlib.Path("diagnostic.json")
        mock_result.output_bundle_filename = pathlib.Path("output.json")
        mock_result.series_filename = pathlib.Path("series.json")
        mock_result.retryable = False
        mock_result.to_output_path = lambda f: output_dir_with_results / f if f else output_dir_with_results
        mock_result.as_relative_path = pathlib.Path
        mocker.patch.object(mock_diagnostic, "build_execution_result", return_value=mock_result)

        ok = reingest_execution(
            config=config,
            database=reingest_db,
            execution=reingest_execution_obj,
            provider_registry=mock_provider_registry,
            mode=ReingestMode.versioned,
        )
        reingest_db.session.commit()

        assert ok is True
        new_count = reingest_db.session.query(Execution).count()
        assert new_count == original_count + 1

        # Original execution should be untouched
        original = reingest_db.session.get(Execution, original_id)
        assert original is not None

    def test_reingest_build_execution_result_failure(
        self,
        config,
        reingest_db,
        reingest_execution_obj,
        mock_provider_registry,
        output_dir_with_results,
        mocker,
    ):
        """Should return False and skip when build_execution_result raises."""
        mock_diagnostic = mock_provider_registry.get_metric("mock_provider", "mock")
        mocker.patch.object(
            mock_diagnostic,
            "build_execution_result",
            side_effect=RuntimeError("Extraction failed"),
        )

        result = reingest_execution(
            config=config,
            database=reingest_db,
            execution=reingest_execution_obj,
            provider_registry=mock_provider_registry,
            mode=ReingestMode.replace,
        )
        assert result is False

    @pytest.mark.filterwarnings("ignore:Unknown dimension values.*:UserWarning")
    def test_reingest_does_not_touch_dirty_flag(
        self,
        config,
        reingest_db,
        reingest_execution_obj,
        mock_provider_registry,
        output_dir_with_results,
        mocker,
    ):
        """The dirty flag should remain unchanged after reingest.

        Note: reingest_execution itself doesn't manage the dirty flag --
        that's the CLI's responsibility. This test verifies that reingest_execution
        does not set dirty=False (unlike handle_execution_result).
        """
        eg = reingest_execution_obj.execution_group
        eg.dirty = True
        reingest_db.session.commit()

        mock_diagnostic = mock_provider_registry.get_metric("mock_provider", "mock")
        mock_result = mocker.Mock(spec=ExecutionResult)
        mock_result.successful = True
        mock_result.metric_bundle_filename = pathlib.Path("diagnostic.json")
        mock_result.output_bundle_filename = None
        mock_result.series_filename = None
        mock_result.retryable = False
        mock_result.to_output_path = lambda f: output_dir_with_results / f if f else output_dir_with_results
        mock_result.as_relative_path = pathlib.Path
        mocker.patch.object(mock_diagnostic, "build_execution_result", return_value=mock_result)

        reingest_execution(
            config=config,
            database=reingest_db,
            execution=reingest_execution_obj,
            provider_registry=mock_provider_registry,
            mode=ReingestMode.replace,
        )
        reingest_db.session.commit()

        reingest_db.session.refresh(eg)
        assert eg.dirty is True

    @pytest.mark.filterwarnings("ignore:Unknown dimension values.*:UserWarning")
    def test_additive_preserves_existing_values(
        self,
        config,
        reingest_db,
        reingest_execution_obj,
        mock_provider_registry,
        output_dir_with_results,
        mocker,
    ):
        """Additive mode should keep pre-existing metric values untouched."""
        execution = reingest_execution_obj

        # Pre-populate with a value that has a unique dimension signature
        reingest_db.session.add(
            ScalarMetricValue(
                execution_id=execution.id,
                value=42.0,
                attributes={"pre_existing": True},
                source_id="pre-existing-model",
            )
        )
        reingest_db.session.commit()

        mock_diagnostic = mock_provider_registry.get_metric("mock_provider", "mock")
        mock_result = mocker.Mock(spec=ExecutionResult)
        mock_result.successful = True
        mock_result.metric_bundle_filename = pathlib.Path("diagnostic.json")
        mock_result.output_bundle_filename = None
        mock_result.series_filename = None
        mock_result.retryable = False
        mock_result.to_output_path = lambda f: output_dir_with_results / f if f else output_dir_with_results
        mock_result.as_relative_path = pathlib.Path
        mocker.patch.object(mock_diagnostic, "build_execution_result", return_value=mock_result)

        reingest_execution(
            config=config,
            database=reingest_db,
            execution=execution,
            provider_registry=mock_provider_registry,
            mode=ReingestMode.additive,
        )
        reingest_db.session.commit()

        # The pre-existing value should still be there
        pre_existing = (
            reingest_db.session.query(ScalarMetricValue)
            .filter_by(execution_id=execution.id)
            .filter(ScalarMetricValue.value == 42.0)
            .all()
        )
        assert len(pre_existing) == 1, "Additive mode should preserve pre-existing values"

    @pytest.mark.filterwarnings("ignore:Unknown dimension values.*:UserWarning")
    def test_additive_reingest_is_idempotent(
        self,
        config,
        reingest_db,
        reingest_execution_obj,
        mock_provider_registry,
        output_dir_with_results,
        mocker,
    ):
        """Running additive reingest twice should not create duplicate rows."""
        execution = reingest_execution_obj

        mock_diagnostic = mock_provider_registry.get_metric("mock_provider", "mock")
        mock_result = mocker.Mock(spec=ExecutionResult)
        mock_result.successful = True
        mock_result.metric_bundle_filename = pathlib.Path("diagnostic.json")
        mock_result.output_bundle_filename = pathlib.Path("output.json")
        mock_result.series_filename = pathlib.Path("series.json")
        mock_result.retryable = False
        mock_result.to_output_path = lambda f: output_dir_with_results / f if f else output_dir_with_results
        mock_result.as_relative_path = pathlib.Path
        mocker.patch.object(mock_diagnostic, "build_execution_result", return_value=mock_result)

        # First reingest
        reingest_execution(
            config=config,
            database=reingest_db,
            execution=execution,
            provider_registry=mock_provider_registry,
            mode=ReingestMode.additive,
        )
        reingest_db.session.commit()

        count_after_first = (
            reingest_db.session.query(ScalarMetricValue).filter_by(execution_id=execution.id).count()
        )
        series_count_first = (
            reingest_db.session.query(MetricValue).filter_by(execution_id=execution.id).count()
        )

        # Second reingest (should be idempotent)
        reingest_execution(
            config=config,
            database=reingest_db,
            execution=execution,
            provider_registry=mock_provider_registry,
            mode=ReingestMode.additive,
        )
        reingest_db.session.commit()

        count_after_second = (
            reingest_db.session.query(ScalarMetricValue).filter_by(execution_id=execution.id).count()
        )
        series_count_second = (
            reingest_db.session.query(MetricValue).filter_by(execution_id=execution.id).count()
        )

        assert count_after_first == count_after_second, (
            f"Additive reingest created duplicates: {count_after_first} -> {count_after_second}"
        )
        assert series_count_first == series_count_second, (
            f"Additive reingest created duplicate series: {series_count_first} -> {series_count_second}"
        )

    @pytest.mark.filterwarnings("ignore:Unknown dimension values.*:UserWarning")
    def test_replace_preserves_data_on_ingestion_failure(
        self,
        config,
        reingest_db,
        reingest_execution_obj,
        mock_provider_registry,
        output_dir_with_results,
        mocker,
    ):
        """If ingestion fails in replace mode, original data should be preserved."""
        execution = reingest_execution_obj

        # Pre-populate with known values
        reingest_db.session.add(
            ScalarMetricValue(
                execution_id=execution.id,
                value=42.0,
                attributes={"original": True},
            )
        )
        reingest_db.session.commit()

        original_count = (
            reingest_db.session.query(ScalarMetricValue).filter_by(execution_id=execution.id).count()
        )
        assert original_count == 1

        # Mock build_execution_result to return a result that will fail during scalar ingestion
        mock_diagnostic = mock_provider_registry.get_metric("mock_provider", "mock")
        mock_result = mocker.Mock(spec=ExecutionResult)
        mock_result.successful = True
        mock_result.metric_bundle_filename = pathlib.Path("diagnostic.json")
        mock_result.output_bundle_filename = None
        mock_result.series_filename = None
        mock_result.retryable = False
        mock_result.to_output_path = lambda f: output_dir_with_results / f if f else output_dir_with_results
        mock_result.as_relative_path = pathlib.Path
        mocker.patch.object(mock_diagnostic, "build_execution_result", return_value=mock_result)

        # Make the scalar ingestion fail by corrupting the metric bundle
        (output_dir_with_results / "diagnostic.json").write_text("not valid json")

        ok = reingest_execution(
            config=config,
            database=reingest_db,
            execution=execution,
            provider_registry=mock_provider_registry,
            mode=ReingestMode.replace,
        )
        reingest_db.session.commit()

        assert ok is False

        # Original data should be preserved since the savepoint rolled back
        preserved_count = (
            reingest_db.session.query(ScalarMetricValue).filter_by(execution_id=execution.id).count()
        )
        assert preserved_count == original_count, (
            f"Replace mode lost data on failure: {original_count} -> {preserved_count}"
        )

    @pytest.mark.filterwarnings("ignore:Unknown dimension values.*:UserWarning")
    def test_versioned_reingest_twice_creates_unique_fragments(
        self,
        config,
        reingest_db,
        reingest_execution_obj,
        mock_provider_registry,
        output_dir_with_results,
        mocker,
    ):
        """Running versioned reingest twice should create distinct output fragments."""
        mock_diagnostic = mock_provider_registry.get_metric("mock_provider", "mock")
        mock_result = mocker.Mock(spec=ExecutionResult)
        mock_result.successful = True
        mock_result.metric_bundle_filename = pathlib.Path("diagnostic.json")
        mock_result.output_bundle_filename = pathlib.Path("output.json")
        mock_result.series_filename = pathlib.Path("series.json")
        mock_result.retryable = False
        mock_result.to_output_path = lambda f: output_dir_with_results / f if f else output_dir_with_results
        mock_result.as_relative_path = pathlib.Path
        mocker.patch.object(mock_diagnostic, "build_execution_result", return_value=mock_result)

        # First versioned reingest
        ok1 = reingest_execution(
            config=config,
            database=reingest_db,
            execution=reingest_execution_obj,
            provider_registry=mock_provider_registry,
            mode=ReingestMode.versioned,
        )
        reingest_db.session.commit()
        assert ok1 is True

        # Second versioned reingest
        reingest_db.session.refresh(reingest_execution_obj)
        ok2 = reingest_execution(
            config=config,
            database=reingest_db,
            execution=reingest_execution_obj,
            provider_registry=mock_provider_registry,
            mode=ReingestMode.versioned,
        )
        reingest_db.session.commit()
        assert ok2 is True

        # Should have 3 executions total: original + 2 versioned
        all_executions = reingest_db.session.query(Execution).all()
        assert len(all_executions) == 3

        fragments = [e.output_fragment for e in all_executions]
        assert len(set(fragments)) == 3, f"Expected unique fragments, got: {fragments}"

    @pytest.mark.filterwarnings("ignore:Unknown dimension values.*:UserWarning")
    def test_reingest_marks_failed_execution_as_successful(
        self,
        config,
        reingest_db,
        reingest_execution_obj,
        mock_provider_registry,
        output_dir_with_results,
        mocker,
    ):
        """Reingest should mark a previously-failed execution as successful."""
        reingest_execution_obj.successful = False
        reingest_db.session.commit()
        assert reingest_execution_obj.successful is False

        mock_diagnostic = mock_provider_registry.get_metric("mock_provider", "mock")
        mock_result = mocker.Mock(spec=ExecutionResult)
        mock_result.successful = True
        mock_result.metric_bundle_filename = pathlib.Path("diagnostic.json")
        mock_result.output_bundle_filename = None
        mock_result.series_filename = None
        mock_result.retryable = False
        mock_result.to_output_path = lambda f: output_dir_with_results / f if f else output_dir_with_results
        mock_result.as_relative_path = pathlib.Path
        mocker.patch.object(mock_diagnostic, "build_execution_result", return_value=mock_result)

        ok = reingest_execution(
            config=config,
            database=reingest_db,
            execution=reingest_execution_obj,
            provider_registry=mock_provider_registry,
            mode=ReingestMode.replace,
        )
        reingest_db.session.commit()

        assert ok is True
        reingest_db.session.refresh(reingest_execution_obj)
        assert reingest_execution_obj.successful is True

    def test_scratch_directory_cleaned_up_on_success(
        self,
        config,
        reingest_db,
        reingest_execution_obj,
        mock_provider_registry,
        output_dir_with_results,
        mocker,
    ):
        """Scratch directory should be removed after successful reingest."""
        mock_diagnostic = mock_provider_registry.get_metric("mock_provider", "mock")
        mock_result = mocker.Mock(spec=ExecutionResult)
        mock_result.successful = True
        mock_result.metric_bundle_filename = pathlib.Path("diagnostic.json")
        mock_result.output_bundle_filename = None
        mock_result.series_filename = None
        mock_result.retryable = False
        mock_result.to_output_path = lambda f: output_dir_with_results / f if f else output_dir_with_results
        mock_result.as_relative_path = pathlib.Path
        mocker.patch.object(mock_diagnostic, "build_execution_result", return_value=mock_result)

        scratch_dir = config.paths.scratch / reingest_execution_obj.output_fragment

        reingest_execution(
            config=config,
            database=reingest_db,
            execution=reingest_execution_obj,
            provider_registry=mock_provider_registry,
            mode=ReingestMode.additive,
        )

        assert not scratch_dir.exists(), f"Scratch directory was not cleaned up: {scratch_dir}"

    def test_scratch_directory_cleaned_up_on_failure(
        self,
        config,
        reingest_db,
        reingest_execution_obj,
        mock_provider_registry,
        output_dir_with_results,
        mocker,
    ):
        """Scratch directory should be removed even when reingest fails."""
        mock_diagnostic = mock_provider_registry.get_metric("mock_provider", "mock")
        mocker.patch.object(
            mock_diagnostic,
            "build_execution_result",
            side_effect=RuntimeError("Extraction failed"),
        )

        scratch_dir = config.paths.scratch / reingest_execution_obj.output_fragment

        result = reingest_execution(
            config=config,
            database=reingest_db,
            execution=reingest_execution_obj,
            provider_registry=mock_provider_registry,
            mode=ReingestMode.replace,
        )

        assert result is False
        assert not scratch_dir.exists(), f"Scratch directory was not cleaned up on failure: {scratch_dir}"


class TestCopyResultsToScratch:
    def test_path_traversal_raises(self, config):
        """Should reject output fragments that escape the base directory."""
        with pytest.raises(ValueError, match="Unsafe source path"):
            _copy_results_to_scratch(config, "/etc/passwd")

    def test_copies_directory(self, config, reingest_execution_obj, output_dir_with_results):
        """Should copy the results directory to scratch."""
        scratch = _copy_results_to_scratch(config, reingest_execution_obj.output_fragment)
        assert scratch.exists()
        assert (scratch / "diagnostic.json").exists()
        assert (scratch / "output.json").exists()


class TestGetExistingMetricDimensions:
    def test_returns_empty_for_no_values(self, reingest_db, reingest_execution_obj):
        """Should return empty set when no metric values exist."""
        result = _get_existing_metric_dimensions(reingest_db, reingest_execution_obj)
        assert result == set()

    def test_returns_dimension_signatures(self, reingest_db, reingest_execution_obj):
        """Should return sorted dimension tuples for existing metric values."""
        reingest_db.session.add(
            ScalarMetricValue(
                execution_id=reingest_execution_obj.id,
                value=1.0,
                attributes={},
                source_id="model-a",
            )
        )
        reingest_db.session.commit()

        result = _get_existing_metric_dimensions(reingest_db, reingest_execution_obj)
        assert len(result) == 1
        sig = next(iter(result))
        # Should contain source_id dimension
        assert any(k == "source_id" and v == "model-a" for k, v in sig)


class TestHandleReingestOutputBundle:
    @pytest.mark.filterwarnings("ignore:Unknown dimension values.*:UserWarning")
    def test_registers_outputs_in_db(self, config, reingest_db, reingest_execution_obj, output_dir_with_data):
        """Should register output entries from the bundle into the database."""
        bundle_path = output_dir_with_data / "output.json"

        _handle_reingest_output_bundle(config, reingest_db, reingest_execution_obj, bundle_path)
        reingest_db.session.commit()

        outputs = (
            reingest_db.session.query(ExecutionOutput).filter_by(execution_id=reingest_execution_obj.id).all()
        )
        assert len(outputs) >= 1
        assert any(o.short_name == "test_plot" for o in outputs)


class TestIngestMetrics:
    @pytest.mark.filterwarnings("ignore:Unknown dimension values.*:UserWarning")
    def test_ingest_scalar_values(
        self, config, reingest_db, reingest_execution_obj, output_dir_with_data, mocker
    ):
        """Should ingest scalar metric values from a real CMEC bundle."""
        mock_result = mocker.Mock(spec=ExecutionResult)
        mock_result.metric_bundle_filename = pathlib.Path("diagnostic.json")
        mock_result.series_filename = pathlib.Path("series.json")
        mock_result.to_output_path = lambda f: output_dir_with_data / f

        cv = CV.load_from_file(config.paths.dimensions_cv)

        _ingest_metrics(reingest_db, mock_result, reingest_execution_obj, cv, additive=False)
        reingest_db.session.commit()

        scalars = (
            reingest_db.session.query(ScalarMetricValue)
            .filter_by(execution_id=reingest_execution_obj.id)
            .all()
        )
        assert len(scalars) >= 1
        assert scalars[0].value == 42.0

    @pytest.mark.filterwarnings("ignore:Unknown dimension values.*:UserWarning")
    def test_ingest_additive_skips_existing(
        self, config, reingest_db, reingest_execution_obj, output_dir_with_data, mocker
    ):
        """Additive mode should skip values with existing dimension signatures."""
        mock_result = mocker.Mock(spec=ExecutionResult)
        mock_result.metric_bundle_filename = pathlib.Path("diagnostic.json")
        mock_result.series_filename = pathlib.Path("series.json")
        mock_result.to_output_path = lambda f: output_dir_with_data / f

        cv = CV.load_from_file(config.paths.dimensions_cv)

        # First ingest
        _ingest_metrics(reingest_db, mock_result, reingest_execution_obj, cv, additive=False)
        reingest_db.session.commit()
        count_first = (
            reingest_db.session.query(MetricValue).filter_by(execution_id=reingest_execution_obj.id).count()
        )

        # Second ingest in additive mode should not duplicate
        _ingest_metrics(reingest_db, mock_result, reingest_execution_obj, cv, additive=True)
        reingest_db.session.commit()
        count_second = (
            reingest_db.session.query(MetricValue).filter_by(execution_id=reingest_execution_obj.id).count()
        )

        assert count_first == count_second


class TestReconstructWithDatasets:
    def test_reconstruct_with_linked_datasets(self, config, db_seeded, provider):
        """reconstruct_execution_definition should build dataset collections from linked datasets."""
        with db_seeded.session.begin():
            diag = db_seeded.session.query(DiagnosticModel).first()
            eg = ExecutionGroup(
                key="recon-test",
                diagnostic_id=diag.id,
                selectors={"cmip6": [["source_id", "ACCESS-ESM1-5"]]},
            )
            db_seeded.session.add(eg)
            db_seeded.session.flush()

            ex = Execution(
                execution_group_id=eg.id,
                successful=True,
                output_fragment="test/recon/abc",
                dataset_hash="h1",
            )
            db_seeded.session.add(ex)
            db_seeded.session.flush()

            # Link a dataset to the execution
            dataset = db_seeded.session.query(CMIP6Dataset).first()
            if dataset:
                db_seeded.session.execute(
                    execution_datasets.insert().values(
                        execution_id=ex.id,
                        dataset_id=dataset.id,
                    )
                )

        diagnostic = provider.get("mock")
        definition = reconstruct_execution_definition(config, ex, diagnostic)

        assert definition.key == "recon-test"
        if dataset:
            assert SourceDatasetType.CMIP6 in definition.datasets


class TestGetExecutionsForReingest:
    def test_filters_by_success(self, db_seeded):
        """By default should only return successful executions."""
        with db_seeded.session.begin():
            _register_provider(db_seeded, pmp_provider)
            _register_provider(db_seeded, esmvaltool_provider)

            diag = db_seeded.session.query(DiagnosticModel).first()
            eg = ExecutionGroup(key="test-filter", diagnostic_id=diag.id, selectors={})
            db_seeded.session.add(eg)
            db_seeded.session.flush()

            db_seeded.session.add(
                Execution(
                    execution_group_id=eg.id,
                    successful=True,
                    output_fragment="out-s",
                    dataset_hash="h1",
                )
            )

        results = get_executions_for_reingest(db_seeded, execution_group_ids=[eg.id], include_failed=False)
        assert len(results) == 1
        assert results[0][1].successful is True

    def test_include_failed(self, db_seeded):
        """With include_failed=True, should also return failed executions."""
        with db_seeded.session.begin():
            _register_provider(db_seeded, pmp_provider)
            _register_provider(db_seeded, esmvaltool_provider)

            diag = db_seeded.session.query(DiagnosticModel).first()
            eg = ExecutionGroup(key="test-failed", diagnostic_id=diag.id, selectors={})
            db_seeded.session.add(eg)
            db_seeded.session.flush()

            db_seeded.session.add(
                Execution(
                    execution_group_id=eg.id,
                    successful=False,
                    output_fragment="out-f",
                    dataset_hash="h2",
                )
            )

        results = get_executions_for_reingest(db_seeded, execution_group_ids=[eg.id], include_failed=True)
        assert len(results) >= 1

    def test_filters_by_group_ids(self, db_seeded):
        """Should only return executions for specified group IDs."""
        with db_seeded.session.begin():
            _register_provider(db_seeded, pmp_provider)

            diag = db_seeded.session.query(DiagnosticModel).first()
            eg1 = ExecutionGroup(key="group-a", diagnostic_id=diag.id, selectors={})
            eg2 = ExecutionGroup(key="group-b", diagnostic_id=diag.id, selectors={})
            db_seeded.session.add_all([eg1, eg2])
            db_seeded.session.flush()

            db_seeded.session.add(
                Execution(
                    execution_group_id=eg1.id,
                    successful=True,
                    output_fragment="out-a",
                    dataset_hash="ha",
                )
            )
            db_seeded.session.add(
                Execution(
                    execution_group_id=eg2.id,
                    successful=True,
                    output_fragment="out-b",
                    dataset_hash="hb",
                )
            )

        results = get_executions_for_reingest(db_seeded, execution_group_ids=[eg1.id])
        assert len(results) == 1
        assert results[0][0].id == eg1.id
