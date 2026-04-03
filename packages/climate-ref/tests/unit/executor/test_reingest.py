"""Tests for the reingest module."""

import json
import pathlib

import pytest
from climate_ref_esmvaltool import provider as esmvaltool_provider
from climate_ref_pmp import provider as pmp_provider

from climate_ref.executor.reingest import (
    ReingestMode,
    _copy_with_backup,
    _delete_execution_metric_values,
    _extract_dataset_attributes,
    _get_existing_metric_dimensions,
    _sync_reingest_files_to_results,
    get_executions_for_reingest,
    reconstruct_execution_definition,
    reingest_execution,
)
from climate_ref.executor.result_handling import (
    ingest_execution_result,
    ingest_scalar_values,
    ingest_series_values,
    register_execution_outputs,
)
from climate_ref.models import ScalarMetricValue, SeriesMetricValue
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
from climate_ref_core.pycmec.output import CMECOutput, OutputDict


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


SAMPLE_SERIES = [
    TSeries(
        dimensions={"source_id": "test-model"},
        values=[1.0, 2.0, 3.0],
        index=[0, 1, 2],
        index_name="time",
        attributes={"units": "K"},
    )
]


def _create_scratch_dir(config, execution):
    """Create and return the scratch directory for an execution."""
    scratch_dir = config.paths.scratch / execution.output_fragment
    scratch_dir.mkdir(parents=True, exist_ok=True)
    return scratch_dir


@pytest.fixture
def scratch_dir_with_results(config, reingest_execution_obj):
    """Create a scratch directory with empty CMEC template files (simulates raw diagnostic output)."""
    scratch_dir = _create_scratch_dir(config, reingest_execution_obj)

    CMECMetric(**CMECMetric.create_template()).dump_to_json(scratch_dir / "diagnostic.json")
    CMECOutput(**CMECOutput.create_template()).dump_to_json(scratch_dir / "output.json")
    TSeries.dump_to_json(scratch_dir / "series.json", SAMPLE_SERIES)

    return scratch_dir


@pytest.fixture
def scratch_dir_with_data(config, reingest_execution_obj):
    """Create a scratch directory with CMEC files containing actual metric/output data."""
    scratch_dir = _create_scratch_dir(config, reingest_execution_obj)

    (scratch_dir / "diagnostic.json").write_text(
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

    (scratch_dir / "plot.png").write_bytes(b"fake png")
    (scratch_dir / "output.json").write_text(
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

    TSeries.dump_to_json(scratch_dir / "series.json", SAMPLE_SERIES)

    return scratch_dir


@pytest.fixture
def mock_result_factory(mocker):
    """Factory to create mock ExecutionResult objects with sensible defaults.

    Accepts an output_dir and optional overrides for output_bundle_filename
    and series_filename.
    """

    def _create(
        output_dir,
        *,
        output_bundle_filename=pathlib.Path("output.json"),
        series_filename=pathlib.Path("series.json"),
    ):
        mock_result = mocker.Mock(spec=ExecutionResult)
        mock_result.successful = True
        mock_result.metric_bundle_filename = pathlib.Path("diagnostic.json")
        mock_result.output_bundle_filename = output_bundle_filename
        mock_result.series_filename = series_filename
        mock_result.retryable = False
        mock_result.to_output_path = lambda f: output_dir / f if f else output_dir
        mock_result.as_relative_path = pathlib.Path
        return mock_result

    return _create


def _patch_build_result(mocker, registry, mock_result):
    """Patch build_execution_result on the mock diagnostic to return mock_result."""
    diagnostic = registry.get_metric("mock_provider", "mock")
    mocker.patch.object(diagnostic, "build_execution_result", return_value=mock_result)
    return diagnostic


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


class TestDeleteExecutionMetricValues:
    def test_deletes_metric_values_only(self, reingest_db, reingest_execution_obj):
        """Should remove metric values but leave outputs untouched."""
        execution = reingest_execution_obj

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

        _delete_execution_metric_values(reingest_db, execution)
        reingest_db.session.commit()

        assert reingest_db.session.query(MetricValue).filter_by(execution_id=execution.id).count() == 0
        # Outputs are not deleted by this function (handled separately in _apply_reingest_mode)
        assert reingest_db.session.query(ExecutionOutput).filter_by(execution_id=execution.id).count() == 1


class TestReingestExecution:
    def test_reingest_missing_output_dir(
        self, config, reingest_db, reingest_execution_obj, mock_provider_registry
    ):
        """Should return unsuccessful result when output directory doesn't exist."""
        result = reingest_execution(
            config=config,
            database=reingest_db,
            execution=reingest_execution_obj,
            provider_registry=mock_provider_registry,
            mode=ReingestMode.replace,
        )
        assert result is False

    def test_reingest_unresolvable_diagnostic(self, config, reingest_db, reingest_execution_obj):
        """Should return unsuccessful result when provider registry can't resolve diagnostic."""
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
        scratch_dir_with_results,
        mock_result_factory,
        mocker,
    ):
        """Replace mode should delete existing values and re-ingest."""
        execution = reingest_execution_obj

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

        mock_result = mock_result_factory(scratch_dir_with_results)
        _patch_build_result(mocker, mock_provider_registry, mock_result)

        ok = reingest_execution(
            config=config,
            database=reingest_db,
            execution=execution,
            provider_registry=mock_provider_registry,
            mode=ReingestMode.replace,
        )
        reingest_db.session.commit()
        assert ok is True
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
        scratch_dir_with_results,
        mock_result_factory,
        mocker,
    ):
        """Versioned mode should create a new Execution record."""
        original_id = reingest_execution_obj.id
        original_count = reingest_db.session.query(Execution).count()

        mock_result = mock_result_factory(scratch_dir_with_results)
        _patch_build_result(mocker, mock_provider_registry, mock_result)

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
        scratch_dir_with_results,
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
        scratch_dir_with_results,
        mock_result_factory,
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

        mock_result = mock_result_factory(
            scratch_dir_with_results, output_bundle_filename=None, series_filename=None
        )
        _patch_build_result(mocker, mock_provider_registry, mock_result)

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
        scratch_dir_with_results,
        mock_result_factory,
        mocker,
    ):
        """Additive mode should keep pre-existing metric values untouched."""
        execution = reingest_execution_obj

        reingest_db.session.add(
            ScalarMetricValue(
                execution_id=execution.id,
                value=42.0,
                attributes={"pre_existing": True},
                source_id="pre-existing-model",
            )
        )
        reingest_db.session.commit()

        mock_result = mock_result_factory(
            scratch_dir_with_results, output_bundle_filename=None, series_filename=None
        )
        _patch_build_result(mocker, mock_provider_registry, mock_result)

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
        scratch_dir_with_results,
        mock_result_factory,
        mocker,
    ):
        """Running additive reingest twice should not create duplicate rows."""
        execution = reingest_execution_obj

        mock_result = mock_result_factory(scratch_dir_with_results)
        _patch_build_result(mocker, mock_provider_registry, mock_result)

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
        scratch_dir_with_results,
        mock_result_factory,
        mocker,
    ):
        """If ingestion fails in replace mode, original data should be preserved."""
        execution = reingest_execution_obj

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

        mock_result = mock_result_factory(
            scratch_dir_with_results, output_bundle_filename=None, series_filename=None
        )
        _patch_build_result(mocker, mock_provider_registry, mock_result)

        # Make the scalar ingestion fail by corrupting the metric bundle
        (scratch_dir_with_results / "diagnostic.json").write_text("not valid json")

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
        scratch_dir_with_results,
        mock_result_factory,
        mocker,
    ):
        """Running versioned reingest twice should create distinct output fragments."""
        mock_result = mock_result_factory(scratch_dir_with_results)
        _patch_build_result(mocker, mock_provider_registry, mock_result)

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
        scratch_dir_with_results,
        mock_result_factory,
        mocker,
    ):
        """Reingest should mark a previously-failed execution as successful."""
        reingest_execution_obj.successful = False
        reingest_db.session.commit()
        assert reingest_execution_obj.successful is False

        mock_result = mock_result_factory(
            scratch_dir_with_results, output_bundle_filename=None, series_filename=None
        )
        _patch_build_result(mocker, mock_provider_registry, mock_result)

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

    def test_scratch_directory_preserved_after_success(
        self,
        config,
        reingest_db,
        reingest_execution_obj,
        mock_provider_registry,
        scratch_dir_with_results,
        mock_result_factory,
        mocker,
    ):
        """Scratch directory should be preserved after reingest (contains raw outputs)."""
        mock_result = mock_result_factory(
            scratch_dir_with_results, output_bundle_filename=None, series_filename=None
        )
        _patch_build_result(mocker, mock_provider_registry, mock_result)

        scratch_dir = config.paths.scratch / reingest_execution_obj.output_fragment

        reingest_execution(
            config=config,
            database=reingest_db,
            execution=reingest_execution_obj,
            provider_registry=mock_provider_registry,
            mode=ReingestMode.additive,
        )

        assert scratch_dir.exists(), "Scratch directory should be preserved after reingest"

    def test_scratch_directory_preserved_after_failure(
        self,
        config,
        reingest_db,
        reingest_execution_obj,
        mock_provider_registry,
        scratch_dir_with_results,
        mocker,
    ):
        """Scratch directory should be preserved even when reingest fails."""
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
        assert scratch_dir.exists(), "Scratch directory should be preserved after failure"


class TestCopyWithBackup:
    def test_copies_file(self, tmp_path):
        """Should copy file to destination."""
        src = tmp_path / "src" / "file.json"
        src.parent.mkdir()
        src.write_text('{"key": "new"}')
        dst = tmp_path / "dst" / "file.json"

        _copy_with_backup(src, dst)

        assert dst.exists()
        assert dst.read_text() == '{"key": "new"}'

    def test_backs_up_existing_file(self, tmp_path):
        """Should create .bak copy when overwriting an existing file."""
        src = tmp_path / "src" / "file.json"
        src.parent.mkdir()
        src.write_text('{"key": "new"}')
        dst = tmp_path / "dst" / "file.json"
        dst.parent.mkdir()
        dst.write_text('{"key": "old"}')

        _copy_with_backup(src, dst)

        assert dst.read_text() == '{"key": "new"}'
        backup = dst.with_suffix(".json.bak")
        assert backup.exists()
        assert backup.read_text() == '{"key": "old"}'

    def test_no_backup_when_no_existing(self, tmp_path):
        """Should not create .bak when destination doesn't exist."""
        src = tmp_path / "src" / "file.json"
        src.parent.mkdir()
        src.write_text('{"key": "value"}')
        dst = tmp_path / "dst" / "file.json"

        _copy_with_backup(src, dst)

        assert dst.exists()
        assert not dst.with_suffix(".json.bak").exists()


class TestSyncReingestFilesToResults:
    def test_non_versioned_copies_with_backup(self, config, reingest_execution_obj, scratch_dir_with_results):
        """Non-versioned mode should copy CMEC bundles with backup of existing files."""
        fragment = reingest_execution_obj.output_fragment
        results_dir = config.paths.results / fragment
        results_dir.mkdir(parents=True, exist_ok=True)
        (results_dir / "diagnostic.json").write_text('{"old": true}')

        mock_result = type(
            "R",
            (),
            {
                "metric_bundle_filename": pathlib.Path("diagnostic.json"),
                "output_bundle_filename": pathlib.Path("output.json"),
                "series_filename": pathlib.Path("series.json"),
            },
        )()

        _sync_reingest_files_to_results(config, fragment, fragment, mock_result, ReingestMode.additive)

        assert (results_dir / "diagnostic.json").exists()
        assert (results_dir / "diagnostic.json.bak").exists()
        assert (results_dir / "diagnostic.json.bak").read_text() == '{"old": true}'

    def test_versioned_copies_entire_tree(self, config, reingest_execution_obj, scratch_dir_with_results):
        """Versioned mode should copy entire scratch tree to new results directory."""
        src_fragment = reingest_execution_obj.output_fragment
        dst_fragment = src_fragment + "_v123abc"

        mock_result = type(
            "R",
            (),
            {
                "metric_bundle_filename": pathlib.Path("diagnostic.json"),
                "output_bundle_filename": None,
                "series_filename": None,
            },
        )()

        _sync_reingest_files_to_results(
            config, src_fragment, dst_fragment, mock_result, ReingestMode.versioned
        )

        dst_dir = config.paths.results / dst_fragment
        assert dst_dir.exists()
        assert (dst_dir / "diagnostic.json").exists()
        assert (dst_dir / "output.json").exists()
        assert (dst_dir / "series.json").exists()


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


class TestRegisterExecutionOutputs:
    @pytest.mark.filterwarnings("ignore:Unknown dimension values.*:UserWarning")
    def test_registers_outputs_in_db(
        self, config, reingest_db, reingest_execution_obj, scratch_dir_with_data
    ):
        """Should register output entries from the bundle into the database."""
        execution = reingest_execution_obj
        bundle_path = scratch_dir_with_data / "output.json"
        cmec_output_bundle = CMECOutput.load_from_json(bundle_path)

        results_base = config.paths.results / execution.output_fragment
        register_execution_outputs(
            reingest_db,
            execution,
            cmec_output_bundle.plots,
            output_type=ResultOutputType.Plot,
            base_path=results_base,
            fallback_path=scratch_dir_with_data,
        )
        reingest_db.session.commit()

        outputs = reingest_db.session.query(ExecutionOutput).filter_by(execution_id=execution.id).all()
        assert len(outputs) >= 1
        assert any(o.short_name == "test_plot" for o in outputs)


class TestIngestMetrics:
    @pytest.mark.filterwarnings("ignore:Unknown dimension values.*:UserWarning")
    def test_ingest_scalar_values(
        self, config, reingest_db, reingest_execution_obj, scratch_dir_with_data, mock_result_factory
    ):
        """Should ingest scalar metric values from a real CMEC bundle."""
        mock_result = mock_result_factory(scratch_dir_with_data)
        cv = CV.load_from_file(config.paths.dimensions_cv)

        ingest_scalar_values(
            database=reingest_db, result=mock_result, execution=reingest_execution_obj, cv=cv
        )
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
        self, config, reingest_db, reingest_execution_obj, scratch_dir_with_data, mock_result_factory
    ):
        """Additive mode should skip values with existing dimension signatures."""
        mock_result = mock_result_factory(scratch_dir_with_data)
        cv = CV.load_from_file(config.paths.dimensions_cv)

        # First ingest
        ingest_scalar_values(
            database=reingest_db, result=mock_result, execution=reingest_execution_obj, cv=cv
        )
        reingest_db.session.commit()
        count_first = (
            reingest_db.session.query(MetricValue).filter_by(execution_id=reingest_execution_obj.id).count()
        )

        # Second ingest in additive mode should not duplicate
        existing = _get_existing_metric_dimensions(reingest_db, reingest_execution_obj)
        ingest_scalar_values(
            database=reingest_db,
            result=mock_result,
            execution=reingest_execution_obj,
            cv=cv,
            existing=existing,
        )
        reingest_db.session.commit()
        count_second = (
            reingest_db.session.query(MetricValue).filter_by(execution_id=reingest_execution_obj.id).count()
        )

        assert count_first == count_second


class TestIngestMetricsWithSeries:
    @pytest.mark.filterwarnings("ignore:Unknown dimension values.*:UserWarning")
    def test_ingest_series_values(
        self, config, reingest_db, reingest_execution_obj, scratch_dir_with_data, mock_result_factory
    ):
        """Should ingest series metric values from a real series file."""
        mock_result = mock_result_factory(scratch_dir_with_data)
        cv = CV.load_from_file(config.paths.dimensions_cv)

        ingest_series_values(
            database=reingest_db, result=mock_result, execution=reingest_execution_obj, cv=cv
        )
        reingest_db.session.commit()

        series = (
            reingest_db.session.query(SeriesMetricValue)
            .filter_by(execution_id=reingest_execution_obj.id)
            .all()
        )
        assert len(series) >= 1

    @pytest.mark.filterwarnings("ignore:Unknown dimension values.*:UserWarning")
    def test_ingest_series_additive_skips_existing(
        self, config, reingest_db, reingest_execution_obj, scratch_dir_with_data, mock_result_factory
    ):
        """Additive mode should skip series with existing dimension signatures."""
        mock_result = mock_result_factory(scratch_dir_with_data)
        cv = CV.load_from_file(config.paths.dimensions_cv)

        # First ingest
        ingest_series_values(
            database=reingest_db, result=mock_result, execution=reingest_execution_obj, cv=cv
        )
        reingest_db.session.commit()

        count_first = (
            reingest_db.session.query(SeriesMetricValue)
            .filter_by(execution_id=reingest_execution_obj.id)
            .count()
        )

        # Second ingest in additive mode
        existing = _get_existing_metric_dimensions(reingest_db, reingest_execution_obj)
        ingest_series_values(
            database=reingest_db,
            result=mock_result,
            execution=reingest_execution_obj,
            cv=cv,
            existing=existing,
        )
        reingest_db.session.commit()

        count_second = (
            reingest_db.session.query(SeriesMetricValue)
            .filter_by(execution_id=reingest_execution_obj.id)
            .count()
        )

        assert count_first == count_second


class TestReconstructEmptyDataset:
    def test_reconstruct_dataset_with_no_files(self, config, db_seeded, provider):
        """Datasets with no files should produce an empty DataFrame."""
        with db_seeded.session.begin():
            diag = db_seeded.session.query(DiagnosticModel).first()
            eg = ExecutionGroup(
                key="empty-files",
                diagnostic_id=diag.id,
                selectors={"cmip6": [["source_id", "NONEXISTENT"]]},
            )
            db_seeded.session.add(eg)
            db_seeded.session.flush()

            # Link a dataset but it has no files
            dataset = db_seeded.session.query(CMIP6Dataset).first()
            ex = Execution(
                execution_group_id=eg.id,
                successful=True,
                output_fragment="test/empty-files/abc",
                dataset_hash="h1",
            )
            db_seeded.session.add(ex)
            db_seeded.session.flush()

            if dataset:
                db_seeded.session.execute(
                    execution_datasets.insert().values(
                        execution_id=ex.id,
                        dataset_id=dataset.id,
                    )
                )

        diagnostic = provider.get("mock")
        definition = reconstruct_execution_definition(config, ex, diagnostic)
        assert definition.key == "empty-files"


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


class TestReingestUnsuccessfulResult:
    def test_reingest_unsuccessful_build_result(
        self,
        config,
        reingest_db,
        reingest_execution_obj,
        mock_provider_registry,
        scratch_dir_with_results,
        mocker,
    ):
        """Should return False when build_execution_result returns unsuccessful."""
        mock_diagnostic = mock_provider_registry.get_metric("mock_provider", "mock")
        mock_result = mocker.Mock(spec=ExecutionResult)
        mock_result.successful = False
        mock_result.metric_bundle_filename = None
        mocker.patch.object(mock_diagnostic, "build_execution_result", return_value=mock_result)

        ok = reingest_execution(
            config=config,
            database=reingest_db,
            execution=reingest_execution_obj,
            provider_registry=mock_provider_registry,
            mode=ReingestMode.replace,
        )
        assert ok is False

    def test_reingest_successful_but_no_metric_bundle(
        self,
        config,
        reingest_db,
        reingest_execution_obj,
        mock_provider_registry,
        scratch_dir_with_results,
        mocker,
    ):
        """Should return False when result is successful but metric_bundle_filename is None."""
        mock_diagnostic = mock_provider_registry.get_metric("mock_provider", "mock")
        mock_result = mocker.Mock(spec=ExecutionResult)
        mock_result.successful = True
        mock_result.metric_bundle_filename = None
        mocker.patch.object(mock_diagnostic, "build_execution_result", return_value=mock_result)

        ok = reingest_execution(
            config=config,
            database=reingest_db,
            execution=reingest_execution_obj,
            provider_registry=mock_provider_registry,
            mode=ReingestMode.replace,
        )
        assert ok is False


class TestHandleReingestOutputsScratchFallback:
    def test_scratch_path_fallback(self, config, reingest_db, reingest_execution_obj):
        """When output filename is absolute under scratch, should fall back to scratch base."""
        execution = reingest_execution_obj

        # Create a scratch directory with a plot file
        scratch_base = config.paths.scratch / execution.output_fragment
        scratch_base.mkdir(parents=True, exist_ok=True)
        (scratch_base / "plot.png").write_bytes(b"fake png")

        # Build an OutputDict with an absolute path under scratch (not under results)
        outputs = {
            "test_plot": OutputDict(
                filename=str(scratch_base / "plot.png"),
                long_name="Test Plot",
                description="A test plot",
            ),
        }

        results_base = config.paths.results / execution.output_fragment
        register_execution_outputs(
            reingest_db,
            execution,
            outputs,
            output_type=ResultOutputType.Plot,
            base_path=results_base,
            fallback_path=scratch_base,
        )
        reingest_db.session.commit()

        db_outputs = reingest_db.session.query(ExecutionOutput).filter_by(execution_id=execution.id).all()
        assert len(db_outputs) == 1
        assert db_outputs[0].short_name == "test_plot"


class TestGetExecutionsForReingest:
    @pytest.fixture(autouse=True)
    def _register_providers(self, db_seeded):
        """Register providers once for all tests in this class."""
        with db_seeded.session.begin():
            _register_provider(db_seeded, pmp_provider)
            _register_provider(db_seeded, esmvaltool_provider)

    def test_filters_by_success(self, db_seeded):
        """By default should only return successful executions."""
        with db_seeded.session.begin():
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

    def test_filters_by_provider(self, db_seeded):
        """Should filter executions by provider slug."""
        with db_seeded.session.begin():
            diag_pmp = db_seeded.session.query(DiagnosticModel).filter_by(slug="enso_tel").first()
            diag_esm = db_seeded.session.query(DiagnosticModel).filter_by(slug="enso-characteristics").first()

            eg1 = ExecutionGroup(key="prov-pmp", diagnostic_id=diag_pmp.id, selectors={})
            eg2 = ExecutionGroup(key="prov-esm", diagnostic_id=diag_esm.id, selectors={})
            db_seeded.session.add_all([eg1, eg2])
            db_seeded.session.flush()

            db_seeded.session.add(
                Execution(
                    execution_group_id=eg1.id,
                    successful=True,
                    output_fragment="out-pmp",
                    dataset_hash="hp",
                )
            )
            db_seeded.session.add(
                Execution(
                    execution_group_id=eg2.id,
                    successful=True,
                    output_fragment="out-esm",
                    dataset_hash="he",
                )
            )

        results = get_executions_for_reingest(db_seeded, provider_filters=["pmp"])
        provider_slugs = {eg.diagnostic.provider.slug for eg, _ in results}
        assert "pmp" in provider_slugs
        assert "esmvaltool" not in provider_slugs

    def test_filters_by_diagnostic(self, db_seeded):
        """Should filter executions by diagnostic slug."""
        with db_seeded.session.begin():
            diag_pmp = db_seeded.session.query(DiagnosticModel).filter_by(slug="enso_tel").first()
            diag_esm = db_seeded.session.query(DiagnosticModel).filter_by(slug="enso-characteristics").first()

            eg1 = ExecutionGroup(key="diag-pmp", diagnostic_id=diag_pmp.id, selectors={})
            eg2 = ExecutionGroup(key="diag-esm", diagnostic_id=diag_esm.id, selectors={})
            db_seeded.session.add_all([eg1, eg2])
            db_seeded.session.flush()

            db_seeded.session.add(
                Execution(
                    execution_group_id=eg1.id,
                    successful=True,
                    output_fragment="out-d-pmp",
                    dataset_hash="hdp",
                )
            )
            db_seeded.session.add(
                Execution(
                    execution_group_id=eg2.id,
                    successful=True,
                    output_fragment="out-d-esm",
                    dataset_hash="hde",
                )
            )

        results = get_executions_for_reingest(db_seeded, diagnostic_filters=["enso_tel"])
        diag_slugs = {eg.diagnostic.slug for eg, _ in results}
        assert "enso_tel" in diag_slugs
        assert "enso-characteristics" not in diag_slugs

    def test_filters_out_none_executions(self, db_seeded):
        """Should filter out execution groups that have no executions."""
        with db_seeded.session.begin():
            diag = db_seeded.session.query(DiagnosticModel).first()
            eg = ExecutionGroup(key="no-exec", diagnostic_id=diag.id, selectors={})
            db_seeded.session.add(eg)

        results = get_executions_for_reingest(db_seeded, execution_group_ids=[eg.id])
        assert len(results) == 0


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


class TestVersionedReingestEquivalence:
    """Versioned reingest should produce the same DB state as fresh ingestion."""

    @pytest.mark.filterwarnings("ignore:Unknown dimension values.*:UserWarning")
    def test_versioned_reingest_matches_original(
        self,
        config,
        reingest_db,
        reingest_execution_obj,
        mock_provider_registry,
        scratch_dir_with_data,
        mock_result_factory,
        mocker,
    ):
        """Versioned reingest should produce equivalent metrics and outputs to the original."""
        execution = reingest_execution_obj
        mock_result = mock_result_factory(scratch_dir_with_data)
        _patch_build_result(mocker, mock_provider_registry, mock_result)

        # Original ingestion via the shared path
        cv = CV.load_from_file(config.paths.dimensions_cv)
        ingest_execution_result(
            reingest_db,
            execution,
            mock_result,
            cv,
            output_base_path=config.paths.scratch / execution.output_fragment,
        )
        execution.mark_successful(mock_result.as_relative_path(mock_result.metric_bundle_filename))
        reingest_db.session.commit()

        # Snapshot original DB state
        original_scalars = _snapshot_scalars(reingest_db, execution)
        original_series = _snapshot_series(reingest_db, execution)
        original_outputs = _snapshot_outputs(reingest_db, execution)

        assert original_scalars, "Original ingestion should produce scalar values"

        # Versioned reingest creates a new execution from the same data
        ok = reingest_execution(
            config=config,
            database=reingest_db,
            execution=execution,
            provider_registry=mock_provider_registry,
            mode=ReingestMode.versioned,
        )
        reingest_db.session.commit()
        assert ok is True

        # Find the new execution created by versioned reingest
        new_execution = reingest_db.session.query(Execution).filter(Execution.id != execution.id).one()

        # Snapshot reingest DB state
        reingest_scalars = _snapshot_scalars(reingest_db, new_execution)
        reingest_series = _snapshot_series(reingest_db, new_execution)
        reingest_outputs = _snapshot_outputs(reingest_db, new_execution)

        # Both paths go through ingest_execution_result, so results must match
        assert original_scalars == reingest_scalars, (
            f"Scalar values differ: original={original_scalars}, reingest={reingest_scalars}"
        )
        assert original_series == reingest_series, (
            f"Series values differ: original={original_series}, reingest={reingest_series}"
        )
        assert original_outputs == reingest_outputs, (
            f"Output entries differ: original={original_outputs}, reingest={reingest_outputs}"
        )
