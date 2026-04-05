"""Tests for the reingest module and allocate_output_fragment helper."""

import datetime
import json
import pathlib
from unittest.mock import patch

import pytest
from climate_ref_esmvaltool import provider as esmvaltool_provider
from climate_ref_pmp import provider as pmp_provider

from climate_ref.executor.fragment import allocate_output_fragment
from climate_ref.executor.reingest import (
    _extract_dataset_attributes,
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
    (scratch_dir / "execution.log").write_text("Execution log from original run\n")

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


# --- allocate_output_fragment tests ---


class TestAllocateOutputFragment:
    def test_appends_timestamp_suffix(self, tmp_path):
        """Should append a UTC timestamp suffix to the base fragment."""
        result = allocate_output_fragment("provider/diag/abc123", tmp_path)
        assert result.startswith("provider/diag/abc123_")
        # Suffix should be a valid timestamp: YYYYMMDDTHHMMSS followed by 6 microsecond digits
        suffix = result.split("_", 1)[1]
        assert len(suffix) == 21  # 8 date + T + 6 time + 6 microseconds
        assert "T" in suffix

    def test_preserves_base_fragment(self, tmp_path):
        """The original fragment should be a prefix of the result."""
        base = "my_provider/my_diag/hash123"
        result = allocate_output_fragment(base, tmp_path)
        assert result.startswith(base + "_")

    def test_different_calls_produce_different_fragments(self, tmp_path):
        """Two rapid calls should produce different fragments (microsecond resolution)."""
        result1 = allocate_output_fragment("provider/diag/abc123", tmp_path)
        result2 = allocate_output_fragment("provider/diag/abc123", tmp_path)
        assert result1 != result2

    def test_raises_if_directory_already_exists(self, tmp_path):
        """Should raise FileExistsError when the target directory already exists."""
        fixed_time = datetime.datetime(2026, 1, 1, 12, 0, 0, 0, tzinfo=datetime.timezone.utc)
        with patch("climate_ref.executor.fragment.datetime") as mock_dt:
            mock_dt.datetime.now.return_value = fixed_time
            mock_dt.timezone = datetime.timezone
            # First call succeeds
            fragment = allocate_output_fragment("provider/diag/abc123", tmp_path)
            # Create the directory so a second call with the same timestamp collides
            (tmp_path / fragment).mkdir(parents=True)
            with pytest.raises(FileExistsError, match="Output directory already exists"):
                allocate_output_fragment("provider/diag/abc123", tmp_path)


# --- extract dataset attributes tests ---


class TestExtractDatasetAttributes:
    def test_extracts_cmip6_attributes(self, db_seeded):
        dataset = db_seeded.session.query(CMIP6Dataset).first()
        attrs = _extract_dataset_attributes(dataset)

        assert "variable_id" in attrs
        assert "source_id" in attrs
        assert "instance_id" in attrs
        assert "id" not in attrs
        assert "dataset_type" not in attrs


# --- reconstruct_execution_definition tests ---


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


# --- reingest_execution tests ---


class TestReingestExecution:
    def test_reingest_missing_output_dir(
        self, config, reingest_db, reingest_execution_obj, mock_provider_registry
    ):
        """Should return False when scratch directory doesn't exist."""
        result = reingest_execution(
            config=config,
            database=reingest_db,
            execution=reingest_execution_obj,
            provider_registry=mock_provider_registry,
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
        )
        assert result is False

    @pytest.mark.filterwarnings("ignore:Unknown dimension values.*:UserWarning")
    def test_reingest_creates_new_execution(
        self,
        config,
        reingest_db,
        reingest_execution_obj,
        mock_provider_registry,
        scratch_dir_with_results,
        mock_result_factory,
        mocker,
    ):
        """Reingest should always create a new Execution record."""
        original_id = reingest_execution_obj.id
        original_count = reingest_db.session.query(Execution).count()

        mock_result = mock_result_factory(scratch_dir_with_results)
        _patch_build_result(mocker, mock_provider_registry, mock_result)

        ok = reingest_execution(
            config=config,
            database=reingest_db,
            execution=reingest_execution_obj,
            provider_registry=mock_provider_registry,
        )
        reingest_db.session.commit()
        assert ok is True
        new_count = reingest_db.session.query(Execution).count()
        assert new_count == original_count + 1

        # Original execution should be untouched
        original = reingest_db.session.get(Execution, original_id)
        assert original is not None
        assert original.successful is True

    @pytest.mark.filterwarnings("ignore:Unknown dimension values.*:UserWarning")
    def test_reingest_creates_unique_fragment(
        self,
        config,
        reingest_db,
        reingest_execution_obj,
        mock_provider_registry,
        scratch_dir_with_results,
        mock_result_factory,
        mocker,
    ):
        """New execution should have a different output_fragment from the original."""
        mock_result = mock_result_factory(scratch_dir_with_results)
        _patch_build_result(mocker, mock_provider_registry, mock_result)

        ok = reingest_execution(
            config=config,
            database=reingest_db,
            execution=reingest_execution_obj,
            provider_registry=mock_provider_registry,
        )
        reingest_db.session.commit()
        assert ok is True

        all_executions = reingest_db.session.query(Execution).all()
        fragments = [e.output_fragment for e in all_executions]
        assert len(set(fragments)) == len(fragments), f"Expected unique fragments, got: {fragments}"

    @pytest.mark.filterwarnings("ignore:Unknown dimension values.*:UserWarning")
    def test_reingest_twice_creates_distinct_fragments(
        self,
        config,
        reingest_db,
        reingest_execution_obj,
        mock_provider_registry,
        scratch_dir_with_results,
        mock_result_factory,
        mocker,
    ):
        """Running reingest twice should create distinct output fragments."""
        mock_result = mock_result_factory(scratch_dir_with_results)
        _patch_build_result(mocker, mock_provider_registry, mock_result)

        # Mock datetime.datetime.now to return different timestamps for each call
        t1 = datetime.datetime(2026, 1, 1, 12, 0, 0, tzinfo=datetime.timezone.utc)
        t2 = datetime.datetime(2026, 1, 1, 12, 0, 1, tzinfo=datetime.timezone.utc)
        mocker.patch(
            "climate_ref.executor.fragment.datetime.datetime",
            **{"now.side_effect": [t1, t2]},
        )

        ok1 = reingest_execution(
            config=config,
            database=reingest_db,
            execution=reingest_execution_obj,
            provider_registry=mock_provider_registry,
        )
        reingest_db.session.commit()
        assert ok1 is True

        reingest_db.session.refresh(reingest_execution_obj)
        ok2 = reingest_execution(
            config=config,
            database=reingest_db,
            execution=reingest_execution_obj,
            provider_registry=mock_provider_registry,
        )
        reingest_db.session.commit()
        assert ok2 is True

        # Should have 3 executions total: original + 2 reingested
        all_executions = reingest_db.session.query(Execution).all()
        assert len(all_executions) == 3

        fragments = [e.output_fragment for e in all_executions]
        assert len(set(fragments)) == 3, f"Expected unique fragments, got: {fragments}"

    @pytest.mark.filterwarnings("ignore:Unknown dimension values.*:UserWarning")
    def test_reingest_copies_results_to_new_directory(
        self,
        config,
        reingest_db,
        reingest_execution_obj,
        mock_provider_registry,
        scratch_dir_with_results,
        mock_result_factory,
        mocker,
    ):
        """Reingest should copy scratch tree to a new results directory."""
        mock_result = mock_result_factory(scratch_dir_with_results)
        _patch_build_result(mocker, mock_provider_registry, mock_result)

        ok = reingest_execution(
            config=config,
            database=reingest_db,
            execution=reingest_execution_obj,
            provider_registry=mock_provider_registry,
        )
        reingest_db.session.commit()
        assert ok is True

        new_execution = (
            reingest_db.session.query(Execution).filter(Execution.id != reingest_execution_obj.id).one()
        )

        results_dir = config.paths.results / new_execution.output_fragment
        assert results_dir.exists(), "Results should be copied to new directory"
        assert (results_dir / "diagnostic.json").exists()

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
        """The dirty flag should remain unchanged after reingest."""
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
        )
        reingest_db.session.commit()
        reingest_db.session.refresh(eg)
        assert eg.dirty is True

    @pytest.mark.filterwarnings("ignore:Unknown dimension values.*:UserWarning")
    def test_reingest_preserves_original_execution(
        self,
        config,
        reingest_db,
        reingest_execution_obj,
        mock_provider_registry,
        scratch_dir_with_results,
        mock_result_factory,
        mocker,
    ):
        """Original execution should remain untouched after reingest."""
        execution = reingest_execution_obj

        reingest_db.session.add(
            ScalarMetricValue(
                execution_id=execution.id,
                value=99.0,
                attributes={"original": True},
            )
        )
        reingest_db.session.commit()

        original_count = (
            reingest_db.session.query(ScalarMetricValue).filter_by(execution_id=execution.id).count()
        )

        mock_result = mock_result_factory(scratch_dir_with_results)
        _patch_build_result(mocker, mock_provider_registry, mock_result)

        ok = reingest_execution(
            config=config,
            database=reingest_db,
            execution=execution,
            provider_registry=mock_provider_registry,
        )
        reingest_db.session.commit()
        assert ok is True

        # Original execution's values should be unchanged
        preserved_count = (
            reingest_db.session.query(ScalarMetricValue).filter_by(execution_id=execution.id).count()
        )
        assert preserved_count == original_count, "Original execution values should be untouched"

    @pytest.mark.filterwarnings("ignore:Unknown dimension values.*:UserWarning")
    def test_reingest_ingestion_failure_logs_but_creates_execution(
        self,
        config,
        reingest_db,
        reingest_execution_obj,
        mock_provider_registry,
        scratch_dir_with_results,
        mock_result_factory,
        mocker,
    ):
        """handle_execution_result swallows ingestion errors; reingest still creates new execution."""
        original_count = reingest_db.session.query(Execution).count()

        mock_result = mock_result_factory(
            scratch_dir_with_results, output_bundle_filename=None, series_filename=None
        )
        _patch_build_result(mocker, mock_provider_registry, mock_result)

        # Corrupt the metric bundle — ingestion will fail internally but handle_execution_result
        # logs the error and continues rather than propagating.
        (scratch_dir_with_results / "diagnostic.json").write_text("not valid json")

        ok = reingest_execution(
            config=config,
            database=reingest_db,
            execution=reingest_execution_obj,
            provider_registry=mock_provider_registry,
        )
        reingest_db.session.commit()
        assert ok is True

        # A new execution record should still have been created
        new_count = reingest_db.session.query(Execution).count()
        assert new_count == original_count + 1, (
            "Reingest should create new execution even with ingestion error"
        )

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
        )

        assert result is False
        assert scratch_dir.exists(), "Scratch directory should be preserved after failure"


# --- register_execution_outputs tests ---


class TestRegisterExecutionOutputs:
    @pytest.mark.filterwarnings("ignore:Unknown dimension values.*:UserWarning")
    def test_registers_outputs_in_db(
        self, config, reingest_db, reingest_execution_obj, scratch_dir_with_data
    ):
        """Should register output entries from the bundle into the database."""
        execution = reingest_execution_obj
        bundle_path = scratch_dir_with_data / "output.json"
        cmec_output_bundle = CMECOutput.load_from_json(bundle_path)

        register_execution_outputs(
            reingest_db,
            execution,
            cmec_output_bundle.plots,
            output_type=ResultOutputType.Plot,
            base_path=scratch_dir_with_data,
        )
        reingest_db.session.commit()

        outputs = reingest_db.session.query(ExecutionOutput).filter_by(execution_id=execution.id).all()
        assert len(outputs) >= 1
        assert any(o.short_name == "test_plot" for o in outputs)


# --- ingest metrics tests ---


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


# --- reconstruct with datasets tests ---


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


# --- unsuccessful result tests ---


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
        )
        assert ok is False


# --- get_executions_for_reingest tests ---


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

    def test_selects_oldest_execution(self, db_seeded):
        """Should return the oldest (original) execution, not the latest.

        Reingested executions only have results directories, not scratch.
        Selecting the oldest ensures we always reingest from the execution
        whose scratch directory actually exists.
        """
        with db_seeded.session.begin():
            diag = db_seeded.session.query(DiagnosticModel).first()
            eg = ExecutionGroup(key="multi-exec", diagnostic_id=diag.id, selectors={})
            db_seeded.session.add(eg)
            db_seeded.session.flush()

            original = Execution(
                execution_group_id=eg.id,
                successful=True,
                output_fragment="original_fragment",
                dataset_hash="h-orig",
            )
            db_seeded.session.add(original)
            db_seeded.session.flush()

            reingested = Execution(
                execution_group_id=eg.id,
                successful=True,
                output_fragment="original_fragment_20260405T120000000000",
                dataset_hash="h-orig",
            )
            db_seeded.session.add(reingested)

        results = get_executions_for_reingest(db_seeded, execution_group_ids=[eg.id])
        assert len(results) == 1
        _, selected_execution = results[0]
        assert selected_execution.output_fragment == "original_fragment"
        assert selected_execution.id == original.id


# --- equivalence tests ---


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


class TestReingestEquivalence:
    """Reingest should produce the same DB state as fresh ingestion."""

    @pytest.mark.filterwarnings("ignore:Unknown dimension values.*:UserWarning")
    def test_reingest_matches_original(
        self,
        config,
        reingest_db,
        reingest_execution_obj,
        mock_provider_registry,
        scratch_dir_with_data,
        mock_result_factory,
        mocker,
    ):
        """Reingest should produce equivalent metrics and outputs to the original."""
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

        # Reingest creates a new execution from the same data
        ok = reingest_execution(
            config=config,
            database=reingest_db,
            execution=execution,
            provider_registry=mock_provider_registry,
        )
        reingest_db.session.commit()
        assert ok is True

        # Find the new execution created by reingest
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


class TestReingestDatasetLinks:
    @pytest.mark.filterwarnings("ignore:Unknown dimension values.*:UserWarning")
    def test_reingest_copies_dataset_links(
        self,
        config,
        reingest_db,
        reingest_execution_obj,
        mock_provider_registry,
        scratch_dir_with_results,
        mock_result_factory,
        mocker,
    ):
        """New execution should have the same dataset links as the original."""
        execution = reingest_execution_obj

        # Create a dataset to link to the execution
        dataset = CMIP6Dataset(
            slug="test-dataset.tas.gn",
            instance_id="CMIP6.test.tas",
            variable_id="tas",
            source_id="ACCESS-ESM1-5",
            experiment_id="historical",
            table_id="Amon",
            grid_label="gn",
            member_id="r1i1p1f1",
            variant_label="r1i1p1f1",
            version="v20200101",
            activity_id="CMIP",
            institution_id="CSIRO",
        )
        reingest_db.session.add(dataset)
        reingest_db.session.flush()

        reingest_db.session.execute(
            execution_datasets.insert().values(
                execution_id=execution.id,
                dataset_id=dataset.id,
            )
        )
        reingest_db.session.commit()

        original_dataset_ids = sorted(d.id for d in execution.datasets)
        assert len(original_dataset_ids) >= 1

        mock_result = mock_result_factory(
            scratch_dir_with_results, output_bundle_filename=None, series_filename=None
        )
        _patch_build_result(mocker, mock_provider_registry, mock_result)

        ok = reingest_execution(
            config=config,
            database=reingest_db,
            execution=execution,
            provider_registry=mock_provider_registry,
        )
        reingest_db.session.commit()
        assert ok is True

        new_execution = reingest_db.session.query(Execution).filter(Execution.id != execution.id).one()
        new_dataset_ids = sorted(d.id for d in new_execution.datasets)
        assert original_dataset_ids == new_dataset_ids, (
            f"Dataset links should be copied: original={original_dataset_ids}, new={new_dataset_ids}"
        )

    @pytest.mark.filterwarnings("ignore:Unknown dimension values.*:UserWarning")
    def test_reingest_with_no_datasets(
        self,
        config,
        reingest_db,
        reingest_execution_obj,
        mock_provider_registry,
        scratch_dir_with_results,
        mock_result_factory,
        mocker,
    ):
        """Reingest should succeed even when original execution has no dataset links."""
        assert len(reingest_execution_obj.datasets) == 0

        mock_result = mock_result_factory(
            scratch_dir_with_results, output_bundle_filename=None, series_filename=None
        )
        _patch_build_result(mocker, mock_provider_registry, mock_result)

        ok = reingest_execution(
            config=config,
            database=reingest_db,
            execution=reingest_execution_obj,
            provider_registry=mock_provider_registry,
        )
        reingest_db.session.commit()
        assert ok is True

        new_execution = (
            reingest_db.session.query(Execution).filter(Execution.id != reingest_execution_obj.id).one()
        )
        assert len(new_execution.datasets) == 0


class TestReingestExecutionState:
    """Verify the state of the new execution record after reingest."""

    @pytest.mark.filterwarnings("ignore:Unknown dimension values.*:UserWarning")
    def test_new_execution_marked_successful_with_correct_path(
        self,
        config,
        reingest_db,
        reingest_execution_obj,
        mock_provider_registry,
        scratch_dir_with_results,
        mock_result_factory,
        mocker,
    ):
        """New execution should be marked successful with the correct metric bundle path."""
        mock_result = mock_result_factory(
            scratch_dir_with_results, output_bundle_filename=None, series_filename=None
        )
        _patch_build_result(mocker, mock_provider_registry, mock_result)

        ok = reingest_execution(
            config=config,
            database=reingest_db,
            execution=reingest_execution_obj,
            provider_registry=mock_provider_registry,
        )
        reingest_db.session.commit()
        assert ok is True

        new_execution = (
            reingest_db.session.query(Execution).filter(Execution.id != reingest_execution_obj.id).one()
        )
        assert new_execution.successful is True
        assert new_execution.path is not None
        assert "diagnostic.json" in new_execution.path

    @pytest.mark.filterwarnings("ignore:Unknown dimension values.*:UserWarning")
    def test_new_execution_belongs_to_same_group(
        self,
        config,
        reingest_db,
        reingest_execution_obj,
        mock_provider_registry,
        scratch_dir_with_results,
        mock_result_factory,
        mocker,
    ):
        """New execution should belong to the same execution group as the original."""
        mock_result = mock_result_factory(
            scratch_dir_with_results, output_bundle_filename=None, series_filename=None
        )
        _patch_build_result(mocker, mock_provider_registry, mock_result)

        ok = reingest_execution(
            config=config,
            database=reingest_db,
            execution=reingest_execution_obj,
            provider_registry=mock_provider_registry,
        )
        reingest_db.session.commit()
        assert ok is True

        new_execution = (
            reingest_db.session.query(Execution).filter(Execution.id != reingest_execution_obj.id).one()
        )
        assert new_execution.execution_group_id == reingest_execution_obj.execution_group_id
        assert new_execution.dataset_hash == reingest_execution_obj.dataset_hash


class TestReingestFailureCleanup:
    """Verify that failed reingest cleans up the results directory."""

    @pytest.mark.filterwarnings("ignore:Unknown dimension values.*:UserWarning")
    def test_results_dir_created_even_on_ingestion_error(
        self,
        config,
        reingest_db,
        reingest_execution_obj,
        mock_provider_registry,
        scratch_dir_with_results,
        mock_result_factory,
        mocker,
    ):
        """handle_execution_result logs ingestion errors; results dir is still created."""
        mock_result = mock_result_factory(
            scratch_dir_with_results, output_bundle_filename=None, series_filename=None
        )
        _patch_build_result(mocker, mock_provider_registry, mock_result)

        # Corrupt the metric bundle — ingestion will log an error but reingest still succeeds
        (scratch_dir_with_results / "diagnostic.json").write_text("not valid json")

        ok = reingest_execution(
            config=config,
            database=reingest_db,
            execution=reingest_execution_obj,
            provider_registry=mock_provider_registry,
        )
        reingest_db.session.commit()
        assert ok is True

        new_execution = (
            reingest_db.session.query(Execution).filter(Execution.id != reingest_execution_obj.id).one()
        )
        results_dir = config.paths.results / new_execution.output_fragment
        assert results_dir.exists(), "Results directory should exist even when ingestion had errors"


# --- ingest_execution_result standalone tests ---


class TestIngestExecutionResultStandalone:
    """Test ingest_execution_result with the simplified signature."""

    @pytest.mark.filterwarnings("ignore:Unknown dimension values.*:UserWarning")
    def test_ingest_with_all_outputs(
        self, config, reingest_db, reingest_execution_obj, scratch_dir_with_data, mock_result_factory
    ):
        """Should ingest scalars, series, and register outputs in one call."""
        mock_result = mock_result_factory(scratch_dir_with_data)
        cv = CV.load_from_file(config.paths.dimensions_cv)

        ingest_execution_result(
            reingest_db,
            reingest_execution_obj,
            mock_result,
            cv,
            output_base_path=scratch_dir_with_data,
        )
        reingest_db.session.commit()

        execution_id = reingest_execution_obj.id

        scalars = reingest_db.session.query(ScalarMetricValue).filter_by(execution_id=execution_id).all()
        assert len(scalars) >= 1, "Should have ingested scalar values"
        assert scalars[0].value == 42.0

        series = reingest_db.session.query(SeriesMetricValue).filter_by(execution_id=execution_id).all()
        assert len(series) >= 1, "Should have ingested series values"

        outputs = reingest_db.session.query(ExecutionOutput).filter_by(execution_id=execution_id).all()
        assert len(outputs) >= 1, "Should have registered outputs"
        assert any(o.short_name == "test_plot" for o in outputs)

    @pytest.mark.filterwarnings("ignore:Unknown dimension values.*:UserWarning")
    def test_ingest_without_optional_outputs(
        self, config, reingest_db, reingest_execution_obj, scratch_dir_with_data, mock_result_factory
    ):
        """Should work with no output_bundle and no series."""
        mock_result = mock_result_factory(
            scratch_dir_with_data, output_bundle_filename=None, series_filename=None
        )
        cv = CV.load_from_file(config.paths.dimensions_cv)

        ingest_execution_result(
            reingest_db,
            reingest_execution_obj,
            mock_result,
            cv,
            output_base_path=scratch_dir_with_data,
        )
        reingest_db.session.commit()

        execution_id = reingest_execution_obj.id

        scalars = reingest_db.session.query(ScalarMetricValue).filter_by(execution_id=execution_id).all()
        assert len(scalars) >= 1, "Should still ingest scalar values"

        series = reingest_db.session.query(SeriesMetricValue).filter_by(execution_id=execution_id).all()
        assert len(series) == 0, "Should have no series values"

        outputs = reingest_db.session.query(ExecutionOutput).filter_by(execution_id=execution_id).all()
        assert len(outputs) == 0, "Should have no registered outputs"
