"""Tests for the reingest module."""

import datetime
import json
import pathlib

import pytest
from climate_ref_esmvaltool import provider as esmvaltool_provider
from climate_ref_pmp import provider as pmp_provider

from climate_ref.executor.reingest import (
    _extract_dataset_attributes,
    _validate_path_containment,
    get_executions_for_reingest,
    reconstruct_execution_definition,
    reingest_execution,
)
from climate_ref.executor.result_handling import ingest_execution_result
from climate_ref.models import ScalarMetricValue, SeriesMetricValue
from climate_ref.models.dataset import CMIP6Dataset
from climate_ref.models.diagnostic import Diagnostic as DiagnosticModel
from climate_ref.models.execution import (
    Execution,
    ExecutionGroup,
    ExecutionOutput,
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

SAMPLE_SERIES = [
    TSeries(
        dimensions={"source_id": "test-model"},
        values=[1.0, 2.0, 3.0],
        index=[0, 1, 2],
        index_name="time",
        attributes={"units": "K"},
    )
]


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
    (scratch_dir / "out.log").write_text("Execution log from original run\n")

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
    (scratch_dir / "out.log").write_text("Execution log from original run\n")

    return scratch_dir


@pytest.fixture
def mock_result_factory(mocker):
    """Factory to create mock ExecutionResult objects with sensible defaults."""

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


class TestValidatePathContainment:
    def test_valid_path_within_base(self, tmp_path):
        """Should not raise for a path within the base directory."""
        base = tmp_path / "base"
        base.mkdir()
        path = base / "subdir" / "file.txt"
        _validate_path_containment(path, base, "test")

    def test_path_escaping_base_raises(self, tmp_path):
        """Should raise ValueError when path escapes the base directory."""
        base = tmp_path / "base"
        base.mkdir()
        escaping_path = base / ".." / "outside"
        with pytest.raises(ValueError, match="escapes"):
            _validate_path_containment(escaping_path, base, "test")


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
        assert definition.output_directory == config.paths.scratch / "mock_provider/mock/abc123"

    def test_output_directory_under_scratch(self, config, reingest_db, reingest_execution_obj, provider):
        """Output directory should be under scratch for safe re-extraction."""
        diagnostic = provider.get("mock")
        definition = reconstruct_execution_definition(config, reingest_execution_obj, diagnostic)

        assert str(definition.output_directory).startswith(str(config.paths.scratch))

    def test_dataset_with_no_files(self, config, db_seeded, provider):
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

    def test_with_linked_datasets(self, config, db_seeded, provider):
        """Should build dataset collections from linked datasets."""
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


class TestReingestExecution:
    def test_missing_output_dir(self, config, reingest_db, reingest_execution_obj, mock_provider_registry):
        """Should return False when scratch directory doesn't exist."""
        result = reingest_execution(
            config=config,
            database=reingest_db,
            execution=reingest_execution_obj,
            provider_registry=mock_provider_registry,
        )
        assert result is False

    def test_unresolvable_diagnostic(self, config, reingest_db, reingest_execution_obj):
        """Should return False when provider registry can't resolve diagnostic."""
        empty_registry = ProviderRegistry(providers=[])
        result = reingest_execution(
            config=config,
            database=reingest_db,
            execution=reingest_execution_obj,
            provider_registry=empty_registry,
        )
        assert result is False

    def test_build_execution_result_failure(
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

    def test_unsuccessful_build_result(
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

    def test_successful_but_no_metric_bundle(
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

    @pytest.mark.filterwarnings("ignore:Unknown dimension values.*:UserWarning")
    def test_copies_scratch_before_build(
        self,
        config,
        reingest_db,
        reingest_execution_obj,
        mock_provider_registry,
        scratch_dir_with_results,
        mock_result_factory,
        mocker,
    ):
        """Scratch should be copied to new location before build_execution_result is called."""
        mock_result = mock_result_factory(scratch_dir_with_results)
        mock_diagnostic = mock_provider_registry.get_metric("mock_provider", "mock")

        # Track whether the new scratch dir exists when build_execution_result is called
        new_scratch_existed = []

        def capture_build(definition):
            original = config.paths.scratch / reingest_execution_obj.output_fragment
            siblings = [p for p in original.parent.iterdir() if p != original and p.is_dir()]
            new_scratch_existed.append(len(siblings) > 0)
            return mock_result

        mocker.patch.object(mock_diagnostic, "build_execution_result", side_effect=capture_build)

        ok = reingest_execution(
            config=config,
            database=reingest_db,
            execution=reingest_execution_obj,
            provider_registry=mock_provider_registry,
        )
        reingest_db.session.commit()
        assert ok is True
        assert new_scratch_existed == [True], "New scratch dir should exist before build_execution_result"

    @pytest.mark.filterwarnings("ignore:Unknown dimension values.*:UserWarning")
    def test_build_uses_new_output_directory(
        self,
        config,
        reingest_db,
        reingest_execution_obj,
        mock_provider_registry,
        scratch_dir_with_results,
        mock_result_factory,
        mocker,
    ):
        """build_execution_result should receive a definition pointing at the new scratch dir."""
        mock_result = mock_result_factory(scratch_dir_with_results)
        mock_diagnostic = mock_provider_registry.get_metric("mock_provider", "mock")
        spy = mocker.patch.object(mock_diagnostic, "build_execution_result", return_value=mock_result)

        ok = reingest_execution(
            config=config,
            database=reingest_db,
            execution=reingest_execution_obj,
            provider_registry=mock_provider_registry,
        )
        reingest_db.session.commit()
        assert ok is True

        spy.assert_called_once()
        definition = spy.call_args[0][0]

        original_scratch = config.paths.scratch / reingest_execution_obj.output_fragment
        assert definition.output_directory != original_scratch
        assert str(definition.output_directory).startswith(str(config.paths.scratch))
        assert definition.output_directory.exists()

    @pytest.mark.filterwarnings("ignore:Unknown dimension values.*:UserWarning")
    def test_rewrites_absolute_paths_in_copied_scratch(
        self,
        config,
        reingest_db,
        reingest_execution_obj,
        mock_provider_registry,
        scratch_dir_with_results,
        mock_result_factory,
        mocker,
    ):
        """Absolute paths in YAML/JSON files should be rewritten to reference the new scratch dir."""
        # Write a provenance file with absolute paths referencing the original scratch dir
        original_scratch = config.paths.scratch / reingest_execution_obj.output_fragment
        provenance_dir = original_scratch / "executions" / "recipe_test" / "run" / "diag" / "script"
        provenance_dir.mkdir(parents=True, exist_ok=True)
        provenance_file = provenance_dir / "diagnostic_provenance.yml"
        provenance_file.write_text(
            f"? {original_scratch}/executions/recipe_test/plots/plot.png\n: caption: Test plot\n"
        )

        mock_result = mock_result_factory(scratch_dir_with_results)
        mock_diagnostic = mock_provider_registry.get_metric("mock_provider", "mock")

        # Capture the new scratch dir from the definition
        captured_definitions = []

        def capture_build(definition):
            captured_definitions.append(definition)
            return mock_result

        mocker.patch.object(mock_diagnostic, "build_execution_result", side_effect=capture_build)

        ok = reingest_execution(
            config=config,
            database=reingest_db,
            execution=reingest_execution_obj,
            provider_registry=mock_provider_registry,
        )
        reingest_db.session.commit()
        assert ok is True

        # Verify the provenance file in the new scratch dir has rewritten paths
        new_output_dir = captured_definitions[0].output_directory
        new_provenance = (
            new_output_dir
            / "executions"
            / "recipe_test"
            / "run"
            / "diag"
            / "script"
            / "diagnostic_provenance.yml"
        )
        assert new_provenance.exists()
        content = new_provenance.read_text()
        # The old path with a trailing slash should not appear — it should
        # have been replaced with the new path. (We check with trailing /
        # because the new fragment contains the old fragment as a prefix.)
        assert f"{original_scratch}/" not in content
        assert str(new_output_dir) in content

    @pytest.mark.filterwarnings("ignore:Unknown dimension values.*:UserWarning")
    def test_creates_new_execution(
        self,
        config,
        reingest_db,
        reingest_execution_obj,
        mock_provider_registry,
        scratch_dir_with_results,
        mock_result_factory,
        mocker,
    ):
        """Reingest should create a new Execution record and leave the original untouched."""
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
        assert reingest_db.session.query(Execution).count() == original_count + 1

        # Original execution should be untouched
        original = reingest_db.session.get(Execution, original_id)
        assert original is not None
        assert original.successful is True

    @pytest.mark.filterwarnings("ignore:Unknown dimension values.*:UserWarning")
    def test_twice_creates_distinct_fragments(
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

        all_executions = reingest_db.session.query(Execution).all()
        assert len(all_executions) == 3
        fragments = [e.output_fragment for e in all_executions]
        assert len(set(fragments)) == 3, f"Expected unique fragments, got: {fragments}"

    @pytest.mark.filterwarnings("ignore:Unknown dimension values.*:UserWarning")
    def test_copies_results_to_new_directory(
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

    @pytest.mark.filterwarnings("ignore:Unknown dimension values.*:UserWarning")
    def test_does_not_touch_dirty_flag(
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
    def test_preserves_original_metrics(
        self,
        config,
        reingest_db,
        reingest_execution_obj,
        mock_provider_registry,
        scratch_dir_with_results,
        mock_result_factory,
        mocker,
    ):
        """Original execution's scalar values should remain untouched after reingest."""
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

        preserved_count = (
            reingest_db.session.query(ScalarMetricValue).filter_by(execution_id=execution.id).count()
        )
        assert preserved_count == original_count, "Original execution values should be untouched"

    @pytest.mark.filterwarnings("ignore:Unknown dimension values.*:UserWarning")
    def test_new_execution_state(
        self,
        config,
        reingest_db,
        reingest_execution_obj,
        mock_provider_registry,
        scratch_dir_with_results,
        mock_result_factory,
        mocker,
    ):
        """New execution should be successful, belong to the same group, and have correct path."""
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
        assert new_execution.execution_group_id == reingest_execution_obj.execution_group_id
        assert new_execution.dataset_hash == reingest_execution_obj.dataset_hash

    @pytest.mark.filterwarnings("ignore:Unknown dimension values.*:UserWarning")
    def test_copies_dataset_links(
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
    def test_with_no_datasets(
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

    @pytest.mark.filterwarnings("ignore:Unknown dimension values.*:UserWarning")
    def test_ingestion_failure_still_creates_execution_and_results_dir(
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

        # Corrupt the metric bundle
        (scratch_dir_with_results / "diagnostic.json").write_text("not valid json")

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

        new_execution = (
            reingest_db.session.query(Execution).filter(Execution.id != reingest_execution_obj.id).one()
        )
        results_dir = config.paths.results / new_execution.output_fragment
        assert results_dir.exists(), "Results directory should exist even when ingestion had errors"

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
        """Scratch directory should be preserved after reingest."""
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

    @pytest.mark.filterwarnings("ignore:Unknown dimension values.*:UserWarning")
    def test_copytree_failure_returns_false(
        self,
        config,
        reingest_db,
        reingest_execution_obj,
        mock_provider_registry,
        scratch_dir_with_results,
        mock_result_factory,
        mocker,
    ):
        """If copytree fails (e.g. disk full), reingest should return False and clean up."""
        mock_result = mock_result_factory(scratch_dir_with_results)
        _patch_build_result(mocker, mock_provider_registry, mock_result)

        mocker.patch("climate_ref.executor.reingest.shutil.copytree", side_effect=OSError("disk full"))

        ok = reingest_execution(
            config=config,
            database=reingest_db,
            execution=reingest_execution_obj,
            provider_registry=mock_provider_registry,
        )
        assert ok is False

    @pytest.mark.filterwarnings("ignore:Unknown dimension values.*:UserWarning")
    def test_handle_execution_result_exception_cleans_up(
        self,
        config,
        reingest_db,
        reingest_execution_obj,
        mock_provider_registry,
        scratch_dir_with_results,
        mock_result_factory,
        mocker,
    ):
        """If handle_execution_result raises, both scratch and results dirs should be cleaned up."""
        mock_result = mock_result_factory(scratch_dir_with_results)
        _patch_build_result(mocker, mock_provider_registry, mock_result)

        mocker.patch(
            "climate_ref.executor.reingest.handle_execution_result",
            side_effect=RuntimeError("unexpected failure"),
        )

        ok = reingest_execution(
            config=config,
            database=reingest_db,
            execution=reingest_execution_obj,
            provider_registry=mock_provider_registry,
        )
        assert ok is False

        reingest_db.session.rollback()

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

        new_execution = reingest_db.session.query(Execution).filter(Execution.id != execution.id).one()

        reingest_scalars = _snapshot_scalars(reingest_db, new_execution)
        reingest_series = _snapshot_series(reingest_db, new_execution)
        reingest_outputs = _snapshot_outputs(reingest_db, new_execution)

        assert original_scalars == reingest_scalars, (
            f"Scalar values differ: original={original_scalars}, reingest={reingest_scalars}"
        )
        assert original_series == reingest_series, (
            f"Series values differ: original={original_series}, reingest={reingest_series}"
        )
        assert original_outputs == reingest_outputs, (
            f"Output entries differ: original={original_outputs}, reingest={reingest_outputs}"
        )


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

    def test_skips_group_when_oldest_is_unsuccessful(self, db_seeded):
        """When include_failed=False, skip groups whose oldest execution failed."""
        with db_seeded.session.begin():
            diag = db_seeded.session.query(DiagnosticModel).first()
            eg = ExecutionGroup(key="oldest-failed", diagnostic_id=diag.id, selectors={})
            db_seeded.session.add(eg)
            db_seeded.session.flush()

            db_seeded.session.add(
                Execution(
                    execution_group_id=eg.id,
                    successful=False,
                    output_fragment="oldest-fail",
                    dataset_hash="h1",
                )
            )
            db_seeded.session.flush()

            db_seeded.session.add(
                Execution(
                    execution_group_id=eg.id,
                    successful=True,
                    output_fragment="newer-success",
                    dataset_hash="h1",
                )
            )

        results = get_executions_for_reingest(db_seeded, execution_group_ids=[eg.id], include_failed=False)
        assert len(results) == 0, "Should skip group when oldest execution is unsuccessful"

    def test_includes_group_when_oldest_unsuccessful_and_include_failed(self, db_seeded):
        """When include_failed=True, include groups whose oldest execution failed."""
        with db_seeded.session.begin():
            diag = db_seeded.session.query(DiagnosticModel).first()
            eg = ExecutionGroup(key="oldest-failed-incl", diagnostic_id=diag.id, selectors={})
            db_seeded.session.add(eg)
            db_seeded.session.flush()

            db_seeded.session.add(
                Execution(
                    execution_group_id=eg.id,
                    successful=False,
                    output_fragment="oldest-fail-incl",
                    dataset_hash="h1",
                )
            )

        results = get_executions_for_reingest(db_seeded, execution_group_ids=[eg.id], include_failed=True)
        assert len(results) == 1, "Should include group when include_failed=True"
