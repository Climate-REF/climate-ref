import json
import pathlib
import shutil

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from climate_ref.executor.result_handling import (
    _copy_file_to_results,
    handle_execution_result,
    ingest_execution_result,
    ingest_scalar_values,
    ingest_series_values,
    register_execution_outputs,
)
from climate_ref.models import ScalarMetricValue, SeriesMetricValue
from climate_ref.models.diagnostic import Diagnostic as DiagnosticModel
from climate_ref.models.execution import (
    Execution,
    ExecutionGroup,
    ExecutionOutput,
    ResultOutputType,
)
from climate_ref.models.metric_value import MetricValueType
from climate_ref.models.provider import Provider as ProviderModel
from climate_ref_core.diagnostics import ExecutionResult
from climate_ref_core.logging import EXECUTION_LOG_FILENAME
from climate_ref_core.metric_values import SeriesMetricValue as TSeries
from climate_ref_core.pycmec.controlled_vocabulary import CV
from climate_ref_core.pycmec.metric import CMECMetric
from climate_ref_core.pycmec.output import CMECOutput


@pytest.fixture
def mock_execution_result(mocker):
    mock_result = mocker.Mock(spec=Execution)
    mock_result.id = 1
    mock_result.output_fragment = "output_fragment"
    return mock_result


@pytest.fixture
def mock_definition(mocker, definition_factory):
    definition = definition_factory(
        diagnostic=mocker.Mock(),
    )

    # Ensure that the output directory exists and a log file is available
    definition.output_directory.mkdir(parents=True, exist_ok=True)
    definition.to_output_path(EXECUTION_LOG_FILENAME).touch()

    return definition


@pytest.mark.filterwarnings("ignore:Unknown dimension values.*CalendarMonths.*:UserWarning")
def test_handle_execution_result_successful(
    db, config, mock_execution_result, mocker, mock_definition, test_data_dir
):
    metric_bundle_filename = pathlib.Path("bundle.json")
    result = ExecutionResult(
        definition=mock_definition, successful=True, metric_bundle_filename=metric_bundle_filename
    )

    shutil.copy(
        test_data_dir / "cmec-output" / "pr_v3-LR_0101_1x1_esmf_metrics_default_v20241023_cmec.json",
        mock_definition.to_output_path(metric_bundle_filename),
    )

    mock_copy = mocker.patch("climate_ref.executor.result_handling._copy_file_to_results")

    handle_execution_result(config, db, mock_execution_result, result)

    mock_copy.assert_any_call(
        config.paths.scratch,
        config.paths.results,
        mock_execution_result.output_fragment,
        EXECUTION_LOG_FILENAME,
    )
    mock_copy.assert_called_with(
        config.paths.scratch,
        config.paths.results,
        mock_execution_result.output_fragment,
        metric_bundle_filename,
    )
    mock_execution_result.mark_successful.assert_called_once_with(metric_bundle_filename)
    assert not mock_execution_result.execution_group.dirty

    scalars = list(db.session.execute(select(ScalarMetricValue)).scalars())
    assert scalars
    assert scalars[0].type == MetricValueType.SCALAR


@pytest.mark.filterwarnings("ignore:Unknown dimension values.*CalendarMonths.*:UserWarning")
def test_handle_execution_result_with_series(
    db, config, mock_execution_result, mocker, mock_definition, test_data_dir
):
    metric_bundle_filename = pathlib.Path("bundle.json")
    series_filename = pathlib.Path("series.json")
    result = ExecutionResult(
        definition=mock_definition,
        successful=True,
        metric_bundle_filename=metric_bundle_filename,
        series_filename=series_filename,
    )

    shutil.copy(
        test_data_dir / "cmec-output" / "pr_v3-LR_0101_1x1_esmf_metrics_default_v20241023_cmec.json",
        mock_definition.to_output_path(metric_bundle_filename),
    )

    series_data = [
        TSeries(
            dimensions={"source_id": "test1"},
            values=[1.0, 2.0, 3.0],
            index=[0, 1, 2],
            index_name="time",
            attributes={"attr": "value1"},
        )
    ]
    TSeries.dump_to_json(mock_definition.to_output_path(series_filename), series_data)

    mock_copy = mocker.patch("climate_ref.executor.result_handling._copy_file_to_results")

    handle_execution_result(config, db, mock_execution_result, result)

    mock_copy.assert_any_call(
        config.paths.scratch,
        config.paths.results,
        mock_execution_result.output_fragment,
        EXECUTION_LOG_FILENAME,
    )
    mock_copy.assert_any_call(
        config.paths.scratch,
        config.paths.results,
        mock_execution_result.output_fragment,
        metric_bundle_filename,
    )
    mock_copy.assert_called_with(
        config.paths.scratch,
        config.paths.results,
        mock_execution_result.output_fragment,
        series_filename,
    )
    mock_execution_result.mark_successful.assert_called_once_with(metric_bundle_filename)
    assert not mock_execution_result.execution_group.dirty

    scalars = list(db.session.execute(select(ScalarMetricValue)).scalars())
    assert scalars
    assert scalars[0].type == MetricValueType.SCALAR

    series = list(db.session.execute(select(SeriesMetricValue)).scalars())
    assert len(series) == 1
    assert series[0].type == MetricValueType.SERIES
    assert series[0].dimensions == {"source_id": "test1"}
    assert series[0].values == [1.0, 2.0, 3.0]
    assert series[0].index == [0, 1, 2]
    assert series[0].index_name == "time"
    assert series[0].attributes == {"attr": "value1"}


def test_handle_execution_result_with_files(config, mock_execution_result, mocker, mock_definition):
    db = mocker.MagicMock()
    db.session = mocker.MagicMock(spec=Session)

    cmec_metric = CMECMetric(**CMECMetric.create_template())
    cmec_output = CMECOutput(**CMECOutput.create_template())
    cmec_output.update(
        "plots",
        short_name="example1",
        dict_content={
            "long_name": "awesome figure",
            "filename": "fig_1.jpg",
            "description": "test add plots",
        },
    )
    cmec_output.update(
        "plots",
        short_name="example2",
        dict_content={
            "long_name": "awesome figure",
            "filename": "folder/fig_2.jpg",
            "description": "test add plots",
        },
    )
    cmec_output.update(
        "html",
        short_name="index",
        dict_content={
            "long_name": "",
            "filename": "index.html",
            "description": "Landing page",
        },
    )

    result = ExecutionResult.build_from_output_bundle(
        definition=mock_definition, cmec_output_bundle=cmec_output, cmec_metric_bundle=cmec_metric
    )

    # The outputs must exist
    mock_definition.to_output_path("fig_1.jpg").touch()
    mock_definition.to_output_path("folder").mkdir()
    mock_definition.to_output_path("folder/fig_2.jpg").touch()
    mock_definition.to_output_path("index.html").touch()

    mock_result_output = mocker.patch(
        "climate_ref.executor.result_handling.ExecutionOutput.build", spec=ExecutionOutput
    )

    handle_execution_result(config, db, mock_execution_result, result)

    assert db.session.add.call_count == 3
    mock_result_output.assert_called_with(
        execution_id=mock_execution_result.id,
        output_type=ResultOutputType.HTML,
        filename="index.html",
        short_name="index",
        long_name="",
        description="Landing page",
        dimensions={},
    )
    db.session.add.assert_called_with(mock_result_output.return_value)


def test_handle_execution_result_diagnostic_failure(config, db, mock_execution_result, mock_definition):
    """A diagnostic logic error should clear dirty so the execution is not retried"""
    result = ExecutionResult(
        definition=mock_definition, successful=False, metric_bundle_filename=None, retryable=False
    )

    handle_execution_result(config, db, mock_execution_result, result)

    mock_execution_result.mark_failed.assert_called_once()
    assert not mock_execution_result.execution_group.dirty


def test_handle_execution_result_system_failure(config, db, mock_execution_result, mock_definition):
    """A system error (e.g. OOM) should leave dirty=True so the execution is retried"""
    mock_execution_result.execution_group.dirty = True
    result = ExecutionResult(
        definition=mock_definition, successful=False, metric_bundle_filename=None, retryable=True
    )

    handle_execution_result(config, db, mock_execution_result, result)

    mock_execution_result.mark_failed.assert_called_once()
    assert mock_execution_result.execution_group.dirty


def test_handle_execution_result_missing_log_file_leaves_dirty(
    config, db, mock_execution_result, mocker, definition_factory
):
    """Missing log file suggests the process was killed, so dirty should remain True"""
    definition = definition_factory(diagnostic=mocker.Mock())
    # Do NOT create the log file
    definition.output_directory.mkdir(parents=True, exist_ok=True)

    mock_execution_result.execution_group.dirty = True
    result = ExecutionResult(
        definition=definition, successful=True, metric_bundle_filename=pathlib.Path("diagnostic.json")
    )

    handle_execution_result(config, db, mock_execution_result, result)

    mock_execution_result.mark_failed.assert_called_once()
    assert mock_execution_result.execution_group.dirty


def test_handle_execution_result_missing_file(config, db, mock_execution_result, mock_definition):
    result = ExecutionResult(
        definition=mock_definition, successful=True, metric_bundle_filename=pathlib.Path("diagnostic.json")
    )

    with pytest.raises(
        FileNotFoundError, match=r"Could not find diagnostic.json in .*/scratch/output_fragment"
    ):
        handle_execution_result(config, db, mock_execution_result, result)


@pytest.mark.parametrize("is_relative", [True, False])
@pytest.mark.parametrize("filename", ("bundle.zip", "nested/bundle.zip"))
def test_copy_file_to_results_success(filename, is_relative, tmp_path):
    scratch_directory = (tmp_path / "scratch").resolve()
    results_directory = (tmp_path / "executions").resolve()
    fragment = "output_fragment"

    scratch_filename = scratch_directory / fragment / filename
    scratch_filename.parent.mkdir(parents=True, exist_ok=True)
    scratch_filename.touch()

    if is_relative:
        _copy_file_to_results(scratch_directory, results_directory, fragment, filename)
    else:
        _copy_file_to_results(
            scratch_directory, results_directory, fragment, scratch_directory / fragment / filename
        )

    assert (results_directory / fragment / filename).exists()


def test_copy_file_to_results_file_not_found(mocker):
    scratch_directory = pathlib.Path("/scratch")
    results_directory = pathlib.Path("/executions")
    fragment = "output_fragment"
    filename = "bundle.zip"

    mocker.patch("pathlib.Path.exists", return_value=False)

    with pytest.raises(
        FileNotFoundError, match=f"Could not find {filename} in {scratch_directory / fragment}"
    ):
        _copy_file_to_results(scratch_directory, results_directory, fragment, filename)


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
def _ingestion_db(db, config):
    """Set up a database with an execution group and execution for ingestion tests."""
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
def ingestion_execution(_ingestion_db):
    """Get the execution object from the ingestion DB fixture."""
    return _ingestion_db.session.query(Execution).one()


@pytest.fixture
def scratch_dir_with_data(config, ingestion_execution):
    """Create a scratch directory with CMEC files containing actual metric/output data."""
    scratch_dir = config.paths.scratch / ingestion_execution.output_fragment
    scratch_dir.mkdir(parents=True, exist_ok=True)

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


class TestRegisterExecutionOutputs:
    @pytest.mark.filterwarnings("ignore:Unknown dimension values.*:UserWarning")
    def test_registers_outputs_in_db(self, config, _ingestion_db, ingestion_execution, scratch_dir_with_data):
        """Should register output entries from the bundle into the database."""
        bundle_path = scratch_dir_with_data / "output.json"
        cmec_output_bundle = CMECOutput.load_from_json(bundle_path)

        register_execution_outputs(
            _ingestion_db,
            ingestion_execution,
            cmec_output_bundle.plots,
            output_type=ResultOutputType.Plot,
            base_path=scratch_dir_with_data,
        )
        _ingestion_db.session.commit()

        outputs = (
            _ingestion_db.session.query(ExecutionOutput).filter_by(execution_id=ingestion_execution.id).all()
        )
        assert len(outputs) >= 1
        assert any(o.short_name == "test_plot" for o in outputs)


class TestIngestScalarValues:
    @pytest.mark.filterwarnings("ignore:Unknown dimension values.*:UserWarning")
    def test_ingest_scalar_values(
        self, config, _ingestion_db, ingestion_execution, scratch_dir_with_data, mock_result_factory
    ):
        """Should ingest scalar metric values from a real CMEC bundle."""
        mock_result = mock_result_factory(scratch_dir_with_data)
        cv = CV.load_from_file(config.paths.dimensions_cv)

        ingest_scalar_values(database=_ingestion_db, result=mock_result, execution=ingestion_execution, cv=cv)
        _ingestion_db.session.commit()

        scalars = (
            _ingestion_db.session.query(ScalarMetricValue)
            .filter_by(execution_id=ingestion_execution.id)
            .all()
        )
        assert len(scalars) >= 1
        assert scalars[0].value == 42.0


class TestIngestSeriesValues:
    @pytest.mark.filterwarnings("ignore:Unknown dimension values.*:UserWarning")
    def test_ingest_series_values(
        self, config, _ingestion_db, ingestion_execution, scratch_dir_with_data, mock_result_factory
    ):
        """Should ingest series metric values from a real series file."""
        mock_result = mock_result_factory(scratch_dir_with_data)
        cv = CV.load_from_file(config.paths.dimensions_cv)

        ingest_series_values(database=_ingestion_db, result=mock_result, execution=ingestion_execution, cv=cv)
        _ingestion_db.session.commit()

        series = (
            _ingestion_db.session.query(SeriesMetricValue)
            .filter_by(execution_id=ingestion_execution.id)
            .all()
        )
        assert len(series) >= 1


class TestIngestExecutionResult:
    @pytest.mark.filterwarnings("ignore:Unknown dimension values.*:UserWarning")
    def test_ingest_with_all_outputs(
        self, config, _ingestion_db, ingestion_execution, scratch_dir_with_data, mock_result_factory
    ):
        """Should ingest scalars, series, and register outputs in one call."""
        mock_result = mock_result_factory(scratch_dir_with_data)
        cv = CV.load_from_file(config.paths.dimensions_cv)

        ingest_execution_result(
            _ingestion_db,
            ingestion_execution,
            mock_result,
            cv,
            output_base_path=scratch_dir_with_data,
        )
        _ingestion_db.session.commit()

        execution_id = ingestion_execution.id

        scalars = _ingestion_db.session.query(ScalarMetricValue).filter_by(execution_id=execution_id).all()
        assert len(scalars) >= 1, "Should have ingested scalar values"
        assert scalars[0].value == 42.0

        series = _ingestion_db.session.query(SeriesMetricValue).filter_by(execution_id=execution_id).all()
        assert len(series) >= 1, "Should have ingested series values"

        outputs = _ingestion_db.session.query(ExecutionOutput).filter_by(execution_id=execution_id).all()
        assert len(outputs) >= 1, "Should have registered outputs"
        assert any(o.short_name == "test_plot" for o in outputs)

    @pytest.mark.filterwarnings("ignore:Unknown dimension values.*:UserWarning")
    def test_ingest_without_optional_outputs(
        self, config, _ingestion_db, ingestion_execution, scratch_dir_with_data, mock_result_factory
    ):
        """Should work with no output_bundle and no series."""
        mock_result = mock_result_factory(
            scratch_dir_with_data, output_bundle_filename=None, series_filename=None
        )
        cv = CV.load_from_file(config.paths.dimensions_cv)

        ingest_execution_result(
            _ingestion_db,
            ingestion_execution,
            mock_result,
            cv,
            output_base_path=scratch_dir_with_data,
        )
        _ingestion_db.session.commit()

        execution_id = ingestion_execution.id

        scalars = _ingestion_db.session.query(ScalarMetricValue).filter_by(execution_id=execution_id).all()
        assert len(scalars) >= 1, "Should still ingest scalar values"

        series = _ingestion_db.session.query(SeriesMetricValue).filter_by(execution_id=execution_id).all()
        assert len(series) == 0, "Should have no series values"

        outputs = _ingestion_db.session.query(ExecutionOutput).filter_by(execution_id=execution_id).all()
        assert len(outputs) == 0, "Should have no registered outputs"
