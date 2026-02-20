import pathlib
import shutil

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from climate_ref.executor.result_handling import _copy_file_to_results, handle_execution_result
from climate_ref.models import ScalarMetricValue, SeriesMetricValue
from climate_ref.models.execution import Execution, ExecutionOutput, ResultOutputType
from climate_ref.models.metric_value import MetricValueType
from climate_ref_core.diagnostics import ExecutionResult
from climate_ref_core.logging import EXECUTION_LOG_FILENAME
from climate_ref_core.metric_values import SeriesMetricValue as TSeries
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


def test_handle_execution_result_failed(config, db, mock_execution_result, mock_definition):
    result = ExecutionResult(definition=mock_definition, successful=False, metric_bundle_filename=None)

    handle_execution_result(config, db, mock_execution_result, result)

    mock_execution_result.mark_failed.assert_called_once()
    assert not mock_execution_result.execution_group.dirty


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
