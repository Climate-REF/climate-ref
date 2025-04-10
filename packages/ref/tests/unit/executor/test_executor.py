import pathlib
import shutil

import pytest
from sqlalchemy.orm import Session

from cmip_ref.executor import _copy_file_to_results, handle_execution_result, import_executor_cls
from cmip_ref.executor.local import LocalExecutor
from cmip_ref.models import MetricExecutionResult as MetricExecutionResultModel
from cmip_ref.models.metric_execution import ResultOutput, ResultOutputType
from cmip_ref_core.exceptions import InvalidExecutorException
from cmip_ref_core.executor import Executor
from cmip_ref_core.metrics import MetricExecutionResult
from cmip_ref_core.pycmec.metric import CMECMetric
from cmip_ref_core.pycmec.output import CMECOutput


def test_import_executor():
    executor = import_executor_cls("cmip_ref.executor.local.LocalExecutor")

    assert isinstance(executor, Executor)
    assert executor == LocalExecutor


def test_import_executor_missing():
    fqn = "cmip_ref.executor.local.WrongExecutor"
    match = f"Invalid executor: '{fqn}'\n Executor 'WrongExecutor' not found in cmip_ref.executor.local"
    with pytest.raises(InvalidExecutorException, match=match):
        import_executor_cls(fqn)

    fqn = "missing.executor.local.WrongExecutor"
    match = f"Invalid executor: '{fqn}'\n Module 'missing.executor.local' not found"
    with pytest.raises(InvalidExecutorException, match=match):
        import_executor_cls(fqn)


@pytest.fixture
def mock_execution_result(mocker):
    mock_result = mocker.Mock(spec=MetricExecutionResultModel)
    mock_result.output_fragment = "output_fragment"
    return mock_result


def test_handle_execution_result_successful(
    db, config, mock_execution_result, mocker, definition_factory, test_data_dir
):
    metric_bundle_filename = pathlib.Path("bundle.json")
    definition = definition_factory()
    result = MetricExecutionResult(
        definition=definition, successful=True, metric_bundle_filename=metric_bundle_filename
    )

    # Copy a sample metric bundle to the output directory
    definition.output_directory.mkdir(parents=True, exist_ok=True)
    shutil.copy(
        test_data_dir / "cmec-output" / "pr_v3-LR_0101_1x1_esmf_metrics_default_v20241023_cmec.json",
        definition.to_output_path(metric_bundle_filename),
    )

    mock_copy = mocker.patch("cmip_ref.executor._copy_file_to_results")

    handle_execution_result(config, db, mock_execution_result, result)

    mock_copy.assert_called_once_with(
        config.paths.scratch,
        config.paths.results,
        mock_execution_result.output_fragment,
        metric_bundle_filename,
    )
    mock_execution_result.mark_successful.assert_called_once_with(metric_bundle_filename)
    assert not mock_execution_result.metric_execution_group.dirty


def test_handle_execution_result_with_files(config, mock_execution_result, mocker, definition_factory):
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
            "filename": "fig_2.jpg",
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

    definition = definition_factory()
    result = MetricExecutionResult.build_from_output_bundle(
        definition=definition, cmec_output_bundle=cmec_output, cmec_metric_bundle=cmec_metric
    )

    # The outputs must exist
    definition.to_output_path("fig_1.jpg").touch()
    definition.to_output_path("fig_2.jpg").touch()
    definition.to_output_path("index.html").touch()

    mock_result_output = mocker.patch("cmip_ref.executor.ResultOutput", spec=ResultOutput)

    handle_execution_result(config, db, mock_execution_result, result)

    assert db.session.add.call_count == 3
    mock_result_output.assert_called_with(
        metric_execution_result_id=mock_execution_result.id,
        output_type=ResultOutputType.HTML,
        filename="index.html",
        short_name="index",
        long_name="",
        description="Landing page",
    )
    db.session.add.assert_called_with(mock_result_output.return_value)


def test_handle_execution_result_failed(config, db, mock_execution_result, definition_factory):
    result = MetricExecutionResult(
        definition=definition_factory(), successful=False, metric_bundle_filename=None
    )

    handle_execution_result(config, db, mock_execution_result, result)

    mock_execution_result.mark_failed.assert_called_once()


def test_handle_execution_result_missing_file(config, db, mock_execution_result, definition_factory):
    result = MetricExecutionResult(
        definition=definition_factory(), successful=True, metric_bundle_filename=pathlib.Path("metric.json")
    )

    with pytest.raises(FileNotFoundError, match="Could not find metric.json in .*/scratch/output_fragment"):
        handle_execution_result(config, db, mock_execution_result, result)


@pytest.mark.parametrize("is_relative", [True, False])
def test_copy_file_to_results_success(mocker, is_relative):
    scratch_directory = pathlib.Path("/scratch")
    results_directory = pathlib.Path("/results")
    fragment = "output_fragment"
    filename = "bundle.zip"

    mocker.patch("pathlib.Path.exists", return_value=True)
    mock_copy = mocker.patch("shutil.copy")

    if is_relative:
        _copy_file_to_results(scratch_directory, results_directory, fragment, filename)
    else:
        _copy_file_to_results(
            scratch_directory, results_directory, fragment, scratch_directory / fragment / filename
        )

    mock_copy.assert_called_once_with(
        scratch_directory / fragment / filename, results_directory / fragment / filename
    )


def test_copy_file_to_results_file_not_found(mocker):
    scratch_directory = pathlib.Path("/scratch")
    results_directory = pathlib.Path("/results")
    fragment = "output_fragment"
    filename = "bundle.zip"

    mocker.patch("pathlib.Path.exists", return_value=False)

    with pytest.raises(
        FileNotFoundError, match=f"Could not find {filename} in {scratch_directory / fragment}"
    ):
        _copy_file_to_results(scratch_directory, results_directory, fragment, filename)
