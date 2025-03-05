import pathlib
import shutil
from pathlib import Path
from subprocess import CompletedProcess

import cmip_ref_metrics_pmp.pmp_driver
import pandas as pd
import pytest
from cmip_ref_metrics_pmp.example import ExtratropicalModesOfVariability_PDO, process_json_result

from cmip_ref.solver import extract_covered_datasets
from cmip_ref_core.datasets import DatasetCollection
from cmip_ref_core.metrics import Metric
from cmip_ref_core.pycmec.metric import CMECMetric
from cmip_ref_core.pycmec.output import CMECOutput


@pytest.fixture(scope="module")
def pdo_example_dir() -> Path:
    return Path(__file__).parent / "test-data" / "pdo-example"


def get_first_metric_match(data_catalog: pd.DataFrame, metric: Metric) -> pd.DataFrame:
    datasets = extract_covered_datasets(data_catalog, metric.data_requirements[0])
    assert len(datasets) > 0
    return datasets[0]


def test_process_json_result(pdo_example_dir):
    json_file = (
        pdo_example_dir
        / "var_mode_PDO_EOF1_stat_cmip5_historical_mo_atm_ACCESS-ESM1-5_r1i1p1f1_2000-2005.json"
    )
    png_files = [pdo_example_dir / "pdo.png"]
    data_files = [pdo_example_dir / "pdo.nc"]

    cmec_output, cmec_metric = process_json_result(json_file, png_files, data_files)

    assert CMECMetric.model_validate(cmec_metric)
    assert CMECOutput.model_validate(cmec_output)
    assert len(cmec_metric.RESULTS)


def test_example_metric(cmip6_data_catalog, mocker, definition_factory, pdo_example_dir):
    metric = ExtratropicalModesOfVariability_PDO()
    metric_dataset = get_first_metric_match(cmip6_data_catalog, metric)

    definition = definition_factory(cmip6=DatasetCollection(metric_dataset, "instance_id"))

    def mock_run_call(cmd, *args, **kwargs):
        # Copy the output from the test-data directory to the output directory
        output_path = definition.output_directory
        shutil.copytree(pdo_example_dir, output_path)
        return CompletedProcess(cmd, 0, "stdout", "stderr")

    # Mock the subprocess.run call to avoid running PMP
    # Instead the mock_run_call function will be called
    mocker.patch.object(
        cmip_ref_metrics_pmp.pmp_driver.subprocess,
        "run",
        autospec=True,
        spec_set=True,
        side_effect=mock_run_call,
    )

    def mock_process_json_call(
        json_file: pathlib.Path, png_files: list[pathlib.Path], data_files: list[pathlib.Path]
    ):
        assert json_file.exists()
        assert len(png_files) > 0
        assert len(data_files) > 0
        return CMECOutput.create_template(), CMECMetric.create_template()

    mock_process_json = mocker.patch(
        "cmip_ref_metrics_pmp.example.process_json_result", side_effect=mock_process_json_call
    )

    result = metric.run(definition)

    assert mock_process_json.call_count == 1

    assert str(result.output_bundle_filename) == "output.json"

    output_bundle_path = definition.output_directory / result.output_bundle_filename

    assert result.successful
    assert output_bundle_path.exists()
    assert output_bundle_path.is_file()

    assert str(result.metric_bundle_filename) == "metric.json"

    metric_bundle_path = definition.output_directory / result.metric_bundle_filename

    assert result.successful
    assert metric_bundle_path.exists()
    assert metric_bundle_path.is_file()
