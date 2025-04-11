from pathlib import Path

import numpy as np
import pandas
import pytest
import xarray as xr
from cmip_ref_metrics_esmvaltool.metrics import EquilibriumClimateSensitivity
from cmip_ref_metrics_esmvaltool.recipe import load_recipe

from cmip_ref_core.datasets import DatasetCollection, MetricDataset, SourceDatasetType
from cmip_ref_core.pycmec.metric import CMECMetric
from cmip_ref_core.pycmec.output import CMECOutput


@pytest.fixture
def metric_dataset():
    return MetricDataset(
        {
            SourceDatasetType.CMIP6: DatasetCollection(
                datasets=pandas.read_json(Path(__file__).parent / "input_files_ecs.json"),
                slug_column="test",
            ),
        }
    )


def test_update_recipe(metric_dataset):
    input_files = metric_dataset[SourceDatasetType.CMIP6].datasets
    recipe = load_recipe("recipe_ecs.yml")
    EquilibriumClimateSensitivity().update_recipe(recipe, input_files)
    assert len(recipe["datasets"]) == 2
    assert len(recipe["diagnostics"]) == 1
    assert set(recipe["diagnostics"]["cmip6"]["variables"]) == {"tas", "rtnt"}


def test_format_output(tmp_path, metric_dataset):
    result_dir = tmp_path
    subdir = result_dir / "work" / "cmip6" / "ecs"
    subdir.mkdir(parents=True)
    ecs = xr.Dataset(
        data_vars={
            "ecs": (["dim0"], np.array([1.0])),
        },
    )
    ecs.to_netcdf(subdir / "ecs.nc")
    lambda_ = xr.Dataset(
        data_vars={
            "lambda": (["dim0"], np.array([2.0])),
        },
    )
    lambda_.to_netcdf(subdir / "lambda.nc")

    metric_args, output_args = EquilibriumClimateSensitivity().format_result(
        result_dir,
        metric_dataset=metric_dataset,
        metric_args=CMECMetric.create_template(),
        output_args=CMECOutput.create_template(),
    )

    CMECMetric.model_validate(metric_args)
    assert metric_args["RESULTS"]["ACCESS-ESM1-5"]["global"]["ecs"] == 1.0
    assert metric_args["RESULTS"]["ACCESS-ESM1-5"]["global"]["lambda"] == 2.0
    CMECOutput.model_validate(output_args)
