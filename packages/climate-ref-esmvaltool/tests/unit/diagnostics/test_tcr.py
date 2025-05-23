from pathlib import Path

import numpy as np
import pandas
import pytest
import xarray as xr
from climate_ref_esmvaltool.diagnostics import TransientClimateResponse
from climate_ref_esmvaltool.recipe import load_recipe

from climate_ref_core.datasets import DatasetCollection, ExecutionDatasetCollection, SourceDatasetType
from climate_ref_core.pycmec.metric import CMECMetric
from climate_ref_core.pycmec.output import CMECOutput


@pytest.fixture
def metric_dataset():
    return ExecutionDatasetCollection(
        {
            SourceDatasetType.CMIP6: DatasetCollection(
                datasets=pandas.read_json(Path(__file__).parent / "input_files_ecs.json"),
                slug_column="test",
            ),
        }
    )


def test_update_recipe(metric_dataset):
    input_files = {k: v.datasets for k, v in metric_dataset.items()}
    recipe = load_recipe("recipe_tcr.yml")
    TransientClimateResponse().update_recipe(recipe, input_files)
    assert len(recipe["datasets"]) == 2
    assert len(recipe["diagnostics"]) == 1
    assert set(recipe["diagnostics"]["tcr"]["variables"]) == {"tas"}
    undesired_keys = [
        "TCR",
        "SCATTERPLOT",
        "VAR_SETTING",
    ]
    for key in undesired_keys:
        assert key not in recipe


def test_format_output(tmp_path, metric_dataset):
    result_dir = tmp_path
    subdir = result_dir / "work" / "tcr" / "calculate"
    subdir.mkdir(parents=True)
    tcr = xr.Dataset(
        data_vars={
            "tcr": (["dim0"], np.array([1.0], dtype=np.float32)),
        },
    )
    tcr.to_netcdf(subdir / "tcr.nc")

    metric_args, output_args = TransientClimateResponse().format_result(
        result_dir,
        execution_dataset=metric_dataset,
        metric_args=CMECMetric.create_template(),
        output_args=CMECOutput.create_template(),
    )

    CMECMetric.model_validate(metric_args)
    assert metric_args.RESULTS["global"]["tcr"] == 1.0
    CMECOutput.model_validate(output_args)
