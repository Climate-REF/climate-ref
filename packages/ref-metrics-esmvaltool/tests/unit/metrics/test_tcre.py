from pathlib import Path

import numpy as np
import pandas
import pytest
import xarray as xr
from cmip_ref_metrics_esmvaltool.metrics import TransientClimateResponseEmissions
from cmip_ref_metrics_esmvaltool.recipe import load_recipe

from cmip_ref_core.datasets import DatasetCollection, MetricDataset, SourceDatasetType
from cmip_ref_core.pycmec.metric import CMECMetric
from cmip_ref_core.pycmec.output import CMECOutput


@pytest.fixture
def metric_dataset():
    return MetricDataset(
        {
            SourceDatasetType.CMIP6: DatasetCollection(
                datasets=pandas.read_json(Path(__file__).parent / "input_files_tcre.json"),
                slug_column="test",
            ),
        }
    )


def test_update_recipe(metric_dataset):
    # Insert the following code in ZeroEmissionCommitment.update_recipe to
    # save an example input dataframe:
    # input_files.to_json(Path("input_files_tcre.json"), orient='records', indent=4, date_format="iso")
    input_files = metric_dataset[SourceDatasetType.CMIP6].datasets
    recipe = load_recipe("recipe_tcre.yml")
    TransientClimateResponseEmissions().update_recipe(recipe, input_files)
    assert recipe["diagnostics"]["tcre"]["variables"] == {
        "tas_esm-1pctCO2": {
            "short_name": "tas",
            "preprocessor": "global_annual_mean_anomaly",
            "additional_datasets": [
                {
                    "project": "CMIP6",
                    "activity": "C4MIP CDRMIP",
                    "dataset": "MPI-ESM1-2-LR",
                    "ensemble": "r1i1p1f1",
                    "institute": "MPI-M",
                    "exp": "esm-1pctCO2",
                    "grid": "gn",
                    "mip": "Amon",
                    "timerange": "18500116T120000/19141216T120000",
                }
            ],
        },
        "tas_esm-piControl": {
            "short_name": "tas",
            "preprocessor": "global_annual_mean_anomaly",
            "additional_datasets": [
                {
                    "project": "CMIP6",
                    "activity": "CMIP",
                    "dataset": "MPI-ESM1-2-LR",
                    "ensemble": "r1i1p1f1",
                    "institute": "MPI-M",
                    "exp": "esm-piControl",
                    "grid": "gn",
                    "mip": "Amon",
                    "timerange": "18500116T120000/19141216T120000",
                }
            ],
        },
        "fco2antt": {
            "preprocessor": "global_cumulative_sum",
            "additional_datasets": [
                {
                    "project": "CMIP6",
                    "activity": "C4MIP CDRMIP",
                    "dataset": "MPI-ESM1-2-LR",
                    "ensemble": "r1i1p1f1",
                    "institute": "MPI-M",
                    "exp": "esm-1pctCO2",
                    "grid": "gn",
                    "mip": "Amon",
                    "timerange": "18500116T120000/19141216T120000",
                }
            ],
        },
    }


def test_format_output(tmp_path, metric_dataset):
    result_dir = tmp_path
    subdir = result_dir / "work" / "tcre" / "calculate_tcre"
    subdir.mkdir(parents=True)
    tcr = xr.Dataset(
        data_vars={
            "tcre": (["dim0"], np.array([1.0], dtype=np.float32)),
        },
    )
    tcr.to_netcdf(subdir / "tcre.nc")

    metric_args, output_args = TransientClimateResponseEmissions().format_result(
        result_dir,
        metric_dataset=metric_dataset,
        metric_args=CMECMetric.create_template(),
        output_args=CMECOutput.create_template(),
    )

    CMECMetric.model_validate(metric_args)
    assert metric_args["RESULTS"]["MPI-ESM1-2-LR"]["global"]["tcre"] == 1.0
    CMECOutput.model_validate(output_args)
