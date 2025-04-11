from pathlib import Path

import numpy as np
import pandas
import pytest
import xarray as xr
from cmip_ref_metrics_esmvaltool.metrics import ZeroEmissionCommitment
from cmip_ref_metrics_esmvaltool.recipe import load_recipe

from cmip_ref_core.datasets import DatasetCollection, MetricDataset, SourceDatasetType
from cmip_ref_core.pycmec.metric import CMECMetric
from cmip_ref_core.pycmec.output import CMECOutput


@pytest.fixture
def metric_dataset():
    return MetricDataset(
        {
            SourceDatasetType.CMIP6: DatasetCollection(
                datasets=pandas.read_json(Path(__file__).parent / "input_files_zec.json"),
                slug_column="test",
            ),
        }
    )


def test_update_recipe(metric_dataset):
    # Insert the following code in ZeroEmissionCommitment.update_recipe to
    # save an example input dataframe:
    # input_files.to_json(Path("input_files_zec.json"), indent=4, date_format="iso")
    input_files = metric_dataset[SourceDatasetType.CMIP6].datasets
    recipe = load_recipe("recipe_zec.yml")
    ZeroEmissionCommitment().update_recipe(recipe, input_files)
    assert len(recipe["diagnostics"]) == 1
    assert recipe["diagnostics"]["zec"]["variables"] == {
        "tas_base": {
            "short_name": "tas",
            "preprocessor": "anomaly_base",
            "additional_datasets": [
                {
                    "project": "CMIP6",
                    "activity": "CMIP",
                    "dataset": "ACCESS-ESM1-5",
                    "ensemble": "r1i1p1f1",
                    "institute": "CSIRO",
                    "exp": "1pctCO2",
                    "grid": "gn",
                    "mip": "Amon",
                    "timerange": "01580116T120000.000/01780116T120000.000",
                },
            ],
        },
        "tas": {
            "preprocessor": "spatial_mean",
            "additional_datasets": [
                {
                    "project": "CMIP6",
                    "activity": "C4MIP CDRMIP",
                    "dataset": "ACCESS-ESM1-5",
                    "ensemble": "r1i1p1f1",
                    "institute": "CSIRO",
                    "exp": "esm-1pct-brch-1000PgC",
                    "grid": "gn",
                    "mip": "Amon",
                    "timerange": "01680116T120000.000/02681216T120000.000",
                },
            ],
        },
    }


def test_format_output(tmp_path, metric_dataset):
    result_dir = tmp_path
    subdir = result_dir / "work" / "zec" / "zec"
    subdir.mkdir(parents=True)
    zec = xr.Dataset(
        data_vars={
            "zec": (["dim0"], np.array([-0.11])),
        },
    )
    zec.to_netcdf(subdir / "zec_50.nc")

    metric_args, output_args = ZeroEmissionCommitment().format_result(
        result_dir,
        metric_dataset=metric_dataset,
        metric_args=CMECMetric.create_template(),
        output_args=CMECOutput.create_template(),
    )

    CMECMetric.model_validate(metric_args)
    assert metric_args["RESULTS"]["ACCESS-ESM1-5"]["global"]["zec"] == -0.11
    CMECOutput.model_validate(output_args)
