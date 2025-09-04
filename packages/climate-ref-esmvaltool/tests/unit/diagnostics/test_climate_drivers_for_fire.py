from pathlib import Path

import pandas
import pytest
from climate_ref_esmvaltool.diagnostics.climate_drivers_for_fire import ClimateDriversForFire
from climate_ref_esmvaltool.recipe import load_recipe

from climate_ref_core.datasets import (
    DatasetCollection,
    ExecutionDatasetCollection,
    SourceDatasetType,
)


@pytest.fixture
def metric_dataset():
    return ExecutionDatasetCollection(
        {
            SourceDatasetType.CMIP6: DatasetCollection(
                datasets=pandas.read_json(
                    Path(__file__).parent / "input_files_climate_drivers_for_fire.json"
                ),
                slug_column="test",
            ),
        }
    )


def test_update_recipe(metric_dataset):
    input_files = {k: v.datasets for k, v in metric_dataset.items()}
    recipe = load_recipe(ClimateDriversForFire.base_recipe)
    ClimateDriversForFire().update_recipe(recipe, input_files)
    assert len(recipe["diagnostics"]) == 1
    assert recipe["datasets"] == [
        {
            "activity": "CMIP",
            "dataset": "ACCESS-ESM1-5",
            "end_year": 2014,
            "ensemble": "r1i1p1f1",
            "exp": "historical",
            "grid": "gn",
            "institute": "CSIRO",
            "project": "CMIP6",
            "start_year": 2013,
        },
    ]
