from pathlib import Path

import pandas
import pytest
from climate_ref_esmvaltool.diagnostics import ClimateAtGlobalWarmingLevels
from climate_ref_esmvaltool.recipe import load_recipe

from climate_ref_core.datasets import DatasetCollection, ExecutionDatasetCollection, SourceDatasetType


@pytest.fixture
def metric_dataset():
    return ExecutionDatasetCollection(
        {
            SourceDatasetType.CMIP6: DatasetCollection(
                datasets=pandas.read_json(
                    Path(__file__).parent / "input_files_climate_at_global_warming_levels.json"
                ),
                slug_column="test",
            ),
        }
    )


def test_update_recipe(metric_dataset):
    input_files = {k: v.datasets for k, v in metric_dataset.items()}
    recipe = load_recipe("recipe_calculate_gwl_exceedance_stats.yml")
    ClimateAtGlobalWarmingLevels().update_recipe(recipe, input_files)
    assert "datasets" in recipe
    datasets = recipe["datasets"]
    assert len(datasets) == 6
    assert datasets[0]["exp"] == ["historical", "ssp126"]

    for dataset in datasets:
        # timerange is defined per diagnostic
        assert "timerange" not in datasets
    for diagnostic in recipe["diagnostics"].values():
        for variable in diagnostic["variables"].values():
            assert "preprocessor" in variable
            assert "timerange" in variable
        assert "additional_datasets" not in diagnostic
