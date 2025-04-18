from pathlib import Path

import pandas
from cmip_ref_metrics_esmvaltool.metrics import SeaIceAreaSeasonalCycle
from cmip_ref_metrics_esmvaltool.recipe import load_recipe


def test_update_recipe():
    input_files = pandas.read_json(Path(__file__).parent / "input_files_sea_ice_area.json")
    recipe = load_recipe("ref/recipe_ref_sea_ice_area_basic.yml")
    SeaIceAreaSeasonalCycle().update_recipe(recipe, input_files)
    assert len(recipe["datasets"]) == 1
    dataset = recipe["datasets"][0]
    assert dataset["dataset"] == input_files.iloc[0].source_id
    assert dataset["timerange"] == "1995/2014"
    assert len(recipe["diagnostics"]) == 2
