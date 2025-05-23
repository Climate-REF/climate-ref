from pathlib import Path

import pandas
from climate_ref_esmvaltool.diagnostics import SeaIceAreaBasic
from climate_ref_esmvaltool.recipe import load_recipe


def test_update_recipe():
    input_files = pandas.read_json(Path(__file__).parent / "input_files_sea_ice_area.json")
    recipe = load_recipe("ref/recipe_ref_sea_ice_area_basic.yml")
    SeaIceAreaBasic().update_recipe(recipe, input_files)
    assert len(recipe["datasets"]) == 1
    dataset = recipe["datasets"][0]
    assert dataset["dataset"] == input_files.iloc[0].source_id
    assert "timerange" not in dataset
    assert len(recipe["diagnostics"]) == 2
