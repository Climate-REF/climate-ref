from pathlib import Path

import pandas
from climate_ref_esmvaltool.diagnostics import ENSOBasicClimatology, ENSOCharacteristics
from climate_ref_esmvaltool.recipe import load_recipe

from climate_ref_core.datasets import SourceDatasetType


def test_enso_basicclimatology_update_recipe():
    # Insert the following code in CloudRadiativeEffects.update_recipe to
    # save an example input dataframe:
    # input_files.to_json(Path("input_files_enso_climatology.json"), indent=4, date_format="iso")
    input_files = {
        SourceDatasetType.CMIP6: pandas.read_json(
            Path(__file__).parent / "input_files_enso_climatology.json"
        ),
    }
    recipe = load_recipe("ref/recipe_enso_basicclimatology.yml")
    ENSOBasicClimatology().update_recipe(recipe, input_files)
    assert "datasets" not in recipe
    for diagnostic in recipe["diagnostics"].values():
        for variable in diagnostic["variables"].values():
            assert variable["additional_datasets"][-1]["dataset"] == "ACCESS-ESM1-5"


def test_enso_characteristics_update_recipe():
    # Insert the following code in CloudRadiativeEffects.update_recipe to
    # save an example input dataframe:
    # input_files.to_json(Path("input_files_enso_characteristics.json"), indent=4, date_format="iso")
    input_files = {
        SourceDatasetType.CMIP6: pandas.read_json(
            Path(__file__).parent / "input_files_enso_characteristics.json"
        ),
    }
    recipe = load_recipe("ref/recipe_enso_characteristics.yml")
    ENSOCharacteristics().update_recipe(recipe, input_files)
    assert recipe["datasets"] == [
        {
            "activity": "CMIP",
            "dataset": "ACCESS-ESM1-5",
            "ensemble": "r1i1p1f1",
            "exp": "historical",
            "grid": "gn",
            "institute": "CSIRO",
            "mip": "Omon",
            "project": "CMIP6",
            "timerange": "18500116T120000/20141216T120000",
        },
        {
            "dataset": "TROPFLUX",
            "project": "OBS6",
            "tier": 2,
            "type": "reanaly",
            "version": "v1",
        },
    ]
