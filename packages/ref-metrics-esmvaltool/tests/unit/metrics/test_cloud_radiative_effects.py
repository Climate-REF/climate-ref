from pathlib import Path

import pandas
from cmip_ref_metrics_esmvaltool.metrics import CloudRadiativeEffects
from cmip_ref_metrics_esmvaltool.recipe import load_recipe


def test_update_recipe():
    # Insert the following code in CloudRadiativeEffects.update_recipe to
    # save an example input dataframe:
    # input_files.to_json(Path("input_files_cloud_radiative_effects.json"), indent=4, date_format="iso")
    input_files = pandas.read_json(Path(__file__).parent / "input_files_cloud_radiative_effects.json")
    recipe = load_recipe("ref/recipe_ref_cre.yml")
    CloudRadiativeEffects().update_recipe(recipe, input_files)
    assert recipe["datasets"] == [
        {
            "activity": "CMIP",
            "dataset": "ACCESS-ESM1-5",
            "ensemble": "r1i1p1f1",
            "exp": "historical",
            "grid": "gn",
            "institute": "CSIRO",
            "mip": "Amon",
            "project": "CMIP6",
        },
    ]
    assert recipe["timerange_for_models"] == "20010101T000000/20141216T120000"
