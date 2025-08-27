from pathlib import Path

import pandas
from climate_ref_esmvaltool.diagnostics import SeaIceSensitivity
from climate_ref_esmvaltool.recipe import load_recipe

from climate_ref_core.datasets import SourceDatasetType


def test_update_recipe():
    # Insert the following code in SeaIceSensitivity.update_recipe to
    # save an example input dataframe:
    # input_files.to_json(Path("input_files_sea_ice_sensitivity.json"), indent=4, date_format="iso")
    input_files = {
        SourceDatasetType.CMIP6: pandas.read_json(
            Path(__file__).parent / "input_files_sea_ice_sensitivity.json"
        )
    }
    recipe = load_recipe(SeaIceSensitivity.base_recipe)
    SeaIceSensitivity().update_recipe(recipe, input_files)
    assert recipe["datasets"] == [
        {
            "project": "CMIP6",
            "activity": "CMIP",
            "dataset": "CanESM5",
            "ensemble": "r1i1p1f1",
            "institute": "CCCma",
            "exp": "historical",
            "grid": "gn",
            "timerange": "1979/2014",
        },
        {
            "project": "CMIP6",
            "activity": "CMIP",
            "dataset": "ACCESS-ESM1-5",
            "ensemble": "r1i1p1f1",
            "institute": "CSIRO",
            "exp": "historical",
            "grid": "gn",
            "timerange": "1979/2014",
        },
        {
            "project": "CMIP6",
            "activity": "CMIP",
            "dataset": "HadGEM3-GC31-LL",
            "ensemble": "r1i1p1f3",
            "institute": "MOHC",
            "exp": "historical",
            "grid": "gn",
            "timerange": "1979/2014",
        },
    ]
