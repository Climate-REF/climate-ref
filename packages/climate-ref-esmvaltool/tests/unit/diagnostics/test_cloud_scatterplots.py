from pathlib import Path

import pandas
from climate_ref_esmvaltool.diagnostics.cloud_scatterplots import (
    CloudScatterplotCltSwcre,
    CloudScatterplotsReference,
)
from climate_ref_esmvaltool.recipe import load_recipe

from climate_ref_core.datasets import SourceDatasetType


def test_update_cmip6_recipe():
    # Insert the following code in update_recipe to save an example input dataframe:
    # input_files[SourceDatasetType.CMIP6].to_json(
    #     Path("input_files_cloud_scatterplots_cmip6.json"),
    #     indent=4,
    #     date_format="iso",
    # )
    input_files = {
        SourceDatasetType.CMIP6: pandas.read_json(
            Path(__file__).parent / "input_files_cloud_scatterplots_cmip6.json"
        )
    }
    recipe = load_recipe("ref/recipe_ref_scatterplot.yml")
    CloudScatterplotCltSwcre().update_recipe(recipe, input_files)
    assert "datasets" not in recipe
    assert len(recipe["diagnostics"]) == 1
    diagnostic = recipe["diagnostics"]["plot_joint_clt_swcre_model"]
    assert set(diagnostic["variables"]) == {"clt", "swcre"}
    for variable in diagnostic["variables"].values():
        assert "additional_datasets" not in variable
    assert len(diagnostic["additional_datasets"]) == 1
    assert diagnostic["scripts"]["plot"]["suptitle"].startswith("CMIP6")


def test_update_reference_recipe():
    # Insert the following code in update_recipe to save an example input dataframe:
    # input_files[SourceDatasetType.obs4MIPs].to_json(
    #     Path("input_files_cloud_scatterplots_obs4mips.json"),
    #     indent=4,
    #     date_format="iso",
    #     orient="records",
    # )
    input_files = {
        SourceDatasetType.obs4MIPs: pandas.read_json(
            Path(__file__).parent / "input_files_cloud_scatterplots_obs4mips.json"
        )
    }
    recipe = load_recipe("ref/recipe_ref_scatterplot.yml")
    CloudScatterplotsReference().update_recipe(recipe, input_files)
    assert "datasets" not in recipe
    assert len(recipe["diagnostics"]) == 4

    diagnostic = recipe["diagnostics"]["plot_joint_cli_ta_ref"]
    assert "additional_datasets" not in diagnostic
    assert set(diagnostic["variables"]) == {"cli", "ta"}
    for variable in diagnostic["variables"].values():
        assert len(variable["additional_datasets"]) == 1
    assert diagnostic["scripts"]["plot"]["suptitle"] == "CALIPSO-ICECLOUD / ERA-5 2007/2015"

    diagnostic = recipe["diagnostics"]["plot_joint_clwvi_pr_ref"]
    diagnostic["variables"]["pr"]["additional_datasets"][0]["dataset"] == "GPCP-V2.3"
