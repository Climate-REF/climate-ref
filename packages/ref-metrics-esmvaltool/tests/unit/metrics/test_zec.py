from pathlib import Path

import numpy as np
import pandas
import xarray as xr
from cmip_ref_metrics_esmvaltool.metrics import ZeroEmissionCommitment
from cmip_ref_metrics_esmvaltool.recipe import load_recipe


def test_update_recipe():
    # Insert the following code in ZeroEmissionCommitment.update_recipe to
    # save an example input dataframe:
    # input_files.to_json(Path("input_files_zec.json"), indent=4, date_format="iso")
    input_files = pandas.read_json(Path(__file__).parent / "input_files_zec.json")
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


def test_format_output(tmp_path):
    zec = xr.Dataset(
        data_vars={
            "zec": (["dim0"], np.array([-0.11])),
        },
        coords={
            "dataset": ("dim0", np.array([b"abc"])),
        },
    )
    result_dir = tmp_path
    subdir = result_dir / "work" / "zec" / "zec"
    subdir.mkdir(parents=True)
    zec.to_netcdf(subdir / "zec_50.nc")

    output_bundle = ZeroEmissionCommitment().format_result(result_dir)

    assert isinstance(output_bundle, dict)
    assert output_bundle["RESULTS"]["abc"]["global"]["zec"] == -0.11
