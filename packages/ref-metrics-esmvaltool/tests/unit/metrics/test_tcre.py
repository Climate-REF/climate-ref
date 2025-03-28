from pathlib import Path

import numpy as np
import pandas
import xarray as xr
from cmip_ref_metrics_esmvaltool.metrics import TransientClimateResponseEmissions
from cmip_ref_metrics_esmvaltool.recipe import load_recipe


def test_update_recipe():
    # Insert the following code in ZeroEmissionCommitment.update_recipe to
    # save an example input dataframe:
    # input_files.to_json(Path("input_files_tcre.json"), orient='records', indent=4, date_format="iso")
    input_files = pandas.read_json(Path(__file__).parent / "input_files_tcre.json")
    recipe = load_recipe("recipe_tcre.yml")
    TransientClimateResponseEmissions().update_recipe(recipe, input_files)
    assert recipe["diagnostics"]["tcre"]["variables"] == {
        "tas_esm-1pctCO2": {
            "short_name": "tas",
            "preprocessor": "global_annual_mean_anomaly",
            "additional_datasets": [
                {
                    "project": "CMIP6",
                    "activity": "C4MIP CDRMIP",
                    "dataset": "MPI-ESM1-2-LR",
                    "ensemble": "r1i1p1f1",
                    "institute": "MPI-M",
                    "exp": "esm-1pctCO2",
                    "grid": "gn",
                    "mip": "Amon",
                    "timerange": "18500116T120000/19141216T120000",
                }
            ],
        },
        "tas_esm-piControl": {
            "short_name": "tas",
            "preprocessor": "global_annual_mean_anomaly",
            "additional_datasets": [
                {
                    "project": "CMIP6",
                    "activity": "CMIP",
                    "dataset": "MPI-ESM1-2-LR",
                    "ensemble": "r1i1p1f1",
                    "institute": "MPI-M",
                    "exp": "esm-piControl",
                    "grid": "gn",
                    "mip": "Amon",
                    "timerange": "18500116T120000/19141216T120000",
                }
            ],
        },
        "fco2antt": {
            "preprocessor": "global_cumulative_sum",
            "additional_datasets": [
                {
                    "project": "CMIP6",
                    "activity": "C4MIP CDRMIP",
                    "dataset": "MPI-ESM1-2-LR",
                    "ensemble": "r1i1p1f1",
                    "institute": "MPI-M",
                    "exp": "esm-1pctCO2",
                    "grid": "gn",
                    "mip": "Amon",
                    "timerange": "18500116T120000/19141216T120000",
                }
            ],
        },
    }


def test_format_output(tmp_path):
    tcr = xr.Dataset(
        data_vars={
            "tcre": (["dim0"], np.array([1.0], dtype=np.float32)),
        },
        coords={
            "dataset": ("dim0", np.array([b"abc"])),
        },
    )
    result_dir = tmp_path
    subdir = result_dir / "work" / "tcre" / "calculate_tcre"
    subdir.mkdir(parents=True)
    tcr.to_netcdf(subdir / "tcre.nc")

    output_bundle = TransientClimateResponseEmissions().format_result(result_dir)

    assert isinstance(output_bundle, dict)
    assert output_bundle["RESULTS"]["abc"]["global"]["tcre"] == 1.0
