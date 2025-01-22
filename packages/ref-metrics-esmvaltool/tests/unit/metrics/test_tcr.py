from pathlib import Path

import numpy as np
import pandas
import xarray as xr
from cmip_ref_metrics_esmvaltool.metrics import TransientClimateResponse
from cmip_ref_metrics_esmvaltool.recipe import load_recipe


def test_update_recipe():
    input_files = pandas.read_json(Path(__file__).parent / "input_files_tcr.json")
    recipe = load_recipe("recipe_tcr.yml")
    TransientClimateResponse().update_recipe(recipe, input_files)
    assert len(recipe["datasets"]) == 2
    assert len(recipe["diagnostics"]) == 1
    assert set(recipe["diagnostics"]["cmip6"]["variables"]) == {"tas"}


def test_format_output(tmp_path):
    tcr = xr.Dataset(
        data_vars={
            "tcr": (["dim0"], np.array([1.0], dtype=np.float32)),
        },
        coords={
            "dataset": ("dim0", np.array([b"abc"])),
        },
    )
    result_dir = tmp_path
    subdir = result_dir / "work" / "cmip6" / "tcr"
    subdir.mkdir(parents=True)
    tcr.to_netcdf(subdir / "tcr.nc")

    output_bundle = TransientClimateResponse().format_result(result_dir)

    assert isinstance(output_bundle, dict)
    assert output_bundle["RESULTS"]["abc"]["global"]["tcr"] == 1.0
