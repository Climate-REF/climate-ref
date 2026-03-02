from pathlib import Path

import cftime
import pandas as pd
import pytest
from climate_ref_esmvaltool.recipe import get_child_and_parent_dataset, prepare_climate_data


def test_get_child_and_parent_dataset():
    # Code to extract the dataframe below from the TCR diagnostic:
    # df[df.variable_id == "tas"].to_dict(orient="list")
    df = pd.DataFrame(
        {
            "instance_id": [
                "CMIP6.CMIP.CCCma.CanESM5.1pctCO2.r1i1p1f1.Amon.tas.gn.v20190429",
                "CMIP6.CMIP.CCCma.CanESM5.piControl.r1i1p1f1.Amon.tas.gn.v20190429",
            ],
            "activity_id": ["CMIP", "CMIP"],
            "branch_time_in_child": [0.0, 1223115.0],
            "branch_time_in_parent": [1223115.0, 1223115.0],
            "experiment_id": ["1pctCO2", "piControl"],
            "grid_label": ["gn", "gn"],
            "institution_id": ["CCCma", "CCCma"],
            "source_id": ["CanESM5", "CanESM5"],
            "table_id": ["Amon", "Amon"],
            "variable_id": ["tas", "tas"],
            "variant_label": ["r1i1p1f1", "r1i1p1f1"],
            "member_id": ["r1i1p1f1", "r1i1p1f1"],
            "start_time": [
                cftime.datetime(1850, 1, 16, 12, 0, calendar="365_day"),
                cftime.datetime(5201, 1, 16, 12, 0, calendar="365_day"),
            ],
            "end_time": [
                cftime.datetime(1989, 12, 16, 12, 0, calendar="365_day"),
                cftime.datetime(5340, 12, 16, 12, 0, calendar="365_day"),
            ],
            "time_units": ["days since 1850-01-01", "days since 1850-01-01"],
            "calendar": ["365_day", "365_day"],
            "path": [
                "tas_Amon_CanESM5_1pctCO2_r1i1p1f1_gn_185001-198912.nc",
                "tas_Amon_CanESM5_piControl_r1i1p1f1_gn_520101-534012.nc",
            ],
            "version": ["v20190429", "v20190429"],
        }
    )

    child, parent = get_child_and_parent_dataset(
        df,
        parent_experiment="piControl",
        child_duration_in_years=140,
        parent_offset_in_years=0,
        parent_duration_in_years=140,
    )

    assert child == {
        "project": "CMIP6",
        "activity": "CMIP",
        "dataset": "CanESM5",
        "ensemble": "r1i1p1f1",
        "institute": "CCCma",
        "exp": "1pctCO2",
        "grid": "gn",
        "mip": "Amon",
        "timerange": "1850/1989",
    }
    assert parent == {
        "project": "CMIP6",
        "activity": "CMIP",
        "dataset": "CanESM5",
        "ensemble": "r1i1p1f1",
        "institute": "CCCma",
        "exp": "piControl",
        "grid": "gn",
        "mip": "Amon",
        "timerange": "5201/5340",
    }


@pytest.mark.parametrize(
    ("datasets", "expected"),
    [
        (
            pd.DataFrame(
                {
                    "instance_id": [
                        "CMIP6.ScenarioMIP.CCCma.CanESM5.ssp126.r1i1p1f1.Amon.pr.gn.v20190429",
                    ],
                    "source_id": ["CanESM5"],
                    "path": [
                        "pr_Amon_CanESM5_ssp126_r1i1p1f1_gn_210101-230012.nc",
                    ],
                }
            ),
            [
                "CMIP6/ScenarioMIP/CCCma/CanESM5/ssp126/r1i1p1f1/Amon/pr/gn/v20190429/pr_Amon_CanESM5_ssp126_r1i1p1f1_gn_210101-230012.nc",
            ],
        ),
        (
            pd.DataFrame(
                {
                    "instance_id": [
                        "obs4MIPs.obs4MIPs.ECMWF.ERA-5.ta.gn.v20250220",
                        "obs4MIPs.obs4MIPs.ECMWF.ERA-5.ta.gn.v20250220",
                    ],
                    "source_id": ["ERA-5", "ERA-5"],
                    "path": [
                        "ta_mon_ERA-5_PCMDI_gn_200701-200712.nc",
                        "ta_mon_ERA-5_PCMDI_gn_200801-200812.nc",
                    ],
                }
            ),
            [
                "obs4MIPs/ERA-5/v20250220/ta_mon_ERA-5_PCMDI_gn_200701-200712.nc",
                "obs4MIPs/ERA-5/v20250220/ta_mon_ERA-5_PCMDI_gn_200801-200812.nc",
            ],
        ),
    ],
)
def test_prepare_climate_data(tmp_path, datasets, expected):
    climate_data_dir = tmp_path / "climate_data"
    climate_data_dir.mkdir()

    source_data_dir = tmp_path / "source_data"
    source_data_dir.mkdir()

    datasets["path"] = [f"{source_data_dir / path}" for path in datasets["path"]]
    expected = [f"{climate_data_dir / path}" for path in expected]
    for path in datasets["path"]:
        Path(path).touch()

    prepare_climate_data(datasets, climate_data_dir)

    for source_path, symlink in zip(datasets["path"], expected):
        assert Path(symlink).is_symlink()
        assert Path(symlink).resolve() == Path(source_path).resolve()
