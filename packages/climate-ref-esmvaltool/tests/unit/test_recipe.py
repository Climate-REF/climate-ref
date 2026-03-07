from pathlib import Path

import cftime
import pandas as pd
import pytest
from climate_ref_esmvaltool.recipe import as_facets, get_child_and_parent_dataset, prepare_climate_data


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


def test_as_facets_uses_activity_from_instance_id():
    """activity_id can be space-separated (e.g. 'C4MIP CDRMIP').

    as_facets must derive activity from instance_id (which uses the primary
    activity only) so the facet matches the directory structure created by
    prepare_climate_data.
    """
    group = pd.DataFrame(
        {
            "instance_id": [
                "CMIP6.C4MIP.CSIRO.ACCESS-ESM1-5.esm-1pct-brch-1000PgC.r1i1p1f1.Amon.tas.gn.v20191206",
            ],
            "activity_id": ["C4MIP CDRMIP"],
            "source_id": ["ACCESS-ESM1-5"],
            "member_id": ["r1i1p1f1"],
            "institution_id": ["CSIRO"],
            "experiment_id": ["esm-1pct-brch-1000PgC"],
            "grid_label": ["gn"],
            "table_id": ["Amon"],
            "variable_id": ["tas"],
            "variant_label": ["r1i1p1f1"],
            "start_time": [pd.Timestamp("1850-01-16")],
            "end_time": [pd.Timestamp("1989-12-16")],
        }
    )

    facets = as_facets(group)

    # Must use "C4MIP" (from instance_id), NOT "C4MIP CDRMIP" (from activity_id)
    assert facets["activity"] == "C4MIP"
    assert facets["project"] == "CMIP6"
    assert facets["dataset"] == "ACCESS-ESM1-5"


def test_get_child_and_parent_dataset_multi_file_start_time():
    """child_start must be the earliest start_time across all file entries.

    When a dataset is split across multiple files (time slabs), iloc[0] may
    not be the row with the earliest start_time, leading to wrong timeranges.
    """
    df = pd.DataFrame(
        {
            "instance_id": [
                # Child: 4 time slabs, intentionally out of order
                "CMIP6.C4MIP.MOHC.UKESM1-0-LL.esm-1pct-brch-1000PgC.r2i1p1f2.Amon.tas.gn.v20200210",
                "CMIP6.C4MIP.MOHC.UKESM1-0-LL.esm-1pct-brch-1000PgC.r2i1p1f2.Amon.tas.gn.v20200210",
                "CMIP6.C4MIP.MOHC.UKESM1-0-LL.esm-1pct-brch-1000PgC.r2i1p1f2.Amon.tas.gn.v20200210",
                "CMIP6.C4MIP.MOHC.UKESM1-0-LL.esm-1pct-brch-1000PgC.r2i1p1f2.Amon.tas.gn.v20200210",
                # Parent: 2 time slabs
                "CMIP6.CMIP.MOHC.UKESM1-0-LL.1pctCO2.r2i1p1f2.Amon.tas.gn.v20200210",
                "CMIP6.CMIP.MOHC.UKESM1-0-LL.1pctCO2.r2i1p1f2.Amon.tas.gn.v20200210",
            ],
            "activity_id": ["C4MIP CDRMIP"] * 4 + ["CMIP"] * 2,
            "branch_time_in_child": [23760.0] * 4 + [0.0] * 2,
            "branch_time_in_parent": [23760.0] * 4 + [97200.0] * 2,
            "experiment_id": ["esm-1pct-brch-1000PgC"] * 4 + ["1pctCO2"] * 2,
            "grid_label": ["gn"] * 6,
            "institution_id": ["MOHC"] * 6,
            "source_id": ["UKESM1-0-LL"] * 6,
            "table_id": ["Amon"] * 6,
            "variable_id": ["tas"] * 6,
            "variant_label": ["r2i1p1f2"] * 6,
            "member_id": ["r2i1p1f2"] * 6,
            "start_time": [
                # Child time slabs out of order - first row is latest slab
                cftime.datetime(2150, 1, 1, calendar="360_day"),
                cftime.datetime(1950, 1, 1, calendar="360_day"),
                cftime.datetime(1916, 1, 1, calendar="360_day"),
                cftime.datetime(2050, 1, 16, calendar="360_day"),
                # Parent
                cftime.datetime(1850, 1, 1, calendar="360_day"),
                cftime.datetime(1950, 1, 16, calendar="360_day"),
            ],
            "end_time": [
                cftime.datetime(2181, 12, 30, calendar="360_day"),
                cftime.datetime(2049, 12, 30, calendar="360_day"),
                cftime.datetime(1949, 12, 30, calendar="360_day"),
                cftime.datetime(2149, 12, 16, calendar="360_day"),
                cftime.datetime(1949, 12, 30, calendar="360_day"),
                cftime.datetime(1999, 12, 16, calendar="360_day"),
            ],
            "time_units": ["days since 1850-01-01"] * 6,
            "calendar": ["360_day"] * 6,
            "path": [f"file{i}.nc" for i in range(6)],
            "version": ["v20200210"] * 6,
        }
    )

    child, parent = get_child_and_parent_dataset(
        df,
        parent_experiment="1pctCO2",
        child_duration_in_years=100,
        parent_offset_in_years=-10,
        parent_duration_in_years=20,
    )

    # child_start should be 1916 (earliest), not 2150 (iloc[0])
    assert child["timerange"] == "1916/2015"
    # parent timerange should be based on 1916, not 2150
    assert parent["timerange"] == "1906/1925"


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
