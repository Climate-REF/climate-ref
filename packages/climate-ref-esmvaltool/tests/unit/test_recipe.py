import datetime
from pathlib import Path

import pandas as pd
import pytest
import xarray as xr
from climate_ref_esmvaltool.recipe import get_child_and_parent_dataset, prepare_climate_data


def test_get_child_and_parent_dataset(tmp_path):
    # Code to extract the dataframe below from the TCR diagnostic:
    # df[df.variable_id == "tas"].to_dict(orient="list")
    df = pd.DataFrame(
        {
            "instance_id": [
                "CMIP6.CMIP.CCCma.CanESM5.1pctCO2.r1i1p1f1.Amon.tas.gn.v20190429",
                "CMIP6.CMIP.CCCma.CanESM5.piControl.r1i1p1f1.Amon.tas.gn.v20190429",
            ],
            "activity_id": ["CMIP", "CMIP"],
            "branch_method": ["Spin-up documentation", "Spin-up documentation"],
            "branch_time_in_child": [0.0, 1223115.0],
            "branch_time_in_parent": [1223115.0, 1223115.0],
            "experiment": ["1 percent per year increase in CO2", "pre-industrial control"],
            "experiment_id": ["1pctCO2", "piControl"],
            "grid_label": ["gn", "gn"],
            "institution_id": ["CCCma", "CCCma"],
            "nominal_resolution": ["500 km", "500 km"],
            "parent_activity_id": ["CMIP", "CMIP"],
            "parent_experiment_id": ["piControl", "piControl-spinup"],
            "parent_source_id": ["CanESM5", "CanESM5"],
            "parent_time_units": ["days since 1850-01-01 0:0:0.0", "days since 1850-01-01 0:0:0.0"],
            "parent_variant_label": ["r1i1p1f1", "r1i1p1f1"],
            "product": ["model-output", "model-output"],
            "realm": ["atmos", "atmos"],
            "source_id": ["CanESM5", "CanESM5"],
            "source_type": ["AOGCM", "AOGCM"],
            "sub_experiment": ["none", "none"],
            "sub_experiment_id": ["none", "none"],
            "table_id": ["Amon", "Amon"],
            "variable_id": ["tas", "tas"],
            "variant_label": ["r1i1p1f1", "r1i1p1f1"],
            "member_id": ["r1i1p1f1", "r1i1p1f1"],
            "standard_name": ["air_temperature", "air_temperature"],
            "long_name": ["Near-Surface Air Temperature", "Near-Surface Air Temperature"],
            "units": ["K", "K"],
            "vertical_levels": [1, 1],
            "start_time": [datetime.datetime(1850, 1, 16, 12, 0), datetime.datetime(5201, 1, 16, 12, 0)],
            "end_time": [datetime.datetime(1989, 12, 16, 12, 0), datetime.datetime(5340, 12, 16, 12, 0)],
            "time_range": [
                "1850-01-16 12:00:00-1989-12-16 12:00:00",
                "5201-01-16 12:00:00-5340-12-16 12:00:00",
            ],
            "path": [
                f"{tmp_path}/tas_Amon_CanESM5_1pctCO2_r1i1p1f1_gn_185001-198912.nc",
                f"{tmp_path}/tas_Amon_CanESM5_piControl_r1i1p1f1_gn_520101-534012.nc",
            ],
            "version": ["v20190429", "v20190429"],
            "finalised": [True, True],
        }
    )

    # Code to generate the file content below:
    # ds = xr.open_dataset(
    #     "tas_Amon_CanESM5_1pctCO2_r1i1p1f1_gn_185001-198912.nc",
    #     decode_times=False,
    # )
    # ds.isel(time=slice(0, 1), lat=slice(0,1), lon=slice(0,1)).to_dict()
    xr.Dataset.from_dict(
        {
            "coords": {
                "time": {
                    "dims": ("time",),
                    "attrs": {
                        "bounds": "time_bnds",
                        "axis": "T",
                        "long_name": "time",
                        "standard_name": "time",
                        "units": "days since 1850-01-01",
                        "calendar": "365_day",
                    },
                    "data": [15.5],
                },
                "height": {
                    "dims": (),
                    "attrs": {
                        "units": "m",
                        "axis": "Z",
                        "positive": "up",
                        "long_name": "height",
                        "standard_name": "height",
                    },
                    "data": 2.0,
                },
                "lat": {
                    "dims": ("lat",),
                    "attrs": {
                        "bounds": "lat_bnds",
                        "units": "degrees_north",
                        "axis": "Y",
                        "long_name": "Latitude",
                        "standard_name": "latitude",
                    },
                    "data": [-90.0],
                },
                "lon": {
                    "dims": ("lon",),
                    "attrs": {
                        "bounds": "lon_bnds",
                        "units": "degrees_east",
                        "axis": "X",
                        "long_name": "Longitude",
                        "standard_name": "longitude",
                    },
                    "data": [0.0],
                },
            },
            "attrs": {
                "CCCma_model_hash": "3dedf95315d603326fde4f5340dc0519d80d10c0",
                "CCCma_parent_runid": "rc3-pictrl",
                "CCCma_pycmor_hash": "33c30511acc319a98240633965a04ca99c26427e",
                "CCCma_runid": "rc3.1-1ppy",
                "Conventions": "CF-1.7 CMIP-6.2",
                "YMDH_branch_time_in_child": "1850:01:01:00",
                "YMDH_branch_time_in_parent": "5201:01:01:00",
                "activity_id": "CMIP",
                "branch_method": "Spin-up documentation",
                "branch_time_in_child": 0.0,
                "branch_time_in_parent": 1223115.0,
                "contact": "ec.cccma.info-info.ccmac.ec@canada.ca",
                "creation_date": "2019-04-30T17:28:14Z",
                "data_specs_version": "01.00.29",
                "experiment": "1 percent per year increase in CO2",
                "experiment_id": "1pctCO2",
                "external_variables": "areacella",
                "forcing_index": 1,
                "frequency": "mon",
                "further_info_url": "https://furtherinfo.es-doc.org/CMIP6.CCCma.CanESM5.1pctCO2.none.r1i1p1f1",
                "grid_label": "gn",
                "initialization_index": 1,
                "institution_id": "CCCma",
                "mip_era": "CMIP6",
                "nominal_resolution": "500 km",
                "parent_activity_id": "CMIP",
                "parent_experiment_id": "piControl",
                "parent_mip_era": "CMIP6",
                "parent_source_id": "CanESM5",
                "parent_time_units": "days since 1850-01-01 0:0:0.0",
                "parent_variant_label": "r1i1p1f1",
                "physics_index": 1,
                "product": "model-output",
                "realization_index": 1,
                "realm": "atmos",
                "source_id": "CanESM5",
                "source_type": "AOGCM",
                "sub_experiment": "none",
                "sub_experiment_id": "none",
                "table_id": "Amon",
                "title": "CanESM5 output prepared for CMIP6",
                "tracking_id": "hdl:21.14100/3beb38c2-d30a-4ed6-8198-b7b5182cab28",
                "variable_id": "tas",
                "variant_label": "r1i1p1f1",
                "version": "v20190429",
                "cmor_version": "3.4.0",
            },
            "dims": {"time": 1, "lat": 1, "lon": 1, "bnds": 2},
            "data_vars": {
                "tas": {
                    "dims": ("time", "lat", "lon"),
                    "attrs": {
                        "standard_name": "air_temperature",
                        "long_name": "Near-Surface Air Temperature",
                        "units": "K",
                        "original_name": "ST",
                        "cell_methods": "area: time: mean",
                        "cell_measures": "area: areacella",
                    },
                    "data": [[[247.98342895507812]]],
                },
                "lat_bnds": {"dims": ("lat", "bnds"), "attrs": {}, "data": [[-90.0, -85.0]]},
                "lon_bnds": {"dims": ("lon", "bnds"), "attrs": {}, "data": [[-5.0, 5.0]]},
                "time_bnds": {"dims": ("time", "bnds"), "attrs": {}, "data": [[0.0, 31.0]]},
            },
        }
    ).to_netcdf(tmp_path / "tas_Amon_CanESM5_1pctCO2_r1i1p1f1_gn_185001-198912.nc")

    xr.Dataset.from_dict(
        {
            "coords": {
                "time": {
                    "dims": ("time",),
                    "attrs": {
                        "bounds": "time_bnds",
                        "axis": "T",
                        "long_name": "time",
                        "standard_name": "time",
                        "units": "days since 1850-01-01",
                        "calendar": "365_day",
                    },
                    "data": [1223130.5],
                },
                "height": {
                    "dims": (),
                    "attrs": {
                        "units": "m",
                        "axis": "Z",
                        "positive": "up",
                        "long_name": "height",
                        "standard_name": "height",
                    },
                    "data": 2.0,
                },
                "lat": {
                    "dims": ("lat",),
                    "attrs": {
                        "bounds": "lat_bnds",
                        "units": "degrees_north",
                        "axis": "Y",
                        "long_name": "Latitude",
                        "standard_name": "latitude",
                    },
                    "data": [-90.0],
                },
                "lon": {
                    "dims": ("lon",),
                    "attrs": {
                        "bounds": "lon_bnds",
                        "units": "degrees_east",
                        "axis": "X",
                        "long_name": "Longitude",
                        "standard_name": "longitude",
                    },
                    "data": [0.0],
                },
            },
            "attrs": {
                "CCCma_model_hash": "24718c8346665b218729640ffe79d263b76456c4",
                "CCCma_parent_runid": "rc3-pictrl",
                "CCCma_pycmor_hash": "33c30511acc319a98240633965a04ca99c26427e",
                "CCCma_runid": "rc3.1-pictrl",
                "Conventions": "CF-1.7 CMIP-6.2",
                "YMDH_branch_time_in_child": "5201:01:01:00",
                "YMDH_branch_time_in_parent": "5201:01:01:00",
                "activity_id": "CMIP",
                "branch_method": "Spin-up documentation",
                "branch_time_in_child": 1223115.0,
                "branch_time_in_parent": 1223115.0,
                "contact": "ec.cccma.info-info.ccmac.ec@canada.ca",
                "creation_date": "2019-04-30T17:18:16Z",
                "data_specs_version": "01.00.29",
                "experiment": "pre-industrial control",
                "experiment_id": "piControl",
                "external_variables": "areacella",
                "forcing_index": 1,
                "frequency": "mon",
                "further_info_url": "https://furtherinfo.es-doc.org/CMIP6.CCCma.CanESM5.piControl.none.r1i1p1f1",
                "grid_label": "gn",
                "initialization_index": 1,
                "institution_id": "CCCma",
                "mip_era": "CMIP6",
                "nominal_resolution": "500 km",
                "parent_activity_id": "CMIP",
                "parent_experiment_id": "piControl-spinup",
                "parent_mip_era": "CMIP6",
                "parent_source_id": "CanESM5",
                "parent_time_units": "days since 1850-01-01 0:0:0.0",
                "parent_variant_label": "r1i1p1f1",
                "physics_index": 1,
                "product": "model-output",
                "realization_index": 1,
                "realm": "atmos",
                "source_id": "CanESM5",
                "source_type": "AOGCM",
                "sub_experiment": "none",
                "sub_experiment_id": "none",
                "table_id": "Amon",
                "title": "CanESM5 output prepared for CMIP6",
                "tracking_id": "hdl:21.14100/cd5624dc-52ce-494b-9a12-bd3e6baaf468",
                "variable_id": "tas",
                "variant_label": "r1i1p1f1",
                "version": "v20190429",
                "cmor_version": "3.4.0",
            },
            "dims": {"time": 1, "lat": 1, "lon": 1, "bnds": 2},
            "data_vars": {
                "tas": {
                    "dims": ("time", "lat", "lon"),
                    "attrs": {
                        "standard_name": "air_temperature",
                        "long_name": "Near-Surface Air Temperature",
                        "units": "K",
                        "original_name": "ST",
                        "cell_methods": "area: time: mean",
                        "cell_measures": "area: areacella",
                    },
                    "data": [[[247.7601776123047]]],
                },
                "lat_bnds": {"dims": ("lat", "bnds"), "attrs": {}, "data": [[-90.0, -85.0]]},
                "lon_bnds": {"dims": ("lon", "bnds"), "attrs": {}, "data": [[-5.0, 5.0]]},
                "time_bnds": {"dims": ("time", "bnds"), "attrs": {}, "data": [[1223115.0, 1223146.0]]},
            },
        }
    ).to_netcdf(tmp_path / "tas_Amon_CanESM5_piControl_r1i1p1f1_gn_520101-534012.nc")

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
