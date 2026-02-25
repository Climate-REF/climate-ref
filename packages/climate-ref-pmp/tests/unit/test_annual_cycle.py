import datetime

import pandas as pd
import pytest
from climate_ref_pmp import AnnualCycle
from climate_ref_pmp import provider as pmp_provider
from climate_ref_pmp.diagnostics.annual_cycle import _transform_results
from climate_ref_pmp.pmp_driver import _get_resource

from climate_ref.solver import solve_executions
from climate_ref_core.datasets import DatasetCollection, SourceDatasetType
from climate_ref_core.diagnostics import ExecutionDefinition


def test_expected_executions():
    diagnostic = AnnualCycle()
    data_catalog = {
        SourceDatasetType.CMIP6: pd.DataFrame(
            [
                ["ts", "ACCESS-ESM1-5", "historical", "r1i1p1f1", "mon", "gn"],
                ["ts", "ACCESS-ESM1-5", "ssp119", "r1i1p1f1", "mon", "gn"],
                ["ts", "ACCESS-ESM1-5", "historical", "r2i1p1f1", "mon", "gn"],
                ["pr", "ACCESS-ESM1-5", "historical", "r1i1p1f1", "mon", "gn"],
            ],
            columns=("variable_id", "source_id", "experiment_id", "member_id", "frequency", "grid_label"),
        ),
        SourceDatasetType.PMPClimatology: pd.DataFrame(
            [["ERA-5", "ts"], ["ERA-5", "pr"], ["GPCP-Monthly-3-2", "pr"]],
            columns=["source_id", "variable_id"],
        ),
    }
    executions = list(solve_executions(data_catalog, diagnostic, provider=pmp_provider))
    assert len(executions) == 3

    # ts
    assert executions[0].datasets[SourceDatasetType.CMIP6].selector == (
        ("experiment_id", "historical"),
        ("grid_label", "gn"),
        ("member_id", "r1i1p1f1"),
        ("source_id", "ACCESS-ESM1-5"),
        ("variable_id", "ts"),
    )
    assert executions[0].datasets[SourceDatasetType.PMPClimatology].selector == (
        ("source_id", "ERA-5"),
        ("variable_id", "ts"),
    )

    # ts with different member_id
    assert executions[1].datasets[SourceDatasetType.CMIP6].selector == (
        ("experiment_id", "historical"),
        ("grid_label", "gn"),
        ("member_id", "r2i1p1f1"),
        ("source_id", "ACCESS-ESM1-5"),
        ("variable_id", "ts"),
    )
    assert executions[0].datasets[SourceDatasetType.PMPClimatology].selector == (
        ("source_id", "ERA-5"),
        ("variable_id", "ts"),
    )

    # pr
    assert executions[2].datasets[SourceDatasetType.CMIP6].selector == (
        ("experiment_id", "historical"),
        ("grid_label", "gn"),
        ("member_id", "r1i1p1f1"),
        ("source_id", "ACCESS-ESM1-5"),
        ("variable_id", "pr"),
    )
    assert executions[2].datasets[SourceDatasetType.PMPClimatology].selector == (
        ("source_id", "GPCP-Monthly-3-2"),
        ("variable_id", "pr"),
    )


@pytest.mark.parametrize(
    "variable_id,source_id,member_id",
    [
        ("pr", "ACCESS-ESM1-5", "r1i1p1f1"),
        ("ts", "MPI-ESM1-2-LR", "r2i2p1f1"),
    ],
)
def test_annual_cycle_diagnostic(
    variable_id,
    source_id,
    member_id,
    cmip6_data_catalog,
    obs4mips_data_catalog,
    definition_factory,
    pdo_example_dir,
    provider,
):
    diagnostic = AnnualCycle()
    diagnostic.provider = provider

    expected_input_filename = cmip6_data_catalog["path"].iloc[0]
    expected_reference_filename = obs4mips_data_catalog["path"].iloc[0]

    definition = definition_factory(
        diagnostic=diagnostic,
        cmip6=DatasetCollection(
            pd.Series(
                {
                    "instance_id": "HadISST",
                    "source_id": source_id,
                    "variable_id": variable_id,
                    "experiment_id": "historical",
                    "member_id": member_id,
                    "path": expected_input_filename,
                }
            )
            .to_frame()
            .T,
            "instance_id",
        ),
        pmp_climatology=DatasetCollection(
            pd.Series(
                {
                    "instance_id": "HadISST",
                    "source_id": "HadISST-1-1",
                    "variable_id": variable_id,
                    "path": expected_reference_filename,
                }
            )
            .to_frame()
            .T,
            "instance_id",
        ),
    )
    # The output directory must exist
    output_dir = definition.output_directory
    parameter_file = _get_resource(
        "climate_ref_pmp.params", "pmp_param_annualcycle_1-clims.py", use_resources=True
    )
    date_stamp = datetime.datetime.now().strftime("%Y%m%d")

    definition.output_directory.mkdir(parents=True)

    result = diagnostic.build_cmds(definition)

    assert len(result) == 2

    # Check the first command
    cmd = result[0]
    assert cmd == [
        "pcmdi_compute_climatologies.py",
        "-p",
        parameter_file,
        "--vars",
        variable_id,
        "--infile",
        str(expected_input_filename),
        "--outfile",
        f"{output_dir}/{variable_id}_{source_id}_historical_{member_id}_clims.nc",
        "--version",
        f"v{date_stamp}",
    ]

    # Check the second command
    parameter_file = _get_resource(
        "climate_ref_pmp.params", "pmp_param_annualcycle_2-metrics.py", use_resources=True
    )
    cmd = result[1]
    assert cmd == [
        "mean_climate_driver.py",
        "-p",
        parameter_file,
        "--vars",
        variable_id,
        "--custom_observations",
        f"{output_dir}/obs_dict.json",
        "--test_data_path",
        str(output_dir),
        "--test_data_set",
        source_id,
        "--realization",
        member_id,
        "--filename_template",
        f"%(variable)_{source_id}_historical_{member_id}_clims.198101-200512.AC.v{date_stamp}.nc",
        "--metrics_output_path",
        str(output_dir),
        "--cmec",
    ]


def test_diagnostic_execute(mocker, provider):
    diagnostic = AnnualCycle()
    diagnostic.provider = provider

    # Mock the 'run' method of the 'provider' object
    mocked_provider_run = mocker.patch.object(
        provider,
        "run",
        autospec=True,
        spec_set=True,
    )

    diagnostic.build_cmds = mocker.MagicMock(return_value=[["mocked_command1"], ["mocked_command2"]])
    diagnostic.build_execution_result = mocker.MagicMock()

    # Create a mock ExecutionDefinition
    mock_definition = mocker.MagicMock(spec=ExecutionDefinition)
    diagnostic.execute(mock_definition)

    diagnostic.build_cmds.assert_called_once_with(mock_definition)
    assert mocked_provider_run.call_count == 2
    mocked_provider_run.assert_any_call(["mocked_command1"])
    mocked_provider_run.assert_any_call(["mocked_command2"])


def test_build_cmds(diagnostic_validation):
    diagnostic = AnnualCycle()
    diagnostic.provider = pmp_provider
    validator = diagnostic_validation(diagnostic)

    definition = validator.get_definition()
    definition.output_directory.mkdir(parents=True, exist_ok=True)

    cmds = diagnostic.build_cmds(definition)

    assert len(cmds) == 2

    assert (definition.output_directory / "obs_dict.json").exists()


def test_transform_results_removes_expected_keys():
    input_data = {
        "DIMENSIONS": {
            "json_structure": ["model", "reference", "rip", "region", "statistic", "season"],
            "model": {"ACCESS-ESM1-5": {}},
            "reference": {"default": {}},
            "rip": {"r1i1p1f1": {}},
            "region": {"global": {}, "NHEX": {}, "SHEX": {}, "TROPICS": {}},
            "statistic": {
                "bias_xy": {},
                "cor_xy": {},
                "mae_xy": {},
                "mean-obs_xy": {},
                "mean_xy": {},
                "rms_devzm": {},
                "rms_xy": {},
                "rms_xyt": {},
                "rms_y": {},
                "rmsc_xy": {},
                "std-obs_xy": {},
                "std-obs_xy_devzm": {},
                "std-obs_xyt": {},
                "std_xy": {},
                "std_xy_devzm": {},
                "std_xyt": {},
            },
            "season": {"ann": {}, "djf": {}, "mam": {}, "jja": {}, "son": {}, "CalendarMonths": {}},
        },
        "json_version": 3.0,
        "RESULTS": {
            "ACCESS-ESM1-5": {
                "default": {
                    "r1i1p1f1": {
                        "global": {
                            "bias_xy": {
                                "ann": 1.55238,
                                "djf": 1.66939,
                                "mam": 1.47425,
                                "jja": 1.47011,
                                "son": 1.59322,
                                "CalendarMonths": [
                                    "1.66806e+00",
                                    "1.62649e+00",
                                    "1.54286e+00",
                                    "1.40157e+00",
                                    "1.48475e+00",
                                    "1.55714e+00",
                                    "1.48418e+00",
                                    "1.36947e+00",
                                    "1.39818e+00",
                                    "1.57127e+00",
                                    "1.80950e+00",
                                    "1.71510e+00",
                                ],
                            },
                        },
                    },
                    "attributes": {"source": "ERA-5"},
                },
                "attributes": {"units": ""},
            }
        },
    }
    expected_output = {
        "DIMENSIONS": {
            "json_structure": ["region", "statistic", "season"],
            "region": {"NHEX": {}, "SHEX": {}, "TROPICS": {}, "global": {}},
            "season": {"ann": {}, "djf": {}, "jja": {}, "mam": {}, "son": {}},
            "statistic": {
                "bias_xy": {},
                "cor_xy": {},
                "mae_xy": {},
                "mean-obs_xy": {},
                "mean_xy": {},
                "rms_devzm": {},
                "rms_xy": {},
                "rms_xyt": {},
                "rms_y": {},
                "rmsc_xy": {},
                "std-obs_xy": {},
                "std-obs_xy_devzm": {},
                "std-obs_xyt": {},
                "std_xy": {},
                "std_xy_devzm": {},
                "std_xyt": {},
            },
        },
        "RESULTS": {
            "global": {
                "bias_xy": {"ann": 1.55238, "djf": 1.66939, "jja": 1.47011, "mam": 1.47425, "son": 1.59322}
            },
        },
        "json_version": 3.0,
    }
    transformed_data = _transform_results(input_data)
    assert transformed_data == expected_output


def test_transform_results_empty_results_and_dimensions():
    input_data = {
        "RESULTS": {},
        "DIMENSIONS": {"json_structure": []},
    }
    with pytest.raises(ValueError, match="'model' is not in list"):
        _transform_results(input_data)
