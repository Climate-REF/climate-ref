import shutil

import pytest
from climate_ref_pmp.diagnostics import ExtratropicalModesOfVariability
from climate_ref_pmp.pmp_driver import _get_resource

from climate_ref.solver import solve_executions


def test_pdo_metric(data_catalog, config, mocker, pdo_example_dir, provider):
    diagnostic = ExtratropicalModesOfVariability("PDO")
    diagnostic.provider = provider

    execution = next(
        solve_executions(
            data_catalog=data_catalog,
            diagnostic=diagnostic,
            provider=diagnostic.provider,
        )
    )
    definition = execution.build_execution_definition(output_root=config.paths.scratch)

    def mock_run_fn(cmd, *args, **kwargs):
        # Copy the output from the test-data directory to the output directory
        output_path = definition.output_directory
        shutil.copytree(pdo_example_dir, output_path)

    # Mock the subprocess.run call to avoid running PMP
    # Instead the mock_run_call function will be called
    mock_run = mocker.patch.object(
        provider,
        "run",
        autospec=True,
        spec_set=True,
        side_effect=mock_run_fn,
    )
    result = diagnostic.run(definition)

    mock_run.assert_called_with(
        [
            "variability_modes_driver.py",
            "-p",
            _get_resource("climate_ref_pmp.params", "pmp_param_MoV-ts.py", True),
            "--variability_mode",
            "PDO",
            "--modpath",
            definition.datasets["cmip6"].path.to_list()[0],
            "--modpath_lf",
            "none",
            "--mip",
            "cmip6",
            "--exp",
            "hist-GHG",
            "--realization",
            "r1i1p1f1",
            "--modnames",
            "ACCESS-ESM1-5",
            "--reference_data_name",
            "HadISST-1-1",
            "--reference_data_path",
            definition.datasets["obs4mips"].path.to_list()[0],
            "--results_dir",
            str(definition.output_directory),
            "--cmec",
            "--no_provenance",
        ],
    )

    assert result.successful

    assert str(result.output_bundle_filename) == "output.json"

    output_bundle_path = definition.output_directory / result.output_bundle_filename

    assert output_bundle_path.exists()
    assert output_bundle_path.is_file()

    assert str(result.metric_bundle_filename) == "diagnostic.json"

    metric_bundle_path = definition.output_directory / result.metric_bundle_filename

    assert result.successful
    assert metric_bundle_path.exists()
    assert metric_bundle_path.is_file()


def test_mode_id_valid():
    # Test valid mode_ids and their corresponding parameter files
    valid_modes = {
        "PDO": "pmp_param_MoV-ts.py",
        "NPGO": "pmp_param_MoV-ts.py",
        "AMO": "pmp_param_MoV-ts.py",
        "NAO": "pmp_param_MoV-psl.py",
        "NAM": "pmp_param_MoV-psl.py",
        "PNA": "pmp_param_MoV-psl.py",
        "NPO": "pmp_param_MoV-psl.py",
        "SAM": "pmp_param_MoV-psl.py",
    }

    for mode_id, expected_file in valid_modes.items():
        obj = ExtratropicalModesOfVariability(mode_id)
        assert obj.parameter_file == expected_file


def test_mode_id_invalid():
    # Test an invalid mode_id
    with pytest.raises(ValueError) as excinfo:
        ExtratropicalModesOfVariability("INVALID")
    assert "Unknown mode_id 'INVALID'" in str(excinfo.value)
