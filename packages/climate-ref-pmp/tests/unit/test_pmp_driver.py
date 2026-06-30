import fnmatch

import pytest
from climate_ref_pmp.diagnostics.annual_cycle import AnnualCycle
from climate_ref_pmp.diagnostics.enso import ENSO
from climate_ref_pmp.diagnostics.variability_modes import ExtratropicalModesOfVariability
from climate_ref_pmp.pmp_driver import (
    PMP_RECONSTRUCTION_INPUTS,
    build_glob_pattern,
    build_pmp_command,
    process_json_result,
)

from climate_ref_core.pycmec.metric import CMECMetric
from climate_ref_core.pycmec.output import CMECOutput


def test_pmp_reconstruction_inputs_declared_and_match_driver_cmec_naming():
    """Every PMP diagnostic persists the raw driver CMEC JSON its rebuild re-scans."""
    for cls in (AnnualCycle, ENSO, ExtratropicalModesOfVariability):
        assert cls.reconstruction_inputs == PMP_RECONSTRUCTION_INPUTS

    # The glob matches the driver's CMEC JSON file names (mip-scoped and plain)...
    for name in (
        "var_mode_PDO_EOF1_stat_cmip6_hist-GHG_mo_atm_ACCESS-ESM1-5_r1i1p1f1_2000-2005_cmec.json",
        "enso_perf_CMIP6_cmec.json",
        "annual_cycle_cmec.json",
    ):
        assert any(fnmatch.fnmatch(name, pat) for pat in PMP_RECONSTRUCTION_INPUTS), name
    # ...but not the curated committed bundle, which is captured by the bundle itself.
    assert not any(fnmatch.fnmatch("output.json", pat) for pat in PMP_RECONSTRUCTION_INPUTS)


def test_process_json_result(pdo_example_dir):
    json_file = (
        pdo_example_dir
        / "var_mode_PDO_EOF1_stat_cmip6_hist-GHG_mo_atm_ACCESS-ESM1-5_r1i1p1f1_2000-2005_cmec.json"
    )
    png_files = [pdo_example_dir / "pdo.png"]
    data_files = [pdo_example_dir / "pdo.nc"]

    cmec_output, cmec_metric = process_json_result(json_file, png_files, data_files)

    assert CMECMetric.model_validate(cmec_metric)
    assert CMECOutput.model_validate(cmec_output)
    assert len(cmec_metric.RESULTS)
    assert cmec_metric.DIMENSIONS.root["json_structure"] == [
        "model",
        "realization",
        "reference",
        "mode",
        "season",
        "method",
        "statistic",
    ]


def test_execute_missing_parameter():
    with pytest.raises(
        FileNotFoundError,
        match=r"Resource pmp_missing.py not found in climate_ref_pmp.params package.",
    ):
        build_pmp_command(
            driver_file="variability_mode/variability_modes_driver.py",
            parameter_file="pmp_missing.py",
            model_files=["model1.nc"],
            reference_name="HadISST-1-1",
            reference_paths=["reference.nc"],
            source_id="source_id",
            member_id="member_id",
            output_directory_path="output",
            experiment_id="historical",
        )


def test_build_glob_pattern_from_docstring_example():
    paths = [
        "/home/user/data/folder1/file1.nc",
        "/home/user/data/folder1/file2.nc",
        "/home/user/data/folder2/file3.nc",
    ]

    pattern = build_glob_pattern(paths)
    expected = "/home/user/data/**/file*.nc"
    assert pattern == expected


def test_build_glob_pattern_same_directory():
    paths = [
        "/home/user/data/folder1/sample_A.nc",
        "/home/user/data/folder1/sample_B.nc",
        "/home/user/data/folder1/sample_C.nc",
    ]

    pattern = build_glob_pattern(paths)
    expected = "/home/user/data/folder1/sample_*.nc"
    assert pattern == expected
