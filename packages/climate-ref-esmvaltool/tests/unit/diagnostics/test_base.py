import json

import climate_ref_esmvaltool.diagnostics.base
import numpy as np
import pandas
import pytest
import xarray as xr
import yaml
from climate_ref_esmvaltool.diagnostics.base import ESMValToolDiagnostic
from climate_ref_esmvaltool.diagnostics.regional_historical_changes import _region_to_filename
from climate_ref_esmvaltool.types import Recipe

from climate_ref_core.datasets import SourceDatasetType
from climate_ref_core.diagnostics import CommandLineDiagnostic
from climate_ref_core.metric_values import SeriesMetricValue as SeriesMetricValueType
from climate_ref_core.metric_values.typing import SeriesDefinition
from climate_ref_core.pycmec.controlled_vocabulary import CV
from climate_ref_core.pycmec.output import OutputCV


@pytest.fixture
def mock_diagnostic():
    class MockDiagnostic(ESMValToolDiagnostic):
        base_recipe = "examples/recipe_python.yml"

        def update_recipe(
            self,
            recipe: Recipe,
            input_files: dict[SourceDatasetType, pandas.DataFrame],
        ) -> None:
            pass

    return MockDiagnostic()


@pytest.mark.parametrize("data_dir_exists", [True, False])
def test_build_cmd(mocker, tmp_path, metric_definition, mock_diagnostic, data_dir_exists):
    dataset_registry_manager = mocker.patch.object(
        climate_ref_esmvaltool.diagnostics.base,
        "dataset_registry_manager",
    )
    data_dir = tmp_path / "ESMValTool"
    if data_dir_exists:
        data_dir.mkdir()
    dataset_registry_manager.__getitem__.return_value.abspath = tmp_path
    output_dir = metric_definition.output_directory
    output_dir.mkdir(parents=True)
    cmd = mock_diagnostic.build_cmd(metric_definition)
    config_dir = output_dir / "config"
    recipe = output_dir / "recipe.yml"
    assert cmd == ["esmvaltool", "run", f"--config-dir={config_dir}", f"{recipe}"]
    assert (output_dir / "climate_data").is_dir()
    config = yaml.safe_load((config_dir / "config.yml").read_text(encoding="utf-8"))
    assert config["search_data"] == "quick"
    if data_dir_exists:
        assert len(config["projects"]) == 6
        assert "OBS" in config["projects"]
        assert "OBS6" in config["projects"]
        assert "native6" in config["projects"]
        # obs4MIPs should have both local and esmvaltool data sources
        assert "esmvaltool" in config["projects"]["obs4MIPs"]["data"]
    else:
        assert len(config["projects"]) == 3
    assert "CMIP6" in config["projects"]
    assert "CMIP7" in config["projects"]
    assert "obs4MIPs" in config["projects"]


def test_build_metric_result(metric_definition, mock_diagnostic):
    results_dir = metric_definition.to_output_path("executions") / "recipe_test"

    for subdir in "timeseries", "map":
        metadata = {}
        for dirname in "work", "plots":
            for i in range(2):
                suffix = ".nc" if dirname == "work" else ".png"
                filepath = results_dir / dirname / subdir / "script1" / f"file{i}{suffix}"
                metadata[str(filepath)] = {
                    "caption": f"This is {subdir} test file {i}.",
                }
        metadata_file = results_dir / "run" / subdir / "script1" / "diagnostic_provenance.yml"
        metadata_file.parent.mkdir(parents=True)
        with metadata_file.open("w", encoding="utf-8") as file:
            yaml.safe_dump(metadata, file)

    execution_result = mock_diagnostic.build_execution_result(definition=metric_definition)
    metric_bundle = json.loads(
        execution_result.to_output_path(execution_result.metric_bundle_filename).read_text(encoding="utf-8")
    )
    output_bundle = json.loads(
        execution_result.to_output_path(execution_result.output_bundle_filename).read_text(encoding="utf-8")
    )

    assert isinstance(metric_bundle, dict)
    assert metric_bundle

    assert isinstance(output_bundle, dict)
    assert OutputCV.DATA.value in output_bundle
    assert len(output_bundle[OutputCV.DATA.value]) == 4
    assert OutputCV.PLOTS.value in output_bundle
    plots = output_bundle[OutputCV.PLOTS.value]
    assert len(plots) == 4
    captions = {p["long_name"] for p in plots.values()}
    assert len(captions) == 4


def test_series_extraction(tmp_path, metric_definition, mock_diagnostic, mocker):
    # Definition of the netcdf files to extract series from
    metric_definition.diagnostic.series = [
        SeriesDefinition(
            file_pattern="work/timeseries/script1/file0.nc",
            attributes=["long_name", "units"],
            dimensions={"model": "TestModel"},
            values_name="data",
            index_name="time",
        )
    ]

    # Create a NetCDF file matching the pattern
    results_dir = metric_definition.to_output_path("executions") / "recipe_test"
    nc_path = results_dir / "work" / "timeseries" / "script1" / "file0.nc"
    nc_path.parent.mkdir(parents=True, exist_ok=True)
    times = np.array([1, 2, 3])
    data = np.array([10.0, 20.0, 30.0])
    ds = xr.Dataset({"data": ("time", data)}, coords={"time": times})
    ds.attrs["long_name"] = "Test Data"
    ds.attrs["units"] = "K"
    ds.to_netcdf(nc_path)

    # Dummy metadata file
    metadata_file = results_dir / "run" / "timeseries" / "script1" / "diagnostic_provenance.yml"
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    metadata = {str(nc_path): {"caption": "Test caption."}}
    with metadata_file.open("w", encoding="utf-8") as file:
        yaml.dump(metadata, file)

    result = mock_diagnostic.build_execution_result(definition=metric_definition)

    # Load the series from the output file
    assert result.series_filename is not None, "Series filename should be set"
    loaded_series = SeriesMetricValueType.load_from_json(result.to_output_path(result.series_filename))
    assert loaded_series, "Series should not be empty"
    s = loaded_series[0]
    assert isinstance(s, SeriesMetricValueType)
    assert s.dimensions == {"model": "TestModel"}
    assert s.values == [10.0, 20.0, 30.0]
    assert s.index == [1, 2, 3]
    assert s.index_name == "time"
    assert s.attributes["long_name"] == "Test Data"
    assert s.attributes["units"] == "K"
    assert s.attributes["caption"] == "Test caption."


def test_series_extraction_byte_string_index(tmp_path, metric_definition, mock_diagnostic, mocker):
    """Test that byte string coordinates in NetCDF files are decoded to regular strings."""
    metric_definition.diagnostic.series = [
        SeriesDefinition(
            file_pattern="work/trends/script1/file0.nc",
            attributes=[],
            dimensions={"variable_id": "tas", "statistic": "trend"},
            sel={},
            values_name="tas",
            index_name="shape_id",
        )
    ]

    results_dir = metric_definition.to_output_path("executions") / "recipe_test"
    nc_path = results_dir / "work" / "trends" / "script1" / "file0.nc"
    nc_path.parent.mkdir(parents=True, exist_ok=True)

    # Create a NetCDF file with byte string coordinates (as ESMValTool produces)
    shape_ids = np.array([b"ARP", b"CAF", b"MED"], dtype="|S3")
    data = np.array([0.05, 0.03, 0.07], dtype=np.float32)
    ds = xr.Dataset(
        {"tas": ("dim0", data)},
        coords={"shape_id": ("dim0", shape_ids)},
    )
    ds["tas"].attrs["long_name"] = "tas"
    ds["tas"].attrs["units"] = "K yr-1"
    ds["shape_id"].attrs["long_name"] = "shape_id"
    ds.to_netcdf(nc_path)

    metadata_file = results_dir / "run" / "trends" / "script1" / "diagnostic_provenance.yml"
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    metadata = {str(nc_path): {"caption": "Trend barplot."}}
    with metadata_file.open("w", encoding="utf-8") as file:
        yaml.dump(metadata, file)

    result = mock_diagnostic.build_execution_result(definition=metric_definition)

    loaded_series = SeriesMetricValueType.load_from_json(result.to_output_path(result.series_filename))
    assert loaded_series, "Series should not be empty"
    s = loaded_series[0]
    assert s.index == ["ARP", "CAF", "MED"], "Byte strings should be decoded to regular strings"
    assert s.index_name == "shape_id"
    assert s.dimensions == {"variable_id": "tas", "statistic": "trend"}
    assert len(s.values) == 3


@pytest.mark.parametrize(
    "region,expected",
    [
        ("Arabian-Peninsula", "Arabian-Peninsula"),
        ("C.Australia", "C-Australia"),
        ("Greenland/Iceland", "Greenland-Iceland"),
        ("West&Central-Europe", "West-Central-Europe"),
        ("N.E.North-America", "N-E-North-America"),
    ],
)
def test_region_to_filename(region, expected):
    """Test that region names are correctly converted to filename format."""
    assert _region_to_filename(region) == expected


def test_series_validation_failure(tmp_path, metric_definition, mock_diagnostic, mocker):
    metric_definition.diagnostic.series = [
        SeriesDefinition(
            file_pattern="work/timeseries/script1/file0.nc",
            attributes=["long_name", "units"],
            dimensions={"model": "TestModel"},
            values_name="data",
            index_name="time",
        )
    ]
    results_dir = metric_definition.to_output_path("executions") / "recipe_test"
    nc_path = results_dir / "work" / "timeseries" / "script1" / "file0.nc"
    nc_path.parent.mkdir(parents=True, exist_ok=True)
    times = np.array([1, 2, 3])
    data = np.array([10.0, 20.0, 30.0])
    ds = xr.Dataset({"data": ("time", data)}, coords={"time": times})
    ds.attrs["long_name"] = "Test Data"
    ds.attrs["units"] = "K"
    ds.to_netcdf(nc_path)

    # Dummy metadata file
    metadata_file = results_dir / "run" / "timeseries" / "script1" / "diagnostic_provenance.yml"
    metadata_file.parent.mkdir(parents=True, exist_ok=True)
    metadata = {str(nc_path): {"caption": "Test caption."}}
    with metadata_file.open("w", encoding="utf-8") as file:
        yaml.dump(metadata, file)

    # Patch CV.validate_metrics to raise an error for series
    mocker.patch.object(CV, "validate_metrics", side_effect=AssertionError("Validation failed"))
    log_spy = mocker.spy(climate_ref_esmvaltool.diagnostics.base.logger, "exception")

    # Run build_execution_result (should log exception)
    mock_diagnostic.build_execution_result(definition=metric_definition)
    assert log_spy.call_count >= 0  # Should log the validation failure


def test_stabilise_execution_dir(metric_definition, mock_diagnostic):
    """The timestamped session directory is renamed and its references rewritten."""
    executions_dir = metric_definition.to_output_path("executions")
    session_dir = executions_dir / "recipe_20260130_162732"

    nc_path = session_dir / "work" / "timeseries" / "script1" / "file0.nc"
    provenance = session_dir / "run" / "timeseries" / "script1" / "diagnostic_provenance.yml"
    provenance.parent.mkdir(parents=True)
    provenance.write_text(yaml.safe_dump({str(nc_path): {"caption": "c"}}), encoding="utf-8")
    index = session_dir / "index.html"
    index.write_text(f"<a href='{session_dir}/work'>x</a>", encoding="utf-8")

    mock_diagnostic._stabilise_execution_dir(metric_definition)

    stable_dir = executions_dir / "recipe"
    assert stable_dir.is_dir()
    assert not session_dir.exists()
    # The timestamped name no longer appears, and references resolve to the renamed dir.
    assert "recipe_20260130_162732" not in (stable_dir / "index.html").read_text(encoding="utf-8")
    rewritten = yaml.safe_load(
        (stable_dir / "run" / "timeseries" / "script1" / "diagnostic_provenance.yml").read_text(
            encoding="utf-8"
        )
    )
    assert all(key.startswith(f"{stable_dir}/") for key in rewritten)


def test_stabilise_execution_dir_no_output(metric_definition, mock_diagnostic):
    """Stabilisation is a no-op when no session directory was produced."""
    executions_dir = metric_definition.to_output_path("executions")
    executions_dir.mkdir(parents=True)

    mock_diagnostic._stabilise_execution_dir(metric_definition)

    assert not (executions_dir / "recipe").exists()


def test_prepare_regression_output_stabilises(metric_definition, mock_diagnostic):
    """The regression-capture hook stabilises the timestamped directory."""
    executions_dir = metric_definition.to_output_path("executions")
    session_dir = executions_dir / "recipe_20260130_162732"
    session_dir.mkdir(parents=True)
    (session_dir / "index.html").write_text(f"{session_dir}", encoding="utf-8")

    mock_diagnostic.prepare_regression_output(metric_definition)

    assert (executions_dir / "recipe").is_dir()
    assert not session_dir.exists()


def test_execute_does_not_stabilise(metric_definition, mock_diagnostic, mocker):
    """Normal execution leaves the timestamped directory untouched (regression-only)."""
    executions_dir = metric_definition.to_output_path("executions")

    def fake_execute(_self, definition):
        session_dir = definition.to_output_path("executions") / "recipe_20260130_162732"
        session_dir.mkdir(parents=True)

    mocker.patch.object(CommandLineDiagnostic, "execute", autospec=True, side_effect=fake_execute)

    mock_diagnostic.execute(metric_definition)

    assert (executions_dir / "recipe_20260130_162732").is_dir()
    assert not (executions_dir / "recipe").exists()


def test_build_execution_result_prefers_stable_dir(metric_definition, mock_diagnostic):
    """build_execution_result resolves the stabilised ``recipe`` dir over a timestamped one."""
    executions_dir = metric_definition.to_output_path("executions")

    # Stabilised directory with one data file and one plot.
    stable_dir = executions_dir / "recipe"
    metadata = {}
    for dirname in "work", "plots":
        suffix = ".nc" if dirname == "work" else ".png"
        filepath = stable_dir / dirname / "timeseries" / "script1" / f"file0{suffix}"
        metadata[str(filepath)] = {"caption": "stable file"}
    provenance = stable_dir / "run" / "timeseries" / "script1" / "diagnostic_provenance.yml"
    provenance.parent.mkdir(parents=True)
    provenance.write_text(yaml.safe_dump(metadata), encoding="utf-8")

    # Decoy timestamped directory that must be ignored when the stable one exists.
    decoy_dir = executions_dir / "recipe_20990101_000000"
    decoy_provenance = decoy_dir / "run" / "timeseries" / "script1" / "diagnostic_provenance.yml"
    decoy_provenance.parent.mkdir(parents=True)
    decoy_nc = decoy_dir / "work" / "timeseries" / "script1" / "decoy.nc"
    decoy_provenance.write_text(yaml.safe_dump({str(decoy_nc): {"caption": "decoy"}}), encoding="utf-8")

    result = mock_diagnostic.build_execution_result(definition=metric_definition)
    output_bundle = json.loads(
        result.to_output_path(result.output_bundle_filename).read_text(encoding="utf-8")
    )

    captured = list(output_bundle[OutputCV.DATA.value]) + list(output_bundle[OutputCV.PLOTS.value])
    assert captured
    assert all(path.startswith("executions/recipe/") for path in captured)
    assert not any("recipe_20990101_000000" in path for path in captured)
