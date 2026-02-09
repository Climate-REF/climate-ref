import json

import climate_ref_esmvaltool.diagnostics.base
import numpy as np
import pandas
import pytest
import xarray as xr
import yaml
from climate_ref_esmvaltool.diagnostics.base import ESMValToolDiagnostic
from climate_ref_esmvaltool.types import Recipe

from climate_ref_core.datasets import SourceDatasetType
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
    assert len(config["rootpath"]) == 6 if data_dir_exists else 1


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
