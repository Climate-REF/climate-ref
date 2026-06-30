from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from climate_ref_example.surface_temperature import (
    GlobalMeanSurfaceTemperatureBias,
    compare_model_and_reference,
    format_cmec_metric_bundle,
    latest_version_files,
    model_global_mean_sst,
    reference_global_mean_sst,
)

from climate_ref_core.datasets import (
    DatasetCollection,
    ExecutionDatasetCollection,
    SourceDatasetType,
)

_LAT = np.array([-45.0, 45.0])
_LON = np.array([0.0, 180.0])


def _grid(years: range, var: str, value: float, units: str) -> xr.Dataset:
    """A small synthetic monthly field on a regular lat/lon grid."""
    time = xr.date_range(start=f"{years.start}-01", periods=len(years) * 12, freq="MS", use_cftime=True)
    data = np.full((len(time), _LAT.size, _LON.size), value, dtype="float64")
    ds = xr.Dataset({var: (("time", "lat", "lon"), data)}, coords={"time": time, "lat": _LAT, "lon": _LON})
    ds[var].attrs["units"] = units
    return ds


def _make_model_files(tmp_path: Path, *, years: range, tos_celsius: float, tag: str = "") -> list[Path]:
    """A model ``tos`` file (degC) plus its ``areacello`` supplementary."""
    tos_path = tmp_path / f"tos{tag}.nc"
    _grid(years, "tos", tos_celsius, "degC").to_netcdf(tos_path)
    area_path = tmp_path / f"areacello{tag}.nc"
    area = xr.Dataset(
        {"areacello": (("lat", "lon"), np.full((_LAT.size, _LON.size), 1.0e6))},
        coords={"lat": _LAT, "lon": _LON},
    )
    area.to_netcdf(area_path)
    return [tos_path, area_path]


def _make_obs_file(tmp_path: Path, *, years: range, ts_kelvin: float, name: str = "obs.nc") -> Path:
    """A reference ``ts`` file in Kelvin (as HadISST is distributed)."""
    path = tmp_path / name
    _grid(years, "ts", ts_kelvin, "K").to_netcdf(path)
    return path


def _collection(
    paths: Path | list[Path],
    selector: tuple[tuple[str, str], ...],
    versions: list[str] | None = None,
    variable_ids: list[str] | None = None,
) -> DatasetCollection:
    """Wrap one or more files in a DatasetCollection with the given selector facets."""
    paths = [paths] if isinstance(paths, Path) else paths
    data: dict[str, list[str]] = {"instance_id": [p.stem for p in paths], "path": [str(p) for p in paths]}
    if versions is not None:
        data["version"] = versions
    if variable_ids is not None:
        data["variable_id"] = variable_ids
    return DatasetCollection(pd.DataFrame(data), "instance_id", selector=selector)


def test_reference_global_mean_converts_kelvin(tmp_path):
    obs = _make_obs_file(tmp_path, years=range(2000, 2005), ts_kelvin=288.15)

    series = reference_global_mean_sst([obs])

    assert list(series["year"].values) == [2000, 2001, 2002, 2003, 2004]
    np.testing.assert_allclose(series.values, 15.0)  # 288.15 K -> 15.0 degC


def test_model_global_mean_area_weighted(tmp_path):
    files = _make_model_files(tmp_path, years=range(2000, 2003), tos_celsius=18.0)

    series = model_global_mean_sst(files)

    assert list(series["year"].values) == [2000, 2001, 2002]
    np.testing.assert_allclose(series.values, 18.0)


def test_compare_model_and_reference_restricts_to_overlap():
    model = xr.DataArray([18.0, 19.0, 20.0], coords={"year": [2000, 2001, 2002]}, dims="year")
    reference = xr.DataArray([17.0, 17.0], coords={"year": [2001, 2002]}, dims="year")

    comparison = compare_model_and_reference(model, reference)

    assert list(comparison["year"].values) == [2001, 2002]
    np.testing.assert_allclose(comparison["bias"].values, [2.0, 3.0])
    np.testing.assert_allclose(comparison.attrs["mean_bias"], 2.5)
    np.testing.assert_allclose(comparison.attrs["rmse"], np.sqrt((4.0 + 9.0) / 2))


def test_compare_model_and_reference_requires_overlap():
    model = xr.DataArray([18.0], coords={"year": [2000]}, dims="year")
    reference = xr.DataArray([17.0], coords={"year": [2014]}, dims="year")

    with pytest.raises(ValueError, match="overlapping years"):
        compare_model_and_reference(model, reference)


def test_format_cmec_metric_bundle():
    model = xr.DataArray([20.0, 21.0], coords={"year": [2000, 2001]}, dims="year")
    reference = xr.DataArray([18.0, 18.0], coords={"year": [2000, 2001]}, dims="year")
    comparison = compare_model_and_reference(model, reference)

    bundle = format_cmec_metric_bundle(comparison)

    results = bundle["RESULTS"]["global"]["tos"]
    assert set(results) == {"rmse", "mean-bias"}
    np.testing.assert_allclose(results["mean-bias"], comparison.attrs["mean_bias"])


def test_diagnostic_metadata():
    diagnostic = GlobalMeanSurfaceTemperatureBias()

    assert diagnostic.slug == "global-mean-surface-temperature-bias"
    # Two AND-groups: CMIP6+reference and CMIP7+reference.
    assert len(diagnostic.data_requirements) == 2
    assert all(len(group) == 2 for group in diagnostic.data_requirements)


def test_latest_version_files_picks_latest():
    df = pd.DataFrame(
        {
            "instance_id": ["old", "new"],
            "path": ["/data/old.nc", "/data/new.nc"],
            "version": ["v20210727", "v20250415"],
        }
    )
    collection = DatasetCollection(df, "instance_id", selector=(("source_id", "HadISST-1-1"),))

    files = latest_version_files(collection)

    assert files == [Path("/data/new.nc")]


def test_latest_version_files_keeps_supplementary_per_variable():
    # tos has two versions; areacello has one. The latest tos plus areacello must survive.
    df = pd.DataFrame(
        {
            "instance_id": ["tos_v1", "tos_v2", "area"],
            "path": ["/data/tos_v1.nc", "/data/tos_v2.nc", "/data/areacello.nc"],
            "version": ["v20191115", "v20200101", "v20191115"],
            "variable_id": ["tos", "tos", "areacello"],
        }
    )
    collection = DatasetCollection(df, "instance_id", selector=(("source_id", "ACCESS-ESM1-5"),))

    files = set(latest_version_files(collection))

    assert files == {Path("/data/tos_v2.nc"), Path("/data/areacello.nc")}


def test_execute_end_to_end(tmp_path, definition_factory):
    model_files = _make_model_files(tmp_path, years=range(2000, 2005), tos_celsius=18.0)
    reference_path = _make_obs_file(tmp_path, years=range(1990, 2010), ts_kelvin=288.15)  # 15.0 degC

    diagnostic = GlobalMeanSurfaceTemperatureBias()
    definition = definition_factory(
        diagnostic=diagnostic,
        cmip6=_collection(
            model_files,
            selector=(
                ("source_id", "ACCESS-ESM1-5"),
                ("experiment_id", "historical"),
                ("variant_label", "r1i1p1f1"),
            ),
        ),
        obs4mips=_collection(reference_path, selector=(("source_id", "HadISST-1-1"),)),
    )
    definition.output_directory.mkdir(parents=True, exist_ok=True)

    result = diagnostic.run(definition)

    assert result.successful
    assert (definition.output_directory / "global_mean_surface_temperature.nc").exists()
    assert (definition.output_directory / "surface_temperature_timeseries.png").exists()
    assert (definition.output_directory / "surface_temperature_bias.png").exists()
    comparison = xr.open_dataset(definition.output_directory / "global_mean_surface_temperature.nc")
    np.testing.assert_allclose(comparison.attrs["mean_bias"], 3.0)  # 18.0 - 15.0 degC


def test_execute_end_to_end_cmip7(tmp_path, definition_factory):
    # Drive the CMIP7 branch of _get_model_source_type.
    model_files = _make_model_files(tmp_path, years=range(2000, 2005), tos_celsius=18.0)
    reference_path = _make_obs_file(tmp_path, years=range(1990, 2010), ts_kelvin=288.15)

    collection = ExecutionDatasetCollection(
        {
            SourceDatasetType.CMIP7: _collection(
                model_files,
                selector=(
                    ("source_id", "ACCESS-ESM1-5"),
                    ("experiment_id", "historical"),
                    ("variant_label", "r1i1p1f1"),
                ),
            ),
            SourceDatasetType.obs4MIPs: _collection(reference_path, selector=(("source_id", "HadISST-1-1"),)),
        }
    )

    diagnostic = GlobalMeanSurfaceTemperatureBias()
    definition = definition_factory(diagnostic=diagnostic, execution_dataset_collection=collection)
    definition.output_directory.mkdir(parents=True, exist_ok=True)

    result = diagnostic.run(definition)

    assert result.successful
    comparison = xr.open_dataset(definition.output_directory / "global_mean_surface_temperature.nc")
    np.testing.assert_allclose(comparison.attrs["mean_bias"], 3.0)


def test_execute_with_multiple_reference_versions(tmp_path, definition_factory):
    # The full obs4REF archive ships two HadISST-1-1 versions with overlapping time ranges.
    # The diagnostic must use only the latest rather than combining both.
    model_files = _make_model_files(tmp_path, years=range(2000, 2005), tos_celsius=18.0)
    old_reference = _make_obs_file(tmp_path, years=range(1990, 2011), ts_kelvin=287.15, name="obs_old.nc")
    new_reference = _make_obs_file(tmp_path, years=range(1990, 2016), ts_kelvin=288.15, name="obs_new.nc")

    diagnostic = GlobalMeanSurfaceTemperatureBias()
    definition = definition_factory(
        diagnostic=diagnostic,
        cmip6=_collection(
            model_files,
            selector=(
                ("source_id", "ACCESS-ESM1-5"),
                ("experiment_id", "historical"),
                ("variant_label", "r1i1p1f1"),
            ),
        ),
        obs4mips=_collection(
            [old_reference, new_reference],
            selector=(("source_id", "HadISST-1-1"),),
            versions=["v20210727", "v20250415"],
            variable_ids=["ts", "ts"],
        ),
    )
    definition.output_directory.mkdir(parents=True, exist_ok=True)

    result = diagnostic.run(definition)

    assert result.successful
    comparison = xr.open_dataset(definition.output_directory / "global_mean_surface_temperature.nc")
    # Bias is model (18.0) minus the latest reference (288.15 K = 15.0 degC), not the older one.
    np.testing.assert_allclose(comparison.attrs["mean_bias"], 3.0)
