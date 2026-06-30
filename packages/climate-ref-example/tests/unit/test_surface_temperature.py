from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from climate_ref_example.surface_temperature import (
    GlobalMeanSurfaceTemperatureBias,
    compare_model_and_reference,
    format_cmec_metric_bundle,
    global_mean_surface_temperature,
    latest_version_files,
)

from climate_ref_core.datasets import (
    DatasetCollection,
    ExecutionDatasetCollection,
    SourceDatasetType,
)


def _make_ts_dataset(path: Path, *, years: range, value: float) -> Path:
    """Write a small synthetic monthly ``ts`` dataset on a regular lat/lon grid."""
    time = xr.date_range(start=f"{years.start}-01", periods=len(years) * 12, freq="MS", use_cftime=True)
    lat = np.array([-45.0, 45.0])
    lon = np.array([0.0, 180.0])
    data = np.full((len(time), lat.size, lon.size), value, dtype="float64")
    ds = xr.Dataset(
        {"ts": (("time", "lat", "lon"), data)},
        coords={"time": time, "lat": lat, "lon": lon},
    )
    ds.to_netcdf(path)
    return path


def test_global_mean_surface_temperature_is_annual(tmp_path):
    path = _make_ts_dataset(tmp_path / "model.nc", years=range(2000, 2005), value=288.0)

    series = global_mean_surface_temperature([path])

    # One value per calendar year, area-weighted mean of a constant field is the constant.
    assert list(series["year"].values) == [2000, 2001, 2002, 2003, 2004]
    np.testing.assert_allclose(series.values, 288.0)


def test_compare_model_and_reference_restricts_to_overlap():
    model = xr.DataArray([288.0, 289.0, 290.0], coords={"year": [2000, 2001, 2002]}, dims="year")
    reference = xr.DataArray([287.0, 287.0], coords={"year": [2001, 2002]}, dims="year")

    comparison = compare_model_and_reference(model, reference)

    assert list(comparison["year"].values) == [2001, 2002]
    np.testing.assert_allclose(comparison["bias"].values, [2.0, 3.0])
    np.testing.assert_allclose(comparison.attrs["mean_bias"], 2.5)
    np.testing.assert_allclose(comparison.attrs["rmse"], np.sqrt((4.0 + 9.0) / 2))


def test_compare_model_and_reference_requires_overlap():
    model = xr.DataArray([288.0], coords={"year": [2000]}, dims="year")
    reference = xr.DataArray([287.0], coords={"year": [2014]}, dims="year")

    with pytest.raises(ValueError, match="overlapping years"):
        compare_model_and_reference(model, reference)


def test_format_cmec_metric_bundle():
    model = xr.DataArray([290.0, 291.0], coords={"year": [2000, 2001]}, dims="year")
    reference = xr.DataArray([288.0, 288.0], coords={"year": [2000, 2001]}, dims="year")
    comparison = compare_model_and_reference(model, reference)

    bundle = format_cmec_metric_bundle(comparison)

    results = bundle["RESULTS"]["global"]["ts"]
    assert set(results) == {"rmse", "mean-bias"}
    np.testing.assert_allclose(results["mean-bias"], comparison.attrs["mean_bias"])


def test_diagnostic_metadata():
    diagnostic = GlobalMeanSurfaceTemperatureBias()

    assert diagnostic.slug == "global-mean-surface-temperature-bias"
    # Two AND-groups: CMIP6+reference and CMIP7+reference.
    assert len(diagnostic.data_requirements) == 2
    assert all(len(group) == 2 for group in diagnostic.data_requirements)


def _collection(
    paths: Path | list[Path],
    selector: tuple[tuple[str, str], ...],
    versions: list[str] | None = None,
) -> DatasetCollection:
    """Wrap one or more files in a DatasetCollection with the given selector facets."""
    paths = [paths] if isinstance(paths, Path) else paths
    data = {"instance_id": [p.stem for p in paths], "path": [str(p) for p in paths]}
    if versions is not None:
        data["version"] = versions
    return DatasetCollection(pd.DataFrame(data), "instance_id", selector=selector)


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


def test_execute_end_to_end(tmp_path, definition_factory):
    model_path = _make_ts_dataset(tmp_path / "model.nc", years=range(2000, 2005), value=290.0)
    reference_path = _make_ts_dataset(tmp_path / "obs.nc", years=range(1990, 2010), value=288.0)

    diagnostic = GlobalMeanSurfaceTemperatureBias()
    definition = definition_factory(
        diagnostic=diagnostic,
        cmip6=_collection(
            model_path,
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


def test_execute_end_to_end_cmip7(tmp_path, definition_factory):
    # Drive the CMIP7 branch of _get_model_source_type. definition_factory has no cmip7
    # kwarg, so build the collection explicitly.
    model_path = _make_ts_dataset(tmp_path / "model.nc", years=range(2000, 2005), value=290.0)
    reference_path = _make_ts_dataset(tmp_path / "obs.nc", years=range(1990, 2010), value=288.0)

    collection = ExecutionDatasetCollection(
        {
            SourceDatasetType.CMIP7: _collection(
                model_path,
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
    np.testing.assert_allclose(comparison.attrs["mean_bias"], 2.0)


def test_execute_with_multiple_reference_versions(tmp_path, definition_factory):
    # The full obs4REF archive ships two HadISST-1-1 ts versions with overlapping time
    # ranges. The diagnostic must use only the latest rather than combining both.
    model_path = _make_ts_dataset(tmp_path / "model.nc", years=range(2000, 2005), value=290.0)
    old_reference = _make_ts_dataset(tmp_path / "obs_old.nc", years=range(1990, 2011), value=287.0)
    new_reference = _make_ts_dataset(tmp_path / "obs_new.nc", years=range(1990, 2016), value=288.0)

    diagnostic = GlobalMeanSurfaceTemperatureBias()
    definition = definition_factory(
        diagnostic=diagnostic,
        cmip6=_collection(
            model_path,
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
        ),
    )
    definition.output_directory.mkdir(parents=True, exist_ok=True)

    result = diagnostic.run(definition)

    assert result.successful
    comparison = xr.open_dataset(definition.output_directory / "global_mean_surface_temperature.nc")
    # Bias is model (290) minus the latest reference (288), not the older one (287).
    np.testing.assert_allclose(comparison.attrs["mean_bias"], 2.0)
