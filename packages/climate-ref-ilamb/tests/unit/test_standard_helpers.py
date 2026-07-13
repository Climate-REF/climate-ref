"""
Unit tests for the helpers in the standard module.
"""

import ilamb3
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from climate_ref_ilamb.standard import (
    ILAMBStandard,
    _build_cmec_bundle,
    _build_series,
    _clean_units,
    _CoarsenSpatial,
    _RelationshipTimeTransform,
    _set_ilamb3_options,
)
from ilamb3.dataset import coarsen_dataset

from climate_ref_core.dataset_registry import dataset_registry_manager


class TestRelationshipTimeTransform:
    def _dataset(self, values: list[float]) -> xr.Dataset:
        time = pd.date_range("2000-01-01", periods=len(values), freq="MS")
        return xr.Dataset({"pr": ("time", values)}, coords={"time": time})

    def test_trims_to_valid_time_range(self):
        transform = _RelationshipTimeTransform("pr")
        ds = self._dataset([1.0, 2.0, 3.0, np.nan, np.nan])

        result = transform(ds)

        assert result.sizes["time"] == 3
        assert result["pr"].notnull().all()

    def test_no_time_coordinate_returns_unchanged(self):
        transform = _RelationshipTimeTransform("pr")
        ds = xr.Dataset({"pr": ("x", [1.0, 2.0])})

        result = transform(ds)

        assert result.identical(ds)

    def test_missing_variable_returns_unchanged(self):
        transform = _RelationshipTimeTransform("tas")
        ds = self._dataset([1.0, 2.0, 3.0])

        result = transform(ds)

        assert result.identical(ds)

    def test_all_missing_returns_unchanged(self):
        transform = _RelationshipTimeTransform("pr")
        ds = self._dataset([np.nan, np.nan, np.nan])

        result = transform(ds)

        # Nothing valid to align to, so the dataset is returned as-is
        assert result.sizes["time"] == 3

    def test_required_variables(self):
        assert _RelationshipTimeTransform("pr").required_variables() == ["pr"]


class TestCoarsenSpatial:
    def _dataset(self, spacing: float, extent: float = 2.0) -> xr.Dataset:
        lat = np.arange(0, extent, spacing)
        lon = np.arange(0, extent, spacing)
        data = np.arange(len(lat) * len(lon), dtype=float).reshape(len(lat), len(lon))
        return xr.Dataset(
            {"tas": (("lat", "lon"), data)},
            coords={"lat": lat, "lon": lon},
        )

    def test_required_variables(self):
        assert _CoarsenSpatial("tas").required_variables() == ["tas"]

    def test_missing_variable_returns_unchanged(self):
        transform = _CoarsenSpatial("tas")
        ds = xr.Dataset({"pr": (("lat", "lon"), [[1.0, 2.0], [3.0, 4.0]])})

        result = transform(ds)

        assert result.identical(ds)

    def test_already_coarse_returns_unchanged(self):
        # 1-degree spacing is already coarser than the default 0.5-degree target.
        ds = self._dataset(spacing=1.0, extent=5.0)
        transform = _CoarsenSpatial("tas")

        result = transform(ds)

        assert result.identical(ds)

    def test_fine_field_is_coarsened(self):
        # ~0.1-degree spacing is finer than the 0.5-degree target, so it gets coarsened.
        ds = self._dataset(spacing=0.1, extent=2.0)
        transform = _CoarsenSpatial("tas", resolution=0.5)

        result = transform(ds)

        assert result.sizes["lat"] < ds.sizes["lat"]
        assert result.sizes["lon"] < ds.sizes["lon"]
        original_spacing = float(np.diff(ds["lat"].values).mean())
        coarsened_spacing = float(np.diff(result["lat"].values).mean())
        assert coarsened_spacing > original_spacing

    def _timed_dataset(self, spacing: float = 0.1, extent: float = 2.0, ntime: int = 4) -> xr.Dataset:
        lat = np.arange(0, extent, spacing)
        lon = np.arange(0, extent, spacing)
        time = pd.date_range("2000-01-01", periods=ntime, freq="MS")
        rng = np.random.default_rng(0)
        data = rng.random((ntime, len(lat), len(lon)))
        return xr.Dataset(
            {"tas": (("time", "lat", "lon"), data)},
            coords={"time": time, "lat": lat, "lon": lon},
        )

    def test_streaming_matches_eager_coarsen(self):
        # The chunked (streaming) coarsen must produce the same numbers as coarsening the
        # whole materialised cube in one pass.
        ds = self._timed_dataset()
        expected = coarsen_dataset(ds.compute(), "tas", res=0.5)

        transform = _CoarsenSpatial("tas", resolution=0.5)
        result = transform._coarsen(ds)

        xr.testing.assert_allclose(result["tas"], expected["tas"])

    def test_result_is_cached_and_reused(self, tmp_path, monkeypatch):
        # First call computes and writes the cache; a second call reads it back rather
        # than recomputing. Proven by overwriting the cached file with a sentinel value.
        monkeypatch.setenv("REF_DATASET_CACHE_DIR", str(tmp_path))
        monkeypatch.delenv("REF_ILAMB_COARSEN_NO_CACHE", raising=False)
        ds = self._timed_dataset()
        transform = _CoarsenSpatial("tas", resolution=0.5)

        first = transform(ds)
        cache_path = transform._cache_path(ds)
        assert cache_path is not None
        assert cache_path.exists()

        with xr.open_dataset(cache_path) as cached:
            sentinel = cached.load()
        sentinel["tas"] = sentinel["tas"] * 0.0 + 42.0
        sentinel.to_netcdf(cache_path)

        second = transform(ds)
        assert float(second["tas"].mean()) == 42.0
        # The uncached compute is unchanged and still available.
        assert not np.allclose(first["tas"].values, 42.0)

    def test_cache_disabled_by_env(self, tmp_path, monkeypatch):
        monkeypatch.setenv("REF_DATASET_CACHE_DIR", str(tmp_path))
        monkeypatch.setenv("REF_ILAMB_COARSEN_NO_CACHE", "1")
        ds = self._timed_dataset()
        transform = _CoarsenSpatial("tas", resolution=0.5)

        transform(ds)

        assert transform._cache_path(ds) is None
        assert not list(tmp_path.rglob("*.nc"))


class TestCleanUnits:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            # A pint object repr (what ILAMB writes for model traces) becomes CF/UDUNITS.
            ("kg / meter ** 2 / second", "kg m-2 s-1"),
            ("gram / day / meter ** 2", "g d-1 m-2"),
            # Already-clean CF units round-trip unchanged.
            ("kg m-2 s-1", "kg m-2 s-1"),
            ("K", "K"),
            # A fractional exponent keeps its float power.
            ("m ** 0.5", "m0.5"),
            # Dimensionless and unparseable strings are preserved as-is.
            ("1", "1"),
            ("", ""),
            ("months since 1980", "months since 1980"),
        ],
    )
    def test_clean_units(self, raw, expected):
        assert _clean_units(raw) == expected

    def test_model_and_reference_units_agree(self):
        # The model trace pint repr and the reference CF units normalise to the same string.
        assert _clean_units("kg / meter ** 2 / second") == _clean_units("kg m-2 s-1")


def _write_trace(path, varname, units, *, calendar="noleap"):
    """Write a 1-d time trace netCDF mimicking an ILAMB output file."""
    time = xr.date_range("2000-01-01", periods=3, freq="YS", calendar=calendar, use_cftime=True)
    da = xr.DataArray(
        [1.0, 2.0, 3.0],
        dims="time",
        coords={"time": time},
        attrs={"units": units, "long_name": "Gross Primary Productivity", "standard_name": "gpp"},
    )
    xr.Dataset({varname: da}).to_netcdf(path)


class TestBuildSeries:
    def test_reference_and_model_series(self, tmp_path):
        _write_trace(tmp_path / "Reference.nc", "gpp_global", "kg m-2 s-1")
        _write_trace(tmp_path / "CanESM5.nc", "gpp_global", "kg / meter ** 2 / second")
        common_dimensions = {
            "source_id": "CanESM5",
            "experiment_id": "historical",
            "reference_source_id": "FLUXNET2015",
        }

        series = _build_series(tmp_path, "FLUXNET2015", common_dimensions)
        by_kind = {s.kind: s for s in series}
        assert set(by_kind) == {"reference", "model"}

        reference = by_kind["reference"]
        model = by_kind["model"]

        # The reference is a standalone observation: reference identity, no model source_id.
        assert reference.kind == "reference"
        assert "source_id" not in reference.dimensions
        assert reference.dimensions["reference_source_id"] == "FLUXNET2015"

        # The model series keeps both identities so it groups with its reference.
        assert model.kind == "model"
        assert model.dimensions["source_id"] == "CanESM5"
        assert model.dimensions["reference_source_id"] == "FLUXNET2015"

        # Units are clean (no pint repr) and identical between model and reference.
        assert model.value_units == "kg m-2 s-1"
        assert reference.value_units == "kg m-2 s-1"
        assert "**" not in model.value_units
        assert "meter" not in model.value_units

        # The metric/region split and presentation metadata are populated.
        assert model.dimensions["metric"] == "gpp"
        assert model.dimensions["region"] == "global"
        assert model.value_long_name == "Gross Primary Productivity"
        assert model.calendar == "noleap"
        # The decoded time axis has no units attribute, so index_units stays unset.
        assert model.index_units is None

    def test_month_index_and_higher_dims_skipped(self, tmp_path):
        # A month index has no calendar and stays numeric; a 2-d field is not a series.
        ds = xr.Dataset(
            {
                "nbp_global": ("month", [1.0, 2.0, 3.0]),
                "map_global": (("month", "region"), [[1.0], [2.0], [3.0]]),
            },
            coords={"month": [1, 2, 3]},
        )
        ds["nbp_global"].attrs["units"] = "kg m-2 s-1"
        ds.to_netcdf(tmp_path / "CanESM5.nc")

        series = _build_series(tmp_path, "GFED", {"source_id": "CanESM5"})

        # Only the 1-d month trace becomes a series; the 2-d field is skipped.
        assert len(series) == 1
        (s,) = series
        assert s.index_name == "month"
        assert s.index == [1, 2, 3]
        assert s.calendar is None
        assert s.index_units is None


class TestBuildCmecBundle:
    def test_json_structure_carries_reference_identity_and_kind(self):
        df = pd.DataFrame(
            {
                "source": ["CanESM5"],
                "analysis": ["Bias"],
                "name": ["bias [kg m-2 s-1]"],
                "value": [0.5],
                "experiment_id": ["historical"],
                "source_id": ["CanESM5"],
                "member_id": ["r1i1p1f1"],
                "grid_label": ["gn"],
                "region": ["None"],
                "type": ["scalar"],
                "units": ["kg m-2 s-1"],
                "reference_source_id": ["FLUXNET2015"],
            }
        )

        bundle = _build_cmec_bundle(df)
        json_structure = bundle["DIMENSIONS"]["json_structure"]

        assert "reference_source_id" in json_structure
        assert "kind" in json_structure
        # ILAMB scalars are model-vs-reference comparisons -> kind "model".
        assert list(bundle["DIMENSIONS"]["kind"]) == ["model"]
        assert list(bundle["DIMENSIONS"]["reference_source_id"]) == ["FLUXNET2015"]


class TestVersionOverride:
    def test_per_diagnostic_version_override(self):
        diagnostic = ILAMBStandard(
            realm="land",
            metric_name="test-ver",
            sources={"tas": "ilamb/test/Site/tas.nc"},
            ilamb_registry="ilamb-test",
            version=3,
        )

        assert diagnostic.version == 3

    def test_default_version_used_when_not_overridden(self):
        diagnostic = ILAMBStandard(
            realm="land",
            metric_name="test-ver-default",
            sources={"tas": "ilamb/test/Site/tas.nc"},
            ilamb_registry="ilamb-test",
        )

        assert diagnostic.version == ILAMBStandard.version


class TestRealmMaskDecoupling:
    """
    Proves mask-loading is independent of realm, not just "is land".

    ``ilamb-test`` is a land-realm registry that carries no region masks, so a
    land diagnostic built against it must not attempt to load any.
    """

    def test_land_realm_without_region_masks_loads_no_masks(self):
        diagnostic = ILAMBStandard(
            realm="land",
            metric_name="test-land-no-masks",
            sources={"tas": "ilamb/test/Site/tas.nc"},
            ilamb_registry="ilamb-test",
        )

        assert diagnostic.realm == "land"
        assert diagnostic.region_masks is None

        _set_ilamb3_options(None)
        assert ilamb3.conf["regions"] == [None]

    def test_land_realm_with_region_masks_loads_masks(self):
        diagnostic = ILAMBStandard(
            realm="land",
            metric_name="test-land-with-masks",
            sources={"tas": "ilamb/tas/CRU4.02/tas.nc"},
            region_masks="ilamb-regions",
        )

        assert diagnostic.region_masks == "ilamb-regions"

        _set_ilamb3_options(dataset_registry_manager[diagnostic.region_masks])
        assert set(["global", "tropical"]).issubset(ilamb3.conf["regions"])

    def test_ocean_realm_has_empty_ilamb_data(self):
        diagnostic = ILAMBStandard(
            realm="ocean",
            metric_name="test-ocean-empty",
            sources={
                "thetao": {
                    "obs_source": "obs4ref",
                    "source_id": "WOA-23",
                    "variable_id": "thetao",
                    "grid_label": "gn",
                    "version": "v20251024",
                }
            },
        )

        assert diagnostic.region_masks is None
        assert diagnostic.ilamb_registry is None
        assert diagnostic.ilamb_data.datasets.empty
        assert list(diagnostic.ilamb_data.datasets.columns) == ["key", "path"]
        # The merge in `execute()` must not crash on an empty ILAMB side.
        merged = diagnostic.ilamb_data.datasets.set_index(diagnostic.ilamb_data.slug_column)
        assert merged.empty
