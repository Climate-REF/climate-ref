"""
Unit tests for the helpers in the standard module.
"""

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
)


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
            registry_file="ilamb-test",
            metric_name="test-ver",
            sources={"tas": "ilamb/test/Site/tas.nc"},
            version=3,
        )

        assert diagnostic.version == 3

    def test_default_version_used_when_not_overridden(self):
        diagnostic = ILAMBStandard(
            registry_file="ilamb-test",
            metric_name="test-ver-default",
            sources={"tas": "ilamb/test/Site/tas.nc"},
        )

        assert diagnostic.version == ILAMBStandard.version
