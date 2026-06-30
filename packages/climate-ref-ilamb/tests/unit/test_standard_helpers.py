"""
Unit tests for the helpers in the standard module.
"""

import numpy as np
import pandas as pd
import xarray as xr
from climate_ref_ilamb.standard import (
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
