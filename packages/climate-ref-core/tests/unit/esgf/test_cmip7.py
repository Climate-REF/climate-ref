from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climate_ref_core.esgf.cmip7 import CMIP7Request, _convert_file


class TestCMIP7Request:
    def test_init_default_output_dir(self):
        request = CMIP7Request(
            slug="test",
            facets={"variable_id": "tas"},
        )
        assert request.output_dir == Path.home() / ".cache" / "climate-ref" / "cmip7"

    def test_init_custom_output_dir(self, tmp_path):
        request = CMIP7Request(
            slug="test",
            facets={"variable_id": "tas"},
            output_dir=tmp_path / "cmip7",
        )
        assert request.output_dir == tmp_path / "cmip7"

    def test_source_type(self):
        request = CMIP7Request(
            slug="test",
            facets={"variable_id": "tas"},
        )
        assert request.source_type == "CMIP7"

    def test_repr(self):
        request = CMIP7Request(
            slug="test",
            facets={"variable_id": "tas", "source_id": "ACCESS-ESM1-5"},
        )
        repr_str = repr(request)
        assert "CMIP7Request" in repr_str
        assert "test" in repr_str

    def test_internal_cmip6_request_created(self):
        request = CMIP7Request(
            slug="test-request",
            facets={"variable_id": "tas", "experiment_id": "historical"},
            remove_ensembles=True,
            time_span=("2000-01", "2010-12"),
        )
        assert request._cmip6_request.slug == "test-request_cmip6_source"
        assert request._cmip6_request.facets == {"variable_id": "tas", "experiment_id": "historical"}
        assert request._cmip6_request.remove_ensembles is True
        assert request._cmip6_request.time_span == ("2000-01", "2010-12")

    def test_build_cmip7_metadata(self):
        request = CMIP7Request(
            slug="test",
            facets={"variable_id": "tas"},
        )

        cmip6_row = pd.Series(
            {
                "variable_id": "tas",
                "table_id": "Amon",
                "source_id": "ACCESS-ESM1-5",
                "experiment_id": "historical",
                "variant_label": "r1i1p1f1",
                "institution_id": "CSIRO",
                "activity_id": "CMIP",
                "grid_label": "gn",
                "version": "v20191115",
            }
        )
        cmip7_files = ["/path/to/converted/file.nc"]
        cmip7_attrs = {
            "region": "GLB",
            "archive_id": "WCRP",
            "host_collection": "CMIP7",
            "drs_specs": "MIP-DRS7",
            "cv_version": "7.0.0.0",
            "temporal_label": "tavg",
            "vertical_label": "h2m",
            "horizontal_label": "hxy",
            "area_label": "u",
        }

        result = request._build_cmip7_metadata(cmip6_row, cmip7_files, cmip7_attrs)

        assert result["mip_era"] == "CMIP7"
        assert result["table_id"] == "atmos"  # Converted from Amon
        assert result["frequency"] == "mon"  # Extracted from Amon
        assert result["region"] == "GLB"
        assert result["branding_suffix"] == "tavg-h2m-hxy-u"
        assert result["path"] == "/path/to/converted/file.nc"
        assert result["files"] == cmip7_files
        # Key should be CMIP7 instance_id format
        assert "CMIP7" in result["key"]

    @patch("climate_ref_core.esgf.cmip7.CMIP6Request")
    def test_fetch_datasets_empty(self, mock_cmip6_request_cls):
        mock_cmip6_request = MagicMock()
        mock_cmip6_request.fetch_datasets.return_value = pd.DataFrame()
        mock_cmip6_request_cls.return_value = mock_cmip6_request

        request = CMIP7Request(
            slug="test",
            facets={"variable_id": "tas"},
        )
        request._cmip6_request = mock_cmip6_request

        result = request.fetch_datasets()

        assert result.empty


class TestConvertFile:
    def test_convert_file_missing_file(self, tmp_path):
        result = _convert_file(
            cmip6_path=tmp_path / "nonexistent.nc",
            output_dir=tmp_path / "output",
            rename_variables=False,
        )
        assert result is None

    def test_convert_file_success(self, tmp_path, sample_cmip6_dataset):
        # Write sample dataset to file
        input_file = tmp_path / "input" / "tas_Amon_test.nc"
        input_file.parent.mkdir(parents=True, exist_ok=True)
        sample_cmip6_dataset.to_netcdf(input_file)

        output_dir = tmp_path / "output"

        result = _convert_file(
            cmip6_path=input_file,
            output_dir=output_dir,
            rename_variables=False,
        )

        assert result is not None
        cmip7_path, cmip7_attrs = result
        assert cmip7_path.exists()
        assert cmip7_attrs["mip_era"] == "CMIP7"
        assert cmip7_attrs["branding_suffix"] == "tavg-h2m-hxy-u"

        # Verify the converted file has correct attributes
        ds = xr.open_dataset(cmip7_path)
        assert ds.attrs["mip_era"] == "CMIP7"
        ds.close()

    def test_convert_file_already_exists(self, tmp_path, sample_cmip6_dataset):
        # Write sample dataset to file
        input_file = tmp_path / "input" / "tas_Amon_test.nc"
        input_file.parent.mkdir(parents=True, exist_ok=True)
        sample_cmip6_dataset.to_netcdf(input_file)

        output_dir = tmp_path / "output"

        # First conversion
        result1 = _convert_file(input_file, output_dir, rename_variables=False)
        assert result1 is not None
        cmip7_path1, _ = result1
        mtime1 = cmip7_path1.stat().st_mtime

        # Second conversion - should skip writing
        result2 = _convert_file(input_file, output_dir, rename_variables=False)
        assert result2 is not None
        cmip7_path2, _ = result2
        mtime2 = cmip7_path2.stat().st_mtime

        # File should not have been rewritten
        assert mtime1 == mtime2


@pytest.fixture
def sample_cmip6_dataset() -> xr.Dataset:
    """Create a minimal CMIP6-style dataset for testing."""

    time = np.arange(12)
    lat = np.linspace(-90, 90, 5)
    lon = np.linspace(0, 360, 10)
    rng = np.random.default_rng(42)

    data = rng.random((len(time), len(lat), len(lon)))

    ds = xr.Dataset(
        {"tas": (["time", "lat", "lon"], data)},
        coords={
            "time": time,
            "lat": lat,
            "lon": lon,
        },
        attrs={
            "variable_id": "tas",
            "table_id": "Amon",
            "source_id": "ACCESS-ESM1-5",
            "experiment_id": "historical",
            "variant_label": "r1i1p1f1",
            "member_id": "r1i1p1f1",
            "institution_id": "CSIRO",
            "activity_id": "CMIP",
            "grid_label": "gn",
            "version": "v20191115",
            "Conventions": "CF-1.6",
        },
    )
    return ds
