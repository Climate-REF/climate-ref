"""Tests for the CMIP7 dataset adapter."""

import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climate_ref.datasets.cmip7 import (
    CMIP7DatasetAdapter,
    _add_branded_variable_name,
    parse_cmip7_file,
)
from climate_ref.datasets.utils import clean_branch_time, parse_datetime
from climate_ref.models.dataset import CMIP7Dataset
from climate_ref_core.cmip6_to_cmip7 import (
    convert_cmip6_dataset,
    create_cmip7_filename,
)
from climate_ref_core.datasets import SourceDatasetType


class TestCMIP7Adapter:
    """Tests for CMIP7DatasetAdapter."""

    def test_adapter_initialization(self, config):
        """Test that the adapter initializes correctly."""
        adapter = CMIP7DatasetAdapter(config=config)

        assert adapter.dataset_cls == CMIP7Dataset
        assert adapter.slug_column == "instance_id"
        assert adapter.version_metadata == "version"

    def test_dataset_specific_metadata(self):
        """Test that dataset_specific_metadata contains expected fields."""
        adapter = CMIP7DatasetAdapter()

        # Core DRS attributes
        assert "activity_id" in adapter.dataset_specific_metadata
        assert "institution_id" in adapter.dataset_specific_metadata
        assert "source_id" in adapter.dataset_specific_metadata
        assert "experiment_id" in adapter.dataset_specific_metadata
        assert "variant_label" in adapter.dataset_specific_metadata
        assert "variable_id" in adapter.dataset_specific_metadata
        assert "grid_label" in adapter.dataset_specific_metadata
        assert "frequency" in adapter.dataset_specific_metadata
        assert "region" in adapter.dataset_specific_metadata
        assert "branding_suffix" in adapter.dataset_specific_metadata
        assert "version" in adapter.dataset_specific_metadata

        # Additional mandatory attributes
        assert "mip_era" in adapter.dataset_specific_metadata
        assert "realm" in adapter.dataset_specific_metadata
        assert "nominal_resolution" in adapter.dataset_specific_metadata

        # Parent info
        assert "branch_time_in_child" in adapter.dataset_specific_metadata
        assert "branch_time_in_parent" in adapter.dataset_specific_metadata
        assert "parent_activity_id" in adapter.dataset_specific_metadata
        assert "parent_experiment_id" in adapter.dataset_specific_metadata
        assert "parent_mip_era" in adapter.dataset_specific_metadata
        assert "parent_source_id" in adapter.dataset_specific_metadata
        assert "parent_time_units" in adapter.dataset_specific_metadata
        assert "parent_variant_label" in adapter.dataset_specific_metadata

        # Variable metadata
        assert "standard_name" in adapter.dataset_specific_metadata
        assert "long_name" in adapter.dataset_specific_metadata
        assert "units" in adapter.dataset_specific_metadata

        # Unique identifier
        assert "instance_id" in adapter.dataset_specific_metadata

    def test_file_specific_metadata(self):
        """Test that file_specific_metadata contains expected fields."""
        adapter = CMIP7DatasetAdapter()

        assert "start_time" in adapter.file_specific_metadata
        assert "end_time" in adapter.file_specific_metadata
        assert "path" in adapter.file_specific_metadata
        assert "tracking_id" in adapter.file_specific_metadata

    def test_dataset_id_metadata(self):
        """Test that dataset_id_metadata follows CMIP7 DRS format."""
        adapter = CMIP7DatasetAdapter()

        # CMIP7 DRS order for instance_id construction
        expected = (
            "activity_id",
            "institution_id",
            "source_id",
            "experiment_id",
            "variant_label",
            "region",
            "frequency",
            "variable_id",
            "branding_suffix",
            "grid_label",
        )
        assert adapter.dataset_id_metadata == expected

    def test_catalog_empty(self, db):
        """Test that an empty database returns an empty catalog."""
        adapter = CMIP7DatasetAdapter()
        df = adapter.load_catalog(db)
        assert df.empty

    def test_instance_id_construction(self):
        """Test that instance_id is constructed correctly from DRS components."""
        adapter = CMIP7DatasetAdapter()

        # Create a mock dataframe with CMIP7 metadata
        data = {
            "activity_id": ["CMIP"],
            "institution_id": ["NCAR"],
            "source_id": ["CESM3"],
            "experiment_id": ["historical"],
            "variant_label": ["r1i1p1f1"],
            "region": ["glb"],
            "frequency": ["mon"],
            "variable_id": ["tas"],
            "branding_suffix": ["tavg-h2m-hxy-u"],
            "grid_label": ["gn"],
            "version": ["v20250622"],
            "mip_era": ["CMIP7"],
            "realm": ["atmos"],
            "nominal_resolution": ["100 km"],
            "branch_time_in_child": [None],
            "branch_time_in_parent": [None],
            "parent_activity_id": [None],
            "parent_experiment_id": [None],
            "parent_mip_era": [None],
            "parent_source_id": [None],
            "parent_time_units": [None],
            "parent_variant_label": [None],
            "standard_name": ["air_temperature"],
            "long_name": ["Near-Surface Air Temperature"],
            "units": ["K"],
            "start_time": [None],
            "end_time": [None],
            "path": ["/path/to/file.nc"],
            "tracking_id": [None],
        }
        df = pd.DataFrame(data)

        # Manually construct what the instance_id should be
        drs_items = [
            *adapter.dataset_id_metadata,
            adapter.version_metadata,
        ]
        expected_instance_id = "CMIP7." + ".".join([str(df[item].iloc[0]) for item in drs_items])

        # The adapter applies this transformation in find_local_datasets
        df["instance_id"] = df.apply(
            lambda row: "CMIP7." + ".".join([str(row[item]) for item in drs_items]), axis=1
        )

        assert df["instance_id"].iloc[0] == expected_instance_id
        assert (
            df["instance_id"].iloc[0]
            == "CMIP7.CMIP.NCAR.CESM3.historical.r1i1p1f1.glb.mon.tas.tavg-h2m-hxy-u.gn.v20250622"
        )


class TestCMIP7Model:
    """Tests for CMIP7Dataset SQLAlchemy model."""

    def test_polymorphic_identity(self):
        """Test that CMIP7Dataset has correct polymorphic identity."""
        assert CMIP7Dataset.__mapper_args__["polymorphic_identity"] == SourceDatasetType.CMIP7

    def test_table_name(self):
        """Test that CMIP7Dataset has correct table name."""
        assert CMIP7Dataset.__tablename__ == "cmip7_dataset"

    def test_model_creation(self, db):
        """Test that a CMIP7Dataset can be created in the database."""
        dataset = CMIP7Dataset(
            slug="CMIP7.CMIP.NCAR.CESM3.historical.r1i1p1f1.glb.mon.tas.tavg-h2m-hxy-u.gn.v20250622",
            activity_id="CMIP",
            institution_id="NCAR",
            source_id="CESM3",
            experiment_id="historical",
            variant_label="r1i1p1f1",
            variable_id="tas",
            grid_label="gn",
            frequency="mon",
            region="glb",
            branding_suffix="tavg-h2m-hxy-u",
            version="v20250622",
            mip_era="CMIP7",
            instance_id="CMIP7.CMIP.NCAR.CESM3.historical.r1i1p1f1.glb.mon.tas.tavg-h2m-hxy-u.gn.v20250622",
        )

        with db.session.begin():
            db.session.add(dataset)
            db.session.flush()

            assert dataset.id is not None
            assert dataset.dataset_type == SourceDatasetType.CMIP7

        # Verify the dataset was persisted
        with db.session.begin():
            retrieved = db.session.query(CMIP7Dataset).filter_by(slug=dataset.slug).first()
            assert retrieved is not None
            assert retrieved.source_id == "CESM3"
            assert retrieved.experiment_id == "historical"
            assert retrieved.mip_era == "CMIP7"


class TestCMIP7HelperFunctions:
    """Tests for CMIP7 adapter helper functions."""

    def testclean_branch_time(self):
        """Test branch time cleaning handles various formats."""
        inp = pd.Series(["0D", "12", "12.0", "12.000", None, np.nan])
        result = clean_branch_time(inp)

        assert result.iloc[0] == 0.0
        assert result.iloc[1] == 12.0
        assert result.iloc[2] == 12.0
        assert result.iloc[3] == 12.0
        assert pd.isna(result.iloc[4])
        assert pd.isna(result.iloc[5])

    def testparse_datetime_valid(self):
        """Test datetime parsing with valid inputs."""
        inp = pd.Series(["2025-01-01", "2025-06-15 12:30:00", None])
        result = parse_datetime(inp)

        assert result.iloc[0] == datetime.datetime(2025, 1, 1)
        assert result.iloc[1] == datetime.datetime(2025, 6, 15, 12, 30, 0)
        assert result.iloc[2] is None

    def test_parse_cmip7_file_missing(self):
        """Test that parsing a missing file returns error info."""
        result = parse_cmip7_file("nonexistent_file.nc")

        assert result["INVALID_ASSET"] == "nonexistent_file.nc"
        assert "TRACEBACK" in result


class TestCMIP7ConvertedFile:
    """Tests for CMIP7 adapter with files converted from CMIP6."""

    @pytest.fixture
    def cmip7_converted_file(self, sample_data_dir, tmp_path) -> Path:
        """
        Convert a CMIP6 file to CMIP7 format and save it.

        Uses the cmip6_to_cmip7 converter from climate_ref_core.
        Returns the path to the converted file.
        """
        # Find a CMIP6 file from sample data
        cmip6_dir = sample_data_dir / "CMIP6"
        if not cmip6_dir.exists():
            pytest.skip("CMIP6 sample data not available")

        # Find the first .nc file
        nc_files = list(cmip6_dir.rglob("*.nc"))
        if not nc_files:
            pytest.skip("No CMIP6 netCDF files found in sample data")

        cmip6_file = nc_files[0]

        # Open and convert to CMIP7
        with xr.open_dataset(cmip6_file, use_cftime=True) as ds:
            ds_cmip7 = convert_cmip6_dataset(ds)

            # Add version attribute (required for CMIP7 DRS)
            ds_cmip7.attrs["version"] = "v20250101"

            # Create filename with time range (simplified - just use placeholder)
            cmip7_filename = create_cmip7_filename(ds_cmip7.attrs, time_range="185001-185012")

            # Save to a simple output directory (not full DRS structure for easier testing)
            output_dir = tmp_path / "CMIP7"
            output_dir.mkdir(parents=True, exist_ok=True)
            output_file = output_dir / cmip7_filename

            # Save the converted file
            ds_cmip7.to_netcdf(output_file)

        return output_file

    @pytest.fixture
    def cmip7_converted_dir(self, cmip7_converted_file) -> Path:
        """Return the directory containing the converted CMIP7 file."""
        return cmip7_converted_file.parent

    def test_parse_converted_cmip7_file(self, cmip7_converted_file):
        """Test that parse_cmip7_file correctly parses a converted CMIP6 file."""
        # Parse the file
        result = parse_cmip7_file(str(cmip7_converted_file))

        # Verify key CMIP7 attributes are present
        assert "INVALID_ASSET" not in result, f"Parsing failed: {result.get('TRACEBACK', '')}"
        assert result["mip_era"] == "CMIP7"
        assert result["region"] == "glb"
        assert "branding_suffix" in result
        assert result["branding_suffix"] != ""
        assert "tracking_id" in result
        assert result["tracking_id"].startswith("hdl:21.14107/")

        # Verify DRS components are present
        assert result["activity_id"] != ""
        assert result["institution_id"] != ""
        assert result["source_id"] != ""
        assert result["experiment_id"] != ""
        assert result["variable_id"] != ""
        assert result["grid_label"] != ""
        assert result["frequency"] != ""

    def test_find_local_datasets_converted(self, cmip7_converted_dir, config):
        """Test that find_local_datasets correctly discovers converted CMIP7 files."""
        adapter = CMIP7DatasetAdapter(config=config)

        # Find datasets in the converted directory
        data_catalog = adapter.find_local_datasets(cmip7_converted_dir)

        # Should find exactly one dataset
        assert len(data_catalog) == 1, f"Expected 1 dataset, found {len(data_catalog)}"

        # Check that instance_id is correctly constructed
        row = data_catalog.iloc[0]
        assert row["instance_id"].startswith("CMIP7.")
        assert row["mip_era"] == "CMIP7"
        assert row["region"] == "glb"

        # Verify all dataset_specific_metadata fields are present
        for field in adapter.dataset_specific_metadata:
            assert field in data_catalog.columns, f"Missing field: {field}"

        # Verify all file_specific_metadata fields are present
        for field in adapter.file_specific_metadata:
            assert field in data_catalog.columns, f"Missing field: {field}"

    def test_branded_variable_name_uses_out_name(self, cmip7_converted_file, config):
        """Test that branded_variable_name uses out_name from DReq, not variable_id.

        Converted CMIP7 files store branded_variable as an attribute using
        out_name from the Data Request (e.g., 'tas_tmaxavg-h2m-hxy-u' for tasmax).
        The catalog should use this attribute rather than computing
        variable_id + '_' + branding_suffix (which would give 'tasmax_tmaxavg-h2m-hxy-u').
        """
        adapter = CMIP7DatasetAdapter(config=config)
        data_catalog = adapter.find_local_datasets(cmip7_converted_file.parent)

        assert len(data_catalog) == 1
        row = data_catalog.iloc[0]

        # The branded_variable_name should match the branded_variable attribute
        # from the file (which uses out_name from DReq)
        assert "branded_variable_name" in data_catalog.columns
        assert row["branded_variable_name"] != ""

        # Read the branded_variable attribute directly from the file to verify
        result = parse_cmip7_file(str(cmip7_converted_file))
        if result.get("branded_variable"):
            assert row["branded_variable_name"] == result["branded_variable"]

    def test_branded_variable_name_fallback(self):
        """Test that branded_variable_name falls back to variable_id + branding_suffix."""
        # Catalog without branded_variable column
        catalog = pd.DataFrame(
            {
                "variable_id": ["tas", "pr"],
                "branding_suffix": ["tavg-h2m-hxy-u", "tavg-u-hxy-u"],
            }
        )
        result = _add_branded_variable_name(catalog)
        assert result["branded_variable_name"].tolist() == ["tas_tavg-h2m-hxy-u", "pr_tavg-u-hxy-u"]

    def test_branded_variable_name_prefers_file_attribute(self):
        """Test that branded_variable from file is preferred over computed value.

        When variable_id differs from out_name (e.g. tasmax vs tas),
        the branded_variable attribute uses out_name which is correct.
        """
        catalog = pd.DataFrame(
            {
                "variable_id": ["tasmax", "pr"],
                "branding_suffix": ["tmaxavg-h2m-hxy-u", "tavg-u-hxy-u"],
                "branded_variable": ["tas_tmaxavg-h2m-hxy-u", "pr_tavg-u-hxy-u"],
            }
        )
        result = _add_branded_variable_name(catalog)
        # Should use branded_variable (out_name-based), not variable_id-based
        assert result["branded_variable_name"].tolist() == [
            "tas_tmaxavg-h2m-hxy-u",
            "pr_tavg-u-hxy-u",
        ]

    def test_branded_variable_name_partial_fallback(self):
        """Test fallback when some rows have branded_variable and some don't."""
        catalog = pd.DataFrame(
            {
                "variable_id": ["tasmax", "pr"],
                "branding_suffix": ["tmaxavg-h2m-hxy-u", "tavg-u-hxy-u"],
                "branded_variable": ["tas_tmaxavg-h2m-hxy-u", ""],
            }
        )
        result = _add_branded_variable_name(catalog)
        assert result["branded_variable_name"].tolist() == [
            "tas_tmaxavg-h2m-hxy-u",
            "pr_tavg-u-hxy-u",
        ]

    def test_validate_converted_catalog(self, cmip7_converted_dir, config):
        """Test that the converted file's catalog passes validation."""
        adapter = CMIP7DatasetAdapter(config=config)

        data_catalog = adapter.find_local_datasets(cmip7_converted_dir)

        # Validation should pass without raising
        validated = adapter.validate_data_catalog(data_catalog)
        assert len(validated) == 1
