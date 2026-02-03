"""Tests for the CMIP7 dataset adapter."""

import datetime

import numpy as np
import pandas as pd

from climate_ref.datasets.cmip7 import (
    CMIP7DatasetAdapter,
    _clean_branch_time,
    _parse_datetime,
    parse_cmip7_file,
)
from climate_ref.models.dataset import CMIP7Dataset
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

    def test_clean_branch_time(self):
        """Test branch time cleaning handles various formats."""
        inp = pd.Series(["0D", "12", "12.0", "12.000", None, np.nan])
        result = _clean_branch_time(inp)

        assert result.iloc[0] == 0.0
        assert result.iloc[1] == 12.0
        assert result.iloc[2] == 12.0
        assert result.iloc[3] == 12.0
        assert pd.isna(result.iloc[4])
        assert pd.isna(result.iloc[5])

    def test_parse_datetime_valid(self):
        """Test datetime parsing with valid inputs."""
        inp = pd.Series(["2025-01-01", "2025-06-15 12:30:00", None])
        result = _parse_datetime(inp)

        assert result.iloc[0] == datetime.datetime(2025, 1, 1)
        assert result.iloc[1] == datetime.datetime(2025, 6, 15, 12, 30, 0)
        assert result.iloc[2] is None

    def test_parse_cmip7_file_missing(self):
        """Test that parsing a missing file returns error info."""
        result = parse_cmip7_file("nonexistent_file.nc")

        assert result["INVALID_ASSET"] == "nonexistent_file.nc"
        assert "TRACEBACK" in result
