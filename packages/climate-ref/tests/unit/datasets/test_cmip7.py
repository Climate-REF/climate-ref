"""Tests for the CMIP7 dataset adapter."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from climate_ref.database import Database
from climate_ref.datasets.cmip7 import (
    CMIP7DatasetAdapter,
)
from climate_ref.datasets.cmip7_parsers import (
    parse_cmip7_complete,
    parse_cmip7_drs,
    parse_cmip7_using_directories,
)
from climate_ref.models.dataset import CMIP7Dataset
from climate_ref_core.cmip6_to_cmip7 import (
    convert_cmip6_dataset,
    create_cmip7_filename,
    create_cmip7_path,
    format_cmip7_time_range,
)
from climate_ref_core.datasets import SourceDatasetType


@pytest.fixture
def cmip7_converted_file(sample_data_dir, tmp_path) -> Path:
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
    nc_files = list(cmip6_dir.rglob("**/tas_*.nc"))
    if not nc_files:
        pytest.skip("No CMIP6 netCDF files found in sample data")

    cmip6_file = nc_files[0]

    # Open and convert to CMIP7
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    with xr.open_dataset(cmip6_file, decode_times=time_coder) as ds:
        ds_cmip7 = convert_cmip6_dataset(ds)

        # Create filename
        time_range = format_cmip7_time_range(ds_cmip7, ds_cmip7.attrs["frequency"])
        cmip7_filename = create_cmip7_filename(ds_cmip7.attrs, time_range=time_range)

        cmip7_drs_path = create_cmip7_path(ds_cmip7.attrs)

        # Save to a simple output directory (not full DRS structure for easier testing)
        output_dir = tmp_path / cmip7_drs_path
        output_dir.mkdir(parents=True, exist_ok=True)
        output_file = output_dir / cmip7_filename

        # Save the converted file
        ds_cmip7.to_netcdf(output_file)

    return output_file


@pytest.fixture
def cmip7_converted_dir(cmip7_converted_file) -> Path:
    """Return the directory containing the converted CMIP7 file."""
    return cmip7_converted_file.parent


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

    def test_columns_requiring_finalisation(self):
        """Test that columns_requiring_finalisation contains the right set."""
        adapter = CMIP7DatasetAdapter()

        # These should NOT be in columns_requiring_finalisation because DRS provides them
        drs_available = {
            "activity_id",
            "institution_id",
            "source_id",
            "experiment_id",
            "variant_label",
            "variable_id",
            "grid_label",
            "frequency",
            "region",
            "branding_suffix",
            "version",
            "mip_era",
        }
        for col in drs_available:
            assert col not in adapter.columns_requiring_finalisation, (
                f"{col} should not require finalisation (available from DRS)"
            )

        # These SHOULD require finalisation (only available by opening files)
        requires_finalisation = {
            "realm",
            "nominal_resolution",
            "standard_name",
            "long_name",
            "units",
            "branch_time_in_child",
            "branch_time_in_parent",
            "parent_activity_id",
            "parent_experiment_id",
            "parent_mip_era",
            "parent_source_id",
            "parent_time_units",
            "parent_variant_label",
            "license_id",
            "external_variables",
        }
        assert adapter.columns_requiring_finalisation == requires_finalisation

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


def test_parse_cmip7_file_missing():
    """Test that parsing a missing file returns error info."""
    result = parse_cmip7_complete("nonexistent_file.nc")

    assert result["INVALID_ASSET"] == "nonexistent_file.nc"
    assert "TRACEBACK" in result


class TestCMIP7ConvertedFile:
    """Tests for CMIP7 adapter with files converted from CMIP6."""

    def test_parse_converted_cmip7_file(self, cmip7_converted_file):
        """Test that parse_cmip7_complete correctly parses a converted CMIP6 file."""
        # Parse the file
        result = parse_cmip7_complete(str(cmip7_converted_file))

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

    @pytest.mark.parametrize("cmip7_parser", ["complete", "drs"])
    def test_find_local_datasets(self, cmip7_parser, cmip7_converted_dir, config):
        """Test that find_local_datasets discovers CMIP7 files with both parsers."""
        config.cmip7_parser = cmip7_parser
        adapter = CMIP7DatasetAdapter(config=config)

        data_catalog = adapter.find_local_datasets(cmip7_converted_dir)

        # Should find exactly one dataset
        assert len(data_catalog) == 1, f"Expected 1 dataset, found {len(data_catalog)}"

        # Check that instance_id is correctly constructed
        row = data_catalog.iloc[0]
        assert row["instance_id"].startswith("CMIP7.")
        assert row["mip_era"] == "CMIP7"
        assert row["region"] == "glb"

        # Verify all dataset_specific_metadata fields are present
        for field_name in adapter.dataset_specific_metadata:
            assert field_name in data_catalog.columns, f"Missing field: {field_name}"

        # Verify all file_specific_metadata fields are present
        for field_name in adapter.file_specific_metadata:
            assert field_name in data_catalog.columns, f"Missing field: {field_name}"

        # Check finalised status based on parser type
        if cmip7_parser == "complete":
            assert data_catalog["finalised"].all()
        else:
            assert (~data_catalog["finalised"]).all()

    def test_drs_parser_extracts_core_drs_fields(self, cmip7_converted_file, config):
        """Test that the DRS parser extracts all core DRS fields from filename/directory."""
        config.cmip7_parser = "drs"
        adapter = CMIP7DatasetAdapter(config=config)

        data_catalog = adapter.find_local_datasets(cmip7_converted_file.parent)
        assert len(data_catalog) == 1

        row = data_catalog.iloc[0]

        # All DRS fields should be populated (not NA)
        drs_fields = [
            "variable_id",
            "branding_suffix",
            "frequency",
            "region",
            "grid_label",
            "source_id",
            "experiment_id",
            "variant_label",
            "activity_id",
            "institution_id",
            "version",
            "mip_era",
        ]
        for field_name in drs_fields:
            assert pd.notna(row[field_name]), (
                f"DRS field '{field_name}' should not be NA, got: {row[field_name]}"
            )

    def test_drs_parser_leaves_non_drs_as_na(self, cmip7_converted_file, config):
        """Test that the DRS parser leaves non-DRS fields as NA."""
        config.cmip7_parser = "drs"
        adapter = CMIP7DatasetAdapter(config=config)

        data_catalog = adapter.find_local_datasets(cmip7_converted_file.parent)
        assert len(data_catalog) == 1

        row = data_catalog.iloc[0]

        # Non-DRS fields should be NA (they require opening the file)
        for field_name in adapter.columns_requiring_finalisation:
            if field_name in data_catalog.columns:
                assert pd.isna(row[field_name]), (
                    f"Non-DRS field '{field_name}' should be NA for DRS parser, got: {row[field_name]}"
                )

    def test_drs_and_complete_produce_same_instance_id(self, cmip7_converted_dir, config):
        """Both parsers should produce the same instance_id for the same file."""
        adapter = CMIP7DatasetAdapter(config=config)

        config.cmip7_parser = "complete"
        complete_catalog = adapter.find_local_datasets(cmip7_converted_dir)

        config.cmip7_parser = "drs"
        drs_catalog = adapter.find_local_datasets(cmip7_converted_dir)

        assert set(complete_catalog["instance_id"]) == set(drs_catalog["instance_id"]), (
            f"instance_ids differ:\n  complete: {complete_catalog['instance_id'].tolist()}"
            f"\n  drs: {drs_catalog['instance_id'].tolist()}"
        )

    def test_branded_variable_derived(self, cmip7_converted_file, config):
        """Test that branded_variable is derived as variable_id + branding_suffix."""
        adapter = CMIP7DatasetAdapter(config=config)
        data_catalog = adapter.find_local_datasets(cmip7_converted_file.parent)

        assert len(data_catalog) == 1
        row = data_catalog.iloc[0]

        assert "branded_variable" in data_catalog.columns
        assert row["branded_variable"] == f"{row['variable_id']}_{row['branding_suffix']}"

    def test_validate_converted_catalog(self, cmip7_converted_dir, config):
        """Test that the converted file's catalog passes validation."""
        adapter = CMIP7DatasetAdapter(config=config)

        data_catalog = adapter.find_local_datasets(cmip7_converted_dir)

        # Validation should pass without raising
        validated = adapter.validate_data_catalog(data_catalog)
        assert len(validated) == 1


class TestGetParsingFunction:
    """Tests for CMIP7DatasetAdapter.get_parsing_function."""

    def test_returns_complete_parser(self, config):
        config.cmip7_parser = "complete"
        adapter = CMIP7DatasetAdapter(config=config)
        assert adapter.get_parsing_function() is parse_cmip7_complete

    def test_returns_drs_parser(self, config):
        config.cmip7_parser = "drs"
        adapter = CMIP7DatasetAdapter(config=config)
        assert adapter.get_parsing_function() is parse_cmip7_drs


class TestParseCMIP7UsingDirectories:
    """Tests for parse_cmip7_using_directories."""

    def test_parses_filename_with_time_range(self, tmp_path):
        """Parse a CMIP7 filename with 9 underscore-separated fields."""
        drs_dir = (
            tmp_path
            / "MIP-DRS7"
            / "CMIP7"
            / "CMIP"
            / "NCAR"
            / "CESM3"
            / "historical"
            / "r1i1p1f1"
            / "glb"
            / "mon"
            / "tas"
            / "tavg-h2m-hxy-u"
            / "gn"
            / "v20250622"
        )
        drs_dir.mkdir(parents=True)
        nc_file = drs_dir / "tas_tavg-h2m-hxy-u_mon_glb_gn_CESM3_historical_r1i1p1f1_185001-201412.nc"
        nc_file.touch()

        result = parse_cmip7_using_directories(str(nc_file))

        assert "INVALID_ASSET" not in result
        assert result["variable_id"] == "tas"
        assert result["branding_suffix"] == "tavg-h2m-hxy-u"
        assert result["frequency"] == "mon"
        assert result["region"] == "glb"
        assert result["grid_label"] == "gn"
        assert result["source_id"] == "CESM3"
        assert result["experiment_id"] == "historical"
        assert result["variant_label"] == "r1i1p1f1"
        assert result["time_range"] == "185001-201412"
        assert result["activity_id"] == "CMIP"
        assert result["institution_id"] == "NCAR"
        assert result["version"] == "v20250622"
        assert result["path"] == str(nc_file)

    def test_parses_filename_without_time_range(self, tmp_path):
        """Parse a CMIP7 filename without time range (fixed/time-invariant fields)."""
        drs_dir = (
            tmp_path
            / "MIP-DRS7"
            / "CMIP7"
            / "CMIP"
            / "NCAR"
            / "CESM3"
            / "historical"
            / "r1i1p1f1"
            / "glb"
            / "fx"
            / "areacella"
            / "tavg-hxy-u"
            / "gn"
            / "v20250622"
        )
        drs_dir.mkdir(parents=True)
        nc_file = drs_dir / "areacella_tavg-hxy-u_fx_glb_gn_CESM3_historical_r1i1p1f1.nc"
        nc_file.touch()

        result = parse_cmip7_using_directories(str(nc_file))

        assert "INVALID_ASSET" not in result
        assert result["variable_id"] == "areacella"
        assert result["frequency"] == "fx"
        assert "time_range" not in result

    def test_invalid_filename_too_few_parts(self, tmp_path):
        """Filenames with fewer than 8 fields are invalid."""
        nc_file = tmp_path / "too_few_parts.nc"
        nc_file.touch()

        result = parse_cmip7_using_directories(str(nc_file))

        assert "INVALID_ASSET" in result
        assert "Cannot parse CMIP7 filename" in result["TRACEBACK"]

    def test_invalid_filename_too_many_parts(self, tmp_path):
        """Filenames with more than 9 fields are invalid."""
        nc_file = tmp_path / "a_b_c_d_e_f_g_h_i_j.nc"
        nc_file.touch()

        result = parse_cmip7_using_directories(str(nc_file))

        assert "INVALID_ASSET" in result

    def test_version_extracted_from_directory(self, tmp_path):
        """Version is extracted from the v-prefixed directory component."""
        drs_dir = (
            tmp_path
            / "MIP-DRS7"
            / "CMIP7"
            / "CMIP"
            / "NCAR"
            / "CESM3"
            / "hist"
            / "r1"
            / "glb"
            / "mon"
            / "tas"
            / "tavg-h2m-hxy-u"
            / "gn"
            / "v20250101"
        )
        drs_dir.mkdir(parents=True)
        nc_file = drs_dir / "tas_tavg-h2m-hxy-u_mon_glb_gn_CESM3_hist_r1.nc"
        nc_file.touch()

        result = parse_cmip7_using_directories(str(nc_file))

        assert "INVALID_ASSET" not in result
        assert result["version"] == "v20250101"


class TestParseCmip7Drs:
    """Tests for parse_cmip7_drs (the DRS-only parser)."""

    def test_sets_finalised_false(self, tmp_path):
        """DRS parser always marks datasets as unfinalised."""
        drs_dir = (
            tmp_path
            / "MIP-DRS7"
            / "CMIP7"
            / "CMIP"
            / "NCAR"
            / "CESM3"
            / "hist"
            / "r1"
            / "glb"
            / "mon"
            / "tas"
            / "tavg-h2m-hxy-u"
            / "gn"
            / "v1"
        )
        drs_dir.mkdir(parents=True)
        nc_file = drs_dir / "tas_tavg-h2m-hxy-u_mon_glb_gn_CESM3_hist_r1_185001-201412.nc"
        nc_file.touch()

        result = parse_cmip7_drs(str(nc_file))

        assert "INVALID_ASSET" not in result
        assert result["finalised"] is False

    def test_sets_mip_era(self, tmp_path):
        """DRS parser sets mip_era to CMIP7."""
        drs_dir = (
            tmp_path
            / "MIP-DRS7"
            / "CMIP7"
            / "CMIP"
            / "NCAR"
            / "CESM3"
            / "hist"
            / "r1"
            / "glb"
            / "mon"
            / "tas"
            / "tavg-h2m-hxy-u"
            / "gn"
            / "v1"
        )
        drs_dir.mkdir(parents=True)
        nc_file = drs_dir / "tas_tavg-h2m-hxy-u_mon_glb_gn_CESM3_hist_r1_185001-201412.nc"
        nc_file.touch()

        result = parse_cmip7_drs(str(nc_file))

        assert result["mip_era"] == "CMIP7"

    def test_parses_time_range(self, tmp_path):
        """DRS parser converts time range from filename to start/end dates."""
        drs_dir = (
            tmp_path
            / "MIP-DRS7"
            / "CMIP7"
            / "CMIP"
            / "NCAR"
            / "CESM3"
            / "hist"
            / "r1"
            / "glb"
            / "mon"
            / "tas"
            / "tavg-h2m-hxy-u"
            / "gn"
            / "v1"
        )
        drs_dir.mkdir(parents=True)
        nc_file = drs_dir / "tas_tavg-h2m-hxy-u_mon_glb_gn_CESM3_hist_r1_185001-201412.nc"
        nc_file.touch()

        result = parse_cmip7_drs(str(nc_file))

        assert result["start_time"] == "1850-01-01"
        assert result["end_time"] == "2014-12-30"

    def test_no_time_range_for_fixed_fields(self, tmp_path):
        """DRS parser handles files without a time range (fixed/fx fields)."""
        drs_dir = (
            tmp_path
            / "MIP-DRS7"
            / "CMIP7"
            / "CMIP"
            / "NCAR"
            / "CESM3"
            / "hist"
            / "r1"
            / "glb"
            / "fx"
            / "areacella"
            / "tavg-hxy-u"
            / "gn"
            / "v1"
        )
        drs_dir.mkdir(parents=True)
        nc_file = drs_dir / "areacella_tavg-hxy-u_fx_glb_gn_CESM3_hist_r1.nc"
        nc_file.touch()

        result = parse_cmip7_drs(str(nc_file))

        assert "INVALID_ASSET" not in result
        assert "start_time" not in result
        assert "end_time" not in result

    def test_invalid_file_returns_invalid_asset(self, tmp_path):
        """DRS parser returns INVALID_ASSET for unparseable filenames."""
        nc_file = tmp_path / "invalid.nc"
        nc_file.touch()

        result = parse_cmip7_drs(str(nc_file))

        assert "INVALID_ASSET" in result

    @pytest.mark.parametrize("parsing_func", [parse_cmip7_complete, parse_cmip7_drs])
    def test_parse_exception(self, parsing_func):
        """Both parsers return INVALID_ASSET for missing files."""
        result = parsing_func("missing_file.nc")

        assert result["INVALID_ASSET"] == "missing_file.nc"
        assert "TRACEBACK" in result


class TestCMIP7RoundTrip:
    """Tests for CMIP7 dataset round-trip: ingest -> register -> load_catalog."""

    @pytest.mark.parametrize("cmip7_parser", ["complete", "drs"])
    def test_round_trip(self, cmip7_parser, config, cmip7_converted_dir):
        """Ingest, register, and reload a CMIP7 dataset - verify DataFrame equality."""
        config.cmip7_parser = cmip7_parser
        adapter = CMIP7DatasetAdapter(config=config)
        catalog = adapter.find_local_datasets(cmip7_converted_dir)

        with Database.from_config(config, run_migrations=True) as database:
            with database.session.begin():
                for instance_id, data_catalog_dataset in catalog.groupby(adapter.slug_column):
                    adapter.register_dataset(database, data_catalog_dataset)

            # Drop columns that are not round-trippable through the DB:
            # - time_range: synthetic, not stored
            # - branded_variable: derived, not stored
            # - tracking_id: file-level metadata not persisted to the DB
            non_roundtrip_cols = ["time_range", "branded_variable", "tracking_id"]
            local_data_catalog = (
                catalog.drop(columns=non_roundtrip_cols, errors="ignore")
                .sort_values(["instance_id", "start_time"])
                .reset_index(drop=True)
            )

            db_data_catalog = (
                adapter.load_catalog(database)
                .drop(columns=non_roundtrip_cols, errors="ignore")
                .sort_values(["instance_id", "start_time"])
                .reset_index(drop=True)
            )

            # Normalize null values for consistent comparison
            with pd.option_context("future.no_silent_downcasting", True):
                local_normalized = local_data_catalog.fillna(np.nan).infer_objects()
                db_normalized = db_data_catalog.fillna(np.nan).infer_objects()

            pd.testing.assert_frame_equal(
                local_normalized,
                db_normalized,
                check_like=True,
            )

    def test_load_catalog_has_all_columns(self, config, cmip7_converted_dir):
        """Verify that load_catalog returns all required metadata columns."""
        config.cmip7_parser = "complete"
        adapter = CMIP7DatasetAdapter(config=config)
        catalog = adapter.find_local_datasets(cmip7_converted_dir)

        with Database.from_config(config, run_migrations=True) as database:
            with database.session.begin():
                for instance_id, group in catalog.groupby(adapter.slug_column):
                    adapter.register_dataset(database, group)

            db_catalog = adapter.load_catalog(database)
            for k in adapter.dataset_specific_metadata + adapter.file_specific_metadata:
                assert k in db_catalog.columns, f"Missing column: {k}"


class TestCMIP7EndToEndFinalisation:
    """End-to-end test: DRS ingest -> register -> finalise with real files."""

    def test_finalise_datasets_with_real_files(self, config, cmip7_converted_dir):
        """Full two-phase flow: DRS ingest, register, then finalise with complete parser."""
        config.cmip7_parser = "drs"
        adapter = CMIP7DatasetAdapter(config=config)

        # Phase 1: DRS ingest (fast, no file I/O)
        drs_catalog = adapter.find_local_datasets(cmip7_converted_dir)
        assert (~drs_catalog["finalised"]).all(), "DRS parser should produce unfinalised datasets"

        with Database.from_config(config, run_migrations=True) as database:
            # Register unfinalised datasets
            with database.session.begin():
                for _instance_id, group in drs_catalog.groupby(adapter.slug_column):
                    adapter.register_dataset(database, group)

            # Load from DB - should be unfinalised
            db_catalog = adapter.load_catalog(database)
            assert not db_catalog["finalised"].any(), "DB catalog should have unfinalised datasets"

            # Phase 2: Finalise (opens files, extracts full metadata)
            target_instance = db_catalog["instance_id"].iloc[0]
            subset = db_catalog[db_catalog["instance_id"] == target_instance].copy()

            result = adapter.finalise_datasets(database, subset)

            # Verify finalised
            assert result["finalised"].all(), "Finalised datasets should have finalised=True"

            # Verify non-DRS fields are now populated
            assert result["source_id"].notna().all()
            assert result["experiment_id"].notna().all()

            # These fields require opening the file - should now be populated
            # (they were NA after DRS parse)
            for field in ["realm", "standard_name", "long_name", "units"]:
                val = result[field].iloc[0]
                assert pd.notna(val), f"Field '{field}' should be populated after finalisation, got NA"

    def test_drs_then_complete_produces_same_core_metadata(self, config, cmip7_converted_dir):
        """DRS and complete parsers produce the same values for core DRS fields."""
        adapter = CMIP7DatasetAdapter(config=config)

        config.cmip7_parser = "drs"
        drs_catalog = adapter.find_local_datasets(cmip7_converted_dir)

        config.cmip7_parser = "complete"
        complete_catalog = adapter.find_local_datasets(cmip7_converted_dir)

        # Core DRS fields should match exactly
        drs_fields = [*adapter.dataset_id_metadata, "version", "mip_era"]
        for field in drs_fields:
            drs_val = drs_catalog[field].iloc[0]
            complete_val = complete_catalog[field].iloc[0]
            assert str(drs_val) == str(complete_val), (
                f"Field '{field}' differs: DRS={drs_val!r} vs complete={complete_val!r}"
            )
