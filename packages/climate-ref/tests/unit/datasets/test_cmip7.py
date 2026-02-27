"""Tests for the CMIP7 dataset adapter."""

import pandas as pd

from climate_ref.datasets.cmip7 import (
    CMIP7DatasetAdapter,
)
from climate_ref.datasets.cmip7_parsers import (
    parse_cmip7_complete,
    parse_cmip7_drs,
    parse_cmip7_using_directories,
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

    def test_branded_variable_derived(self, cmip7_converted_file, config):
        """Test that branded_variable is derived as variable_id + branding_suffix."""
        adapter = CMIP7DatasetAdapter(config=config)
        data_catalog = adapter.find_local_datasets(cmip7_converted_file.parent)

        assert len(data_catalog) == 1
        row = data_catalog.iloc[0]

        assert "branded_variable" in data_catalog.columns
        assert row["branded_variable"] == f"{row['variable_id']}_{row['branding_suffix']}"


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
