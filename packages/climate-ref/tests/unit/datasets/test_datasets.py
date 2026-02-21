from pathlib import Path

import pandas as pd
import pytest

from climate_ref.config import Config
from climate_ref.database import Database, ModelState
from climate_ref.datasets import IngestionStats, get_dataset_adapter, ingest_datasets
from climate_ref.datasets import base as base_module
from climate_ref.datasets.base import DatasetAdapter
from climate_ref.datasets.cmip6 import CMIP6DatasetAdapter
from climate_ref.models.dataset import CMIP6Dataset, DatasetFile
from climate_ref_core.datasets import SourceDatasetType
from climate_ref_core.exceptions import RefException


class MockDatasetAdapter(DatasetAdapter):
    dataset_cls = CMIP6Dataset
    slug_column: str = "variable_id"  # Use variable_id as the slug for this mock
    dataset_specific_metadata: tuple[str, ...] = ("variable_id", "source_id", "grid_label")
    file_specific_metadata: tuple[str, ...] = ("start_time", "end_time", "path")

    def pretty_subset(self, data_catalog: pd.DataFrame) -> pd.DataFrame:
        # Return a subset with the most important columns for display
        return data_catalog[["variable_id", "source_id", "grid_label"]]

    def find_local_datasets(self, file_or_directory: Path) -> pd.DataFrame:
        # Mock implementation with more realistic CMIP6-like data
        data = {
            "variable_id": ["tas", "tas"],  # Temperature at surface
            "source_id": ["CESM2", "CESM2"],  # Model name
            "grid_label": ["gn", "gn"],  # Native grid
            "dataset_slug": ["tas_CESM2_gn", "tas_CESM2_gn"],  # Dataset identifier
            "start_time": [pd.Timestamp("2001-01-01"), pd.Timestamp("2002-01-01")],
            "end_time": [pd.Timestamp("2001-12-31"), pd.Timestamp("2002-12-31")],
            "time_range": ["2001-01-01/2001-12-31", "2002-01-01/2002-12-31"],
            "path": [f"{file_or_directory.name}_2001.nc", f"{file_or_directory.name}_2002.nc"],
        }

        return pd.DataFrame(data)


def test_validate_data_catalog_complete_data():
    adapter = MockDatasetAdapter()
    data_catalog = adapter.find_local_datasets(Path("path/to/dataset"))

    validated_catalog = adapter.validate_data_catalog(data_catalog)
    assert not validated_catalog.empty


def test_validate_data_catalog_extra_columns():
    adapter = MockDatasetAdapter()
    data_catalog = adapter.find_local_datasets(Path("path/to/dataset"))
    data_catalog["extra_column"] = "extra"

    adapter.validate_data_catalog(data_catalog)


def test_validate_data_catalog_missing_columns():
    adapter = MockDatasetAdapter()
    data_catalog = adapter.find_local_datasets(Path("path/to/dataset"))
    with pytest.raises(ValueError, match=r"Data catalog is missing required columns: {'source_id'}"):
        adapter.validate_data_catalog(data_catalog.drop(columns=["source_id"]))

    with pytest.raises(ValueError, match=r"Data catalog is missing required columns: {'path'}"):
        adapter.validate_data_catalog(data_catalog.drop(columns=["path"]))


def test_validate_data_catalog_metadata_variance(caplog):
    adapter = MockDatasetAdapter()
    data_catalog = adapter.find_local_datasets(Path("path/to/dataset"))
    # file_name differs between datasets
    adapter.dataset_specific_metadata = (*adapter.dataset_specific_metadata, "path")

    exp_message = (
        "Dataset tas has varying metadata:\n"
        "              path             time_range\n"
        "0  dataset_2001.nc  2001-01-01/2001-12-31\n"
        "1  dataset_2002.nc  2002-01-01/2002-12-31"
    )

    with pytest.raises(
        ValueError,
        match="Dataset specific metadata varies by dataset",
    ):
        adapter.validate_data_catalog(data_catalog)
    assert len(caplog.records) == 1
    assert caplog.records[0].message == exp_message

    caplog.clear()
    assert len(adapter.validate_data_catalog(data_catalog, skip_invalid=True)) == 0
    assert len(caplog.records) == 1
    assert caplog.records[0].message == exp_message


@pytest.mark.parametrize(
    "source_type, expected_adapter",
    [
        (SourceDatasetType.CMIP6.value, "climate_ref.datasets.cmip6.CMIP6DatasetAdapter"),
        (SourceDatasetType.CMIP7.value, "climate_ref.datasets.cmip7.CMIP7DatasetAdapter"),
        (SourceDatasetType.obs4MIPs.value, "climate_ref.datasets.obs4mips.Obs4MIPsDatasetAdapter"),
    ],
)
def test_get_dataset_adapter_valid(source_type, expected_adapter):
    adapter = get_dataset_adapter(source_type)
    assert adapter.__class__.__module__ + "." + adapter.__class__.__name__ == expected_adapter


def test_get_dataset_adapter_invalid():
    with pytest.raises(ValueError, match="Unknown source type: INVALID_TYPE"):
        get_dataset_adapter("INVALID_TYPE")


@pytest.fixture
def test_db(monkeypatch):
    """Create an in-memory SQLite database for testing"""

    # Keep validate_path from resolving to absolute paths
    monkeypatch.setattr(base_module, "validate_path", lambda p: p, raising=True)
    adapter = CMIP6DatasetAdapter()
    adapter.dataset_specific_metadata = (
        "activity_id",
        "experiment_id",
        "institution_id",
        "frequency",
        "grid_label",
        "source_id",
        "table_id",
        "variable_id",
        "variant_label",
        "member_id",
        "version",
        "instance_id",
    )
    # Bypass validation
    adapter.validate_data_catalog = lambda df, **kwargs: df

    config = Config.default()
    db = Database("sqlite:///:memory:")
    db.migrate(config)
    yield adapter, config, db
    db.close()


def _mk_df(instance_id="CESM2.tas.gn", rows=None):
    rows = rows or []
    base = {
        "instance_id": instance_id,
        "source_id": "CESM2",
        "variable_id": "tas",
        "grid_label": "gn",
    }
    missing = set(CMIP6DatasetAdapter.dataset_specific_metadata) - set(base.keys())
    for k in missing:
        base[k] = f"default_{k}"

    return pd.DataFrame([{**base, **r} for r in rows])


def test_register_dataset_creates_and_adds_files(monkeypatch, test_db):
    adapter, config, db = test_db

    df = _mk_df(
        rows=[
            {
                "path": "f1.nc",
                "start_time": pd.Timestamp("2001-01-01"),
                "end_time": pd.Timestamp("2001-12-31"),
            },
            {
                "path": "f2.nc",
                "start_time": pd.Timestamp("2001-01-01"),
                "end_time": pd.Timestamp("2001-12-31"),
            },
        ]
    )

    with db.session.begin():
        result = adapter.register_dataset(config=config, db=db, data_catalog_dataset=df)

    assert result.dataset_state == ModelState.CREATED
    assert set(result.files_added) == {"f1.nc", "f2.nc"}
    assert result.files_updated == []
    assert result.files_removed == []
    assert result.files_unchanged == []
    assert result.total_changes == 2

    # Verify the CMIP6 dataset was actually created in the database
    dataset = db.session.query(CMIP6Dataset).filter_by(slug="CESM2.tas.gn").first()
    assert dataset is not None
    assert dataset.dataset_type == SourceDatasetType.CMIP6
    assert dataset.source_id == "CESM2"
    assert dataset.variable_id == "tas"
    assert dataset.experiment_id == "default_experiment_id"
    assert dataset.institution_id == "default_institution_id"

    # Verify the files were actually created in the database
    files = db.session.query(DatasetFile).filter_by(dataset_id=dataset.id).all()
    assert len(files) == 2
    file_paths = {f.path for f in files}
    assert file_paths == {"f1.nc", "f2.nc"}


def test_register_dataset_updates_and_adds_without_removal(monkeypatch, test_db):
    adapter, config, db = test_db

    # First, create initial dataset with existing files
    initial_df = _mk_df(
        rows=[
            {
                "path": "f1.nc",
                "start_time": pd.Timestamp("2001-01-01"),
                "end_time": pd.Timestamp("2001-12-31"),
            },
            {
                "path": "f2.nc",
                "start_time": pd.Timestamp("2000-01-01"),
                "end_time": pd.Timestamp("2000-12-31"),
            },
        ]
    )

    with db.session.begin():
        adapter.register_dataset(config=config, db=db, data_catalog_dataset=initial_df)

    # Now update with modified data
    updated_df = _mk_df(
        rows=[
            {
                "path": "f1.nc",
                "start_time": pd.Timestamp("2001-01-01"),
                "end_time": pd.Timestamp("2001-12-31"),
            },  # unchanged
            {
                "path": "f2.nc",
                "start_time": pd.Timestamp("2001-01-01"),
                "end_time": pd.Timestamp("2001-12-31"),
            },  # updated
            {
                "path": "f3.nc",
                "start_time": pd.Timestamp("2001-01-01"),
                "end_time": pd.Timestamp("2001-12-31"),
            },  # added
        ]
    )

    with db.session.begin():
        result = adapter.register_dataset(config=config, db=db, data_catalog_dataset=updated_df)

    assert result.dataset_state == ModelState.UPDATED
    assert set(result.files_added) == {"f3.nc"}
    assert set(result.files_updated) == {"f2.nc"}
    assert set(result.files_unchanged) == {"f1.nc"}
    assert result.files_removed == []
    assert result.total_changes == 2

    # Verify the database state
    dataset = db.session.query(CMIP6Dataset).filter_by(slug="CESM2.tas.gn").first()
    files = db.session.query(DatasetFile).filter_by(dataset_id=dataset.id).all()
    assert len(files) == 3

    # Check that f2.nc was actually updated
    f2_file = next(f for f in files if f.path == "f2.nc")
    assert f2_file.start_time == pd.Timestamp("2001-01-01")
    assert f2_file.end_time == pd.Timestamp("2001-12-31")


def test_register_dataset_raises_on_removal(monkeypatch, test_db):
    adapter, config, db = test_db

    # First, create initial dataset with files
    initial_df = _mk_df(
        rows=[
            {
                "path": "keep.nc",
                "start_time": pd.Timestamp("2001-01-01"),
                "end_time": pd.Timestamp("2001-12-31"),
            },
            {
                "path": "remove.nc",
                "start_time": pd.Timestamp("2001-01-01"),
                "end_time": pd.Timestamp("2001-12-31"),
            },
        ]
    )

    with db.session.begin():
        adapter.register_dataset(config=config, db=db, data_catalog_dataset=initial_df)

    # New catalog omits "remove.nc" -> triggers removal path
    updated_df = _mk_df(
        rows=[
            {
                "path": "keep.nc",
                "start_time": pd.Timestamp("2001-01-01"),
                "end_time": pd.Timestamp("2001-12-31"),
            },
        ]
    )

    with pytest.raises(NotImplementedError, match="Removing files is not yet supported"):
        with db.session.begin():
            adapter.register_dataset(config=config, db=db, data_catalog_dataset=updated_df)


def test_register_dataset_multiple_datasets_error(monkeypatch, test_db):
    adapter, config, db = test_db

    df = pd.concat(
        [
            _mk_df(
                instance_id="CESM2.tas.gn",
                rows=[
                    {
                        "path": "a.nc",
                        "start_time": pd.Timestamp("2001-01-01"),
                        "end_time": pd.Timestamp("2001-12-31"),
                    }
                ],
            ),
            _mk_df(
                instance_id="CESM2.pr.gn",
                rows=[
                    {
                        "path": "b.nc",
                        "start_time": pd.Timestamp("2001-01-01"),
                        "end_time": pd.Timestamp("2001-12-31"),
                    }
                ],
            ),
        ],
        ignore_index=True,
    )

    with pytest.raises(RefException, match="Found multiple datasets in the same directory"):
        with db.session.begin():
            adapter.register_dataset(config=config, db=db, data_catalog_dataset=df)


def test_register_dataset_updates_dataset_metadata(monkeypatch, test_db):
    """Test that changes to dataset metadata are properly captured and result in UPDATED state"""
    adapter, config, db = test_db

    # First, create initial dataset with original metadata
    df = _mk_df(
        instance_id="CESM2.tas.gn",
        rows=[
            {
                "path": "tas_file.nc",
                "start_time": pd.Timestamp("2001-01-01"),
                "end_time": pd.Timestamp("2001-12-31"),
            },
        ],
    )

    with db.session.begin():
        initial_result = adapter.register_dataset(config=config, db=db, data_catalog_dataset=df)

    assert initial_result.dataset_state == ModelState.CREATED

    # Update the dataset metadata
    df.loc[0, "grid_label"] = "gr2"

    with db.session.begin():
        update_result = adapter.register_dataset(config=config, db=db, data_catalog_dataset=df)

    # Should be UPDATED because dataset metadata changed
    assert update_result.dataset_state == ModelState.UPDATED
    assert update_result.files_added == []
    assert update_result.files_updated == []
    assert update_result.files_removed == []
    assert update_result.files_unchanged == ["tas_file.nc"]
    assert update_result.total_changes == 0  # No file changes, only metadata changes

    # Verify the dataset metadata was actually updated in the database
    dataset = db.session.query(CMIP6Dataset).filter_by(slug="CESM2.tas.gn").first()
    assert dataset is not None
    assert dataset.grid_label == "gr2"


class TestFilterLatestVersions:
    """Tests for DatasetAdapter.filter_latest_versions."""

    def test_keeps_latest_version_per_group(self):
        adapter = CMIP6DatasetAdapter()
        catalog = pd.DataFrame(
            {
                "activity_id": ["CMIP"] * 4,
                "institution_id": ["NCAR"] * 4,
                "source_id": ["CESM2"] * 4,
                "experiment_id": ["historical"] * 4,
                "member_id": ["r1i1p1f1"] * 4,
                "table_id": ["Amon"] * 4,
                "variable_id": ["tas", "tas", "pr", "pr"],
                "grid_label": ["gn"] * 4,
                "version": ["v20200101", "v20210601", "v20200101", "v20210601"],
                "instance_id": [
                    "CMIP6.CMIP.NCAR.CESM2.historical.r1i1p1f1.Amon.tas.gn.v20200101",
                    "CMIP6.CMIP.NCAR.CESM2.historical.r1i1p1f1.Amon.tas.gn.v20210601",
                    "CMIP6.CMIP.NCAR.CESM2.historical.r1i1p1f1.Amon.pr.gn.v20200101",
                    "CMIP6.CMIP.NCAR.CESM2.historical.r1i1p1f1.Amon.pr.gn.v20210601",
                ],
            }
        )

        result = adapter.filter_latest_versions(catalog)

        assert len(result) == 2
        assert set(result["version"]) == {"v20210601"}
        assert set(result["variable_id"]) == {"tas", "pr"}

    def test_keeps_all_when_no_duplicates(self):
        adapter = CMIP6DatasetAdapter()
        catalog = pd.DataFrame(
            {
                "activity_id": ["CMIP", "CMIP"],
                "institution_id": ["NCAR", "NCAR"],
                "source_id": ["CESM2", "CESM2"],
                "experiment_id": ["historical", "historical"],
                "member_id": ["r1i1p1f1", "r1i1p1f1"],
                "table_id": ["Amon", "Amon"],
                "variable_id": ["tas", "pr"],
                "grid_label": ["gn", "gn"],
                "version": ["v20210601", "v20210601"],
            }
        )

        result = adapter.filter_latest_versions(catalog)
        assert len(result) == 2

    def test_handles_empty_catalog(self):
        adapter = CMIP6DatasetAdapter()
        catalog = pd.DataFrame()

        result = adapter.filter_latest_versions(catalog)
        assert result.empty

    def test_handles_multiple_files_per_dataset(self):
        """Multiple files (rows) for the same dataset version should all be kept."""
        adapter = CMIP6DatasetAdapter()
        catalog = pd.DataFrame(
            {
                "activity_id": ["CMIP"] * 4,
                "institution_id": ["NCAR"] * 4,
                "source_id": ["CESM2"] * 4,
                "experiment_id": ["historical"] * 4,
                "member_id": ["r1i1p1f1"] * 4,
                "table_id": ["Amon"] * 4,
                "variable_id": ["tas"] * 4,
                "grid_label": ["gn"] * 4,
                "version": ["v20200101", "v20200101", "v20210601", "v20210601"],
                "path": ["old_1.nc", "old_2.nc", "new_1.nc", "new_2.nc"],
            }
        )

        result = adapter.filter_latest_versions(catalog)

        assert len(result) == 2
        assert set(result["path"]) == {"new_1.nc", "new_2.nc"}
        assert set(result["version"]) == {"v20210601"}

    def test_no_dataset_id_metadata_returns_unfiltered(self):
        """Adapters with empty dataset_id_metadata should return the catalog as-is."""
        adapter = MockDatasetAdapter()
        adapter.dataset_id_metadata = ()
        catalog = pd.DataFrame({"version": ["v1", "v2"], "variable_id": ["tas", "tas"]})

        result = adapter.filter_latest_versions(catalog)
        assert len(result) == 2


class TestIngestionStats:
    """Tests for IngestionStats dataclass."""

    def test_default_values(self):
        stats = IngestionStats()
        assert stats.datasets_created == 0
        assert stats.datasets_updated == 0
        assert stats.datasets_unchanged == 0
        assert stats.files_added == 0
        assert stats.files_updated == 0
        assert stats.files_removed == 0
        assert stats.files_unchanged == 0

    def test_custom_values(self):
        stats = IngestionStats(
            datasets_created=1,
            datasets_updated=2,
            datasets_unchanged=3,
            files_added=4,
            files_updated=5,
            files_removed=6,
            files_unchanged=7,
        )
        assert stats.datasets_created == 1
        assert stats.datasets_updated == 2
        assert stats.datasets_unchanged == 3
        assert stats.files_added == 4
        assert stats.files_updated == 5
        assert stats.files_removed == 6
        assert stats.files_unchanged == 7

    def test_log_summary(self, caplog):
        stats = IngestionStats(
            datasets_created=1,
            datasets_updated=2,
            datasets_unchanged=3,
            files_added=4,
            files_updated=5,
            files_removed=6,
            files_unchanged=7,
        )
        stats.log_summary()
        assert "Datasets: 1/2/3 (created/updated/unchanged)" in caplog.text
        assert "Files: 4/5/6/7 (created/updated/removed/unchanged)" in caplog.text

    def test_log_summary_with_prefix(self, caplog):
        stats = IngestionStats(datasets_created=1)
        stats.log_summary("Test prefix:")
        assert "Test prefix: Datasets:" in caplog.text


class TestIngestDatasets:
    """Tests for the ingest_datasets shared function."""

    def test_ingest_datasets_directory_not_exists(self, test_db):
        adapter, config, db = test_db
        with pytest.raises(ValueError, match="does not exist"):
            ingest_datasets(adapter, Path("/nonexistent/path"), config, db)

    def test_ingest_datasets_no_nc_files(self, test_db, tmp_path):
        adapter, config, db = test_db
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        with pytest.raises(ValueError, match=r"No \.nc files found"):
            ingest_datasets(adapter, empty_dir, config, db)

    def test_ingest_datasets_requires_directory_or_catalog(self, test_db):
        adapter, config, db = test_db
        with pytest.raises(ValueError, match="Either directory or data_catalog must be provided"):
            ingest_datasets(adapter, None, config, db)

    def test_ingest_datasets_with_pre_validated_catalog(self, monkeypatch, test_db):
        """Test that ingest_datasets works with a pre-validated data catalog."""
        adapter, config, db = test_db

        df = _mk_df(
            rows=[
                {
                    "path": "f1.nc",
                    "start_time": pd.Timestamp("2001-01-01"),
                    "end_time": pd.Timestamp("2001-12-31"),
                },
                {
                    "path": "f2.nc",
                    "start_time": pd.Timestamp("2002-01-01"),
                    "end_time": pd.Timestamp("2002-12-31"),
                },
            ]
        )

        # Call with pre-validated catalog (directory=None)
        stats = ingest_datasets(adapter, None, config, db, data_catalog=df)

        assert stats.datasets_created == 1
        assert stats.datasets_updated == 0
        assert stats.datasets_unchanged == 0
        assert stats.files_added == 2
        assert stats.files_updated == 0
        assert stats.files_removed == 0
        assert stats.files_unchanged == 0

    def test_ingest_datasets_idempotent(self, monkeypatch, test_db):
        """Test that calling ingest_datasets twice is idempotent."""
        adapter, config, db = test_db

        df = _mk_df(
            rows=[
                {
                    "path": "f1.nc",
                    "start_time": pd.Timestamp("2001-01-01"),
                    "end_time": pd.Timestamp("2001-12-31"),
                },
            ]
        )

        # First call creates the dataset
        stats1 = ingest_datasets(adapter, None, config, db, data_catalog=df)
        assert stats1.datasets_created == 1
        assert stats1.files_added == 1

        # Second call should find it unchanged
        stats2 = ingest_datasets(adapter, None, config, db, data_catalog=df)
        assert stats2.datasets_created == 0
        assert stats2.datasets_unchanged == 1
        assert stats2.files_added == 0
        assert stats2.files_unchanged == 1

    def test_ingest_datasets_with_directory(self, monkeypatch, test_db, tmp_path):
        """Test ingest_datasets with directory finds and validates datasets."""
        adapter, config, db = test_db

        # Create a directory with .nc files
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "test1.nc").touch()
        (data_dir / "test2.nc").touch()

        # Create mock data catalog that find_local_datasets will return
        mock_df = _mk_df(
            rows=[
                {
                    "path": str(data_dir / "test1.nc"),
                    "start_time": pd.Timestamp("2001-01-01"),
                    "end_time": pd.Timestamp("2001-12-31"),
                },
                {
                    "path": str(data_dir / "test2.nc"),
                    "start_time": pd.Timestamp("2002-01-01"),
                    "end_time": pd.Timestamp("2002-12-31"),
                },
            ]
        )

        # Patch find_local_datasets to return our mock catalog
        monkeypatch.setattr(adapter, "find_local_datasets", lambda d: mock_df)

        stats = ingest_datasets(adapter, data_dir, config, db)

        assert stats.datasets_created == 1
        assert stats.files_added == 2

    def test_ingest_datasets_empty_after_validation(self, monkeypatch, test_db, tmp_path):
        """Test ingest_datasets raises ValueError when catalog is empty after validation."""
        adapter, config, db = test_db

        # Create a directory with .nc files
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        (data_dir / "test.nc").touch()

        # Mock find_local_datasets to return a non-empty catalog
        mock_df = _mk_df(
            rows=[
                {
                    "path": str(data_dir / "test.nc"),
                    "start_time": pd.Timestamp("2001-01-01"),
                    "end_time": pd.Timestamp("2001-12-31"),
                },
            ]
        )
        monkeypatch.setattr(adapter, "find_local_datasets", lambda d: mock_df)

        # Mock validate_data_catalog to return empty DataFrame (all invalid)
        monkeypatch.setattr(adapter, "validate_data_catalog", lambda df, **kwargs: pd.DataFrame())

        with pytest.raises(ValueError, match="No valid datasets found"):
            ingest_datasets(adapter, data_dir, config, db)

    def test_ingest_datasets_updated_state(self, monkeypatch, test_db):
        """Test that ingest_datasets correctly tracks updated datasets."""
        adapter, config, db = test_db

        # First create a dataset
        df1 = _mk_df(
            rows=[
                {
                    "path": "f1.nc",
                    "start_time": pd.Timestamp("2001-01-01"),
                    "end_time": pd.Timestamp("2001-12-31"),
                },
            ]
        )
        stats1 = ingest_datasets(adapter, None, config, db, data_catalog=df1)
        assert stats1.datasets_created == 1

        # Now update with additional file (triggers UPDATED state)
        df2 = _mk_df(
            rows=[
                {
                    "path": "f1.nc",
                    "start_time": pd.Timestamp("2001-01-01"),
                    "end_time": pd.Timestamp("2001-12-31"),
                },
                {
                    "path": "f2.nc",
                    "start_time": pd.Timestamp("2002-01-01"),
                    "end_time": pd.Timestamp("2002-12-31"),
                },
            ]
        )
        stats2 = ingest_datasets(adapter, None, config, db, data_catalog=df2)

        assert stats2.datasets_created == 0
        assert stats2.datasets_updated == 1
        assert stats2.datasets_unchanged == 0
        assert stats2.files_added == 1
        assert stats2.files_unchanged == 1
