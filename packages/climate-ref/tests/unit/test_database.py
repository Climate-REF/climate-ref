import re
from datetime import datetime, timedelta
from pathlib import Path

import pytest
import sqlalchemy
from sqlalchemy import inspect

from climate_ref.database import Database, _create_backup, validate_database_url
from climate_ref.models import MetricValue
from climate_ref.models.dataset import CMIP6Dataset, Dataset, Obs4MIPsDataset
from climate_ref_core.datasets import SourceDatasetType
from climate_ref_core.pycmec.controlled_vocabulary import CV


@pytest.mark.parametrize(
    "database_url",
    [
        "sqlite:///:memory:",
        "sqlite:///{tmp_path}/climate_ref.db",
        "postgresql://localhost:5432/climate_ref",
    ],
)
def test_validate_database_url(config, database_url, tmp_path):
    validate_database_url(database_url.format(tmp_path=str(tmp_path)))


@pytest.mark.parametrize("database_url", ["mysql:///:memory:", "no_scheme/test"])
def test_invalid_urls(config, database_url, tmp_path):
    with pytest.raises(ValueError):
        validate_database_url(database_url.format(tmp_path=str(tmp_path)))


def test_database(db):
    assert db._engine
    assert db.session.is_active


def test_database_migrate_with_old_revision(db, mocker, config):
    # New migrations are fine
    db.migrate(config)

    # Old migrations should raise a useful error message
    mocker.patch("climate_ref.database._get_database_revision", return_value="ea2aa1134cb3")
    with pytest.raises(ValueError, match="Please delete your database and start again"):
        db.migrate(config)


def test_dataset_polymorphic(db):
    db.session.add(
        CMIP6Dataset(
            activity_id="",
            branch_method="",
            branch_time_in_child=12,
            branch_time_in_parent=21,
            experiment="",
            experiment_id="",
            frequency="",
            grid="",
            grid_label="",
            institution_id="",
            long_name="",
            member_id="",
            nominal_resolution="",
            parent_activity_id="",
            parent_experiment_id="",
            parent_source_id="",
            parent_time_units="",
            parent_variant_label="",
            realm="",
            product="",
            source_id="",
            standard_name="",
            source_type="",
            sub_experiment="",
            sub_experiment_id="",
            table_id="",
            units="",
            variable_id="",
            variant_label="",
            vertical_levels=2,
            version="v12",
            instance_id="test",
            slug="test",
        )
    )
    assert db.session.query(CMIP6Dataset).count() == 1
    assert db.session.query(Dataset).first().slug == "test"
    assert db.session.query(Dataset).first().dataset_type == SourceDatasetType.CMIP6

    db.session.add(
        Obs4MIPsDataset(
            activity_id="obs4MIPs",
            frequency="",
            grid="",
            grid_label="",
            institution_id="",
            long_name="",
            nominal_resolution="",
            realm="",
            product="",
            source_id="",
            source_type="",
            source_version_number="",
            units="",
            variable_id="",
            variant_label="",
            version="v12",
            vertical_levels=2,
            instance_id="test_obs",
            slug="test_obs",
        )
    )
    assert db.session.query(Obs4MIPsDataset).count() == 1
    assert db.session.query(Obs4MIPsDataset).first().slug == "test_obs"
    assert db.session.query(Obs4MIPsDataset).first().dataset_type == SourceDatasetType.obs4MIPs


def test_transaction_cleanup(db):
    with pytest.raises(sqlalchemy.exc.IntegrityError):
        with db.session.begin():
            db.session.add(CMIP6Dataset(slug="test"))
            db.session.add(CMIP6Dataset(slug="test"))
            db.session.add(Obs4MIPsDataset(slug="test_obs"))
            db.session.add(Obs4MIPsDataset(slug="test_obs"))
    assert db.session.query(CMIP6Dataset).count() == 0
    assert db.session.query(Obs4MIPsDataset).count() == 0


def test_database_invalid_url(config, monkeypatch):
    monkeypatch.setenv("REF_DATABASE_URL", "postgresql:///localhost:12323/climate_ref")
    config = config.refresh()

    with pytest.raises(sqlalchemy.exc.OperationalError):
        Database.from_config(config, run_migrations=True)


def test_database_cvs(config, mocker):
    cv = CV.load_from_file(config.paths.dimensions_cv)

    mock_register_cv = mocker.patch.object(MetricValue, "register_cv_dimensions")
    mock_cv = mocker.patch.object(CV, "load_from_file", return_value=cv)

    with Database.from_config(config, run_migrations=True) as db:
        # CV is loaded once during a migration and once with each call to _add_dimension_columns
        assert mock_cv.call_count == 3
        mock_cv.assert_called_with(config.paths.dimensions_cv)
        mock_register_cv.assert_called_once_with(mock_cv.return_value)

        # Verify that the dimensions have automatically been created
        inspector = inspect(db._engine)
        existing_columns = [c["name"] for c in inspector.get_columns("metric_value")]
        for dimension in cv.dimensions:
            assert dimension.name in existing_columns


def test_create_backup(tmp_path):
    # Create a test database file
    db_path = tmp_path / "test.db"
    db_path.write_text("test data")

    # Create a backup
    backup_path = _create_backup(db_path, max_backups=3)

    # Verify backup was created
    assert backup_path.exists()
    assert backup_path.read_text() == "test data"

    # Verify backup is in backups directory
    assert backup_path.parent == db_path.parent / "backups"

    # Verify backup filename format
    timestamp = re.search(r"test_(.*)\.db", backup_path.name).group(1)
    datetime.strptime(timestamp, "%Y%m%d_%H%M%S")


def test_create_backup_nonexistent_file(tmp_path):
    # Try to backup a non-existent file
    db_path = tmp_path / "nonexistent.db"
    backup_path = _create_backup(db_path, max_backups=3)

    # Should return the original path and not create any backups
    assert backup_path == db_path
    assert not (tmp_path / "backups").exists()


def test_create_backup_cleanup_old_backups(tmp_path):
    # Create a test database file
    db_path = tmp_path / "test.db"
    db_path.write_text("test data")

    # Create some old backups
    backup_dir = tmp_path / "backups"
    backup_dir.mkdir()

    # Create 5 old backups with timestamps
    old_backups = []
    for i in range(5):
        timestamp = (datetime.now() - timedelta(hours=i)).strftime("%Y%m%d_%H%M%S")
        backup_path = backup_dir / f"test_{timestamp}.db"
        backup_path.write_text(f"old data {i}")
        old_backups.append(backup_path)

    # Create a new backup with max_backups=3
    new_backup = _create_backup(db_path, max_backups=3)

    # Verify only the 3 most recent backups exist
    remaining_backups = sorted(backup_dir.glob("test_*.db"), reverse=True)
    assert len(remaining_backups) == 3
    assert new_backup in remaining_backups

    # Verify the oldest backups were removed
    for old_backup in old_backups[3:]:
        assert not old_backup.exists()


def test_migrate_creates_backup(tmp_path, config):
    # Create a test database
    db_path = tmp_path / "climate_ref.db"

    # Configure the database URL to point to our test database
    config.db.database_url = f"sqlite:///{db_path}"
    config.db.max_backups = 2

    # Create database instance and run migrations
    db = Database.from_config(config, run_migrations=True)
    db.close()

    # Verify backup was created
    backup_dir = db_path.parent / "backups"
    assert backup_dir.exists()
    backups = list(backup_dir.glob("climate_ref_*.db"))
    assert len(backups) == 1


def test_migrate_no_backup_for_memory_db(config):
    # Configure in-memory database
    config.db.database_url = "sqlite:///:memory:"

    # Create database instance and run migrations
    db = Database.from_config(config, run_migrations=True)
    db.close()

    # Verify no backup directory was created
    assert not (Path("backups")).exists()


def test_migrate_no_backup_for_postgres(config):
    # Configure PostgreSQL database
    config.db.database_url = "postgresql://localhost:5432/climate_ref"

    # Create database instance and run migrations
    # This will fail to connect, but that's okay - we just want to verify no backup is attempted
    with pytest.raises(sqlalchemy.exc.OperationalError):
        Database.from_config(config, run_migrations=True)

    # Verify no backup directory was created
    assert not (Path("backups")).exists()


def test_migrate_skip_backup(tmp_path, config):
    """Test that skip_backup=True prevents backup creation."""
    # Create a test database
    db_path = tmp_path / "climate_ref.db"

    # Configure the database URL to point to our test database
    config.db.database_url = f"sqlite:///{db_path}"
    config.db.max_backups = 2

    # Create database instance with skip_backup=True
    db = Database.from_config(config, run_migrations=True, skip_backup=True)
    db.close()

    # Verify no backup was created
    backup_dir = db_path.parent / "backups"
    assert not backup_dir.exists() or len(list(backup_dir.glob("climate_ref_*.db"))) == 0


def test_from_config_skip_backup_parameter(tmp_path, config, mocker):
    """Test that from_config passes skip_backup to migrate."""
    db_path = tmp_path / "climate_ref.db"
    config.db.database_url = f"sqlite:///{db_path}"

    # Mock _create_backup to verify it's not called
    mock_backup = mocker.patch("climate_ref.database._create_backup")

    db = Database.from_config(config, run_migrations=True, skip_backup=True)
    db.close()

    # Backup should not have been called
    mock_backup.assert_not_called()


def test_from_config_creates_backup_by_default(tmp_path, config, mocker):
    """Test that from_config creates backup by default (skip_backup=False)."""
    db_path = tmp_path / "climate_ref.db"
    config.db.database_url = f"sqlite:///{db_path}"

    # Mock _create_backup to verify it is called
    mock_backup = mocker.patch("climate_ref.database._create_backup")

    db = Database.from_config(config, run_migrations=True, skip_backup=False)
    db.close()

    # Backup should have been called
    mock_backup.assert_called_once()
