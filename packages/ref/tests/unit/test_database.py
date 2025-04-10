import pytest
import sqlalchemy
from sqlalchemy import inspect

from cmip_ref.database import Database, validate_database_url
from cmip_ref.models import MetricValue
from cmip_ref.models.dataset import CMIP6Dataset, Dataset, Obs4MIPsDataset
from cmip_ref_core.datasets import SourceDatasetType
from cmip_ref_core.pycmec.controlled_vocabulary import CV


@pytest.mark.parametrize(
    "database_url",
    [
        "sqlite:///:memory:",
        "sqlite:///{tmp_path}/cmip_ref.db",
        "postgresql://localhost:5432/cmip_ref",
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
            units="",
            variable_id="",
            variant_label="",
            vertical_levels=2,
            source_version_number="v12",
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
    monkeypatch.setenv("REF_DATABASE_URL", "postgresql:///localhost:12323/cmip_ref")
    config = config.refresh()

    with pytest.raises(sqlalchemy.exc.OperationalError):
        Database.from_config(config, run_migrations=True)


def test_database_cvs(config, mocker):
    cv = CV.load_from_file(config.paths.dimensions_cv)

    mock_register_cv = mocker.patch.object(MetricValue, "register_cv_dimensions")
    mock_cv = mocker.patch.object(CV, "load_from_file", return_value=cv)

    db = Database.from_config(config, run_migrations=True)

    # CV is loaded once during a migration and once when registering
    assert mock_cv.call_count == 2
    mock_cv.assert_called_with(config.paths.dimensions_cv)
    mock_register_cv.assert_called_once_with(mock_cv.return_value)

    # Verify that the dimensions have automatically been created
    inspector = inspect(db._engine)
    existing_columns = [c["name"] for c in inspector.get_columns("metric_value")]
    for dimension in cv.dimensions:
        assert dimension.name in existing_columns
