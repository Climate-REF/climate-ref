import shutil
from pathlib import Path
from urllib import parse as urlparse

import pandas as pd
import pytest
from cmip_ref_metrics_example import provider as example_provider
from pytest_regressions.data_regression import RegressionYamlDumper
from yaml.representer import SafeRepresenter

from cmip_ref.config import Config
from cmip_ref.database import Database
from cmip_ref.datasets.cmip6 import CMIP6DatasetAdapter
from cmip_ref.provider_registry import _register_provider

# Ignore the alembic folder
collect_ignore = ["src/cmip_ref/migrations"]

# Add a representer for pandas Timestamps/NaT in the regression tests
RegressionYamlDumper.add_representer(
    pd.Timestamp, lambda dumper, data: SafeRepresenter.represent_datetime(dumper, data.to_pydatetime())
)
RegressionYamlDumper.add_representer(
    type(pd.NaT), lambda dumper, data: SafeRepresenter.represent_none(dumper, data)
)


def _clone_db(target_db_url: str, template_db_path: Path) -> None:
    split_url = urlparse.urlsplit(target_db_url)
    target_db_path = Path(split_url.path[1:])
    target_db_path.parent.mkdir(parents=True)

    shutil.copy(template_db_path, target_db_path)


@pytest.fixture
def db(config) -> Database:
    return Database.from_config(config, run_migrations=True)


@pytest.fixture(scope="session")
def db_seeded_template(tmp_path_session, cmip6_data_catalog) -> Path:
    template_db_path = tmp_path_session / "cmip_ref_template_seeded.db"

    config = Config.default()  # This is just dummy config
    database = Database(f"sqlite:///{template_db_path}", run_migrations=True)

    adapter = CMIP6DatasetAdapter()

    # Seed with all the datasets in the ESGF data directory
    # This includes datasets which span multiple file and until 2300
    with database.session.begin():
        for instance_id, data_catalog_dataset in cmip6_data_catalog.groupby(adapter.slug_column):
            adapter.register_dataset(config, database, data_catalog_dataset)

        _register_provider(database, example_provider)

    return template_db_path


@pytest.fixture
def db_seeded(db_seeded_template, config) -> Database:
    # Copy the template database to the location in the config
    _clone_db(config.db.database_url, db_seeded_template)

    return Database.from_config(config, run_migrations=True)
