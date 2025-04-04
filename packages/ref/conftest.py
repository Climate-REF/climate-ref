import importlib.resources
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
from cmip_ref.datasets.obs4mips import Obs4MIPsDatasetAdapter
from cmip_ref.models.metric_value import MetricValue
from cmip_ref.provider_registry import _register_provider
from cmip_ref_core.pycmec.controlled_vocabulary import CV

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
def ar7_ft_cv() -> CV:
    cv_file = str(importlib.resources.files("cmip_ref_core.pycmec") / "cv_cmip_ar7ft.yaml")

    return CV.load_from_file(cv_file)


@pytest.fixture(scope="session")
def prepare_db(ar7_ft_cv):
    MetricValue.register_cv_dimensions(ar7_ft_cv)


@pytest.fixture(scope="session")
def db_seeded_template(tmp_path_session, cmip6_data_catalog, obs4mips_data_catalog, prepare_db) -> Path:
    template_db_path = tmp_path_session / "cmip_ref_template_seeded.db"

    config = Config.default()  # This is just dummy config
    database = Database(f"sqlite:///{template_db_path}", run_migrations=True)

    # Seed the CMIP6 sample datasets
    adapter = CMIP6DatasetAdapter()
    with database.session.begin():
        for instance_id, data_catalog_dataset in cmip6_data_catalog.groupby(adapter.slug_column):
            adapter.register_dataset(config, database, data_catalog_dataset)

    # Seed the obs4MIPs sample datasets
    adapter_obs = Obs4MIPsDatasetAdapter()
    with database.session.begin():
        for instance_id, data_catalog_dataset in obs4mips_data_catalog.groupby(adapter_obs.slug_column):
            adapter_obs.register_dataset(config, database, data_catalog_dataset)

    with database.session.begin():
        _register_provider(database, example_provider)

    return template_db_path


@pytest.fixture
def db_seeded(db_seeded_template, config) -> Database:
    # Copy the template database to the location in the config
    _clone_db(config.db.database_url, db_seeded_template)

    return Database.from_config(config, run_migrations=True)
