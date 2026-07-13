"""
Runs an integration test for the connecting to a Postgres DB

This runs the migrations and ingests some datasets as a test.

This test requires a running PostgreSQL server, which is started as a Docker container.
"""

import time

import alembic.command
import psycopg2
import pytest
from loguru import logger
from pytest_docker_tools import container, fetch, wrappers

from climate_ref.database import Database
from climate_ref.datasets.cmip6 import CMIP6DatasetAdapter
from climate_ref.models.dataset import Obs4REFDataset
from climate_ref_core.source_types import SourceDatasetType

POSTGRES_USER = "postgres"
POSTGRES_PASSWORD = "example"  # noqa: S105


class PostgresContainer(wrappers.Container):
    PORT_ID = "5432/tcp"

    def ready(self):
        if super().ready() and len(self.ports[self.PORT_ID]) > 0:
            port = self.ports[self.PORT_ID][0]

            try:
                conn = psycopg2.connect(
                    host="localhost",
                    port=port,
                    user=POSTGRES_USER,
                    password=POSTGRES_PASSWORD,
                    dbname="postgres",
                )
                logger.info("Postgres is ready!")
                conn.close()
                return True
            except psycopg2.OperationalError as e:
                logger.info(str(e).strip())
                logger.info("Postgres isn't ready")
                time.sleep(3)

        return False

    def connection_url(self) -> str:
        port = self.ports[self.PORT_ID][0]
        return f"postgresql://{POSTGRES_USER}:{POSTGRES_PASSWORD}@localhost:{port}/postgres"


postgres_image = fetch(repository="postgres:17")

postgres_container = container(
    image="{postgres_image.id}",
    ports={
        PostgresContainer.PORT_ID: None,
    },
    wrapper_class=PostgresContainer,
    environment={
        "POSTGRES_USER": POSTGRES_USER,
        "POSTGRES_PASSWORD": POSTGRES_PASSWORD,
        "POSTGRES_DB": "postgres",
    },
)


@pytest.fixture
def config(config, postgres_container):
    config.db.database_url = postgres_container.connection_url()
    config.save()

    return config


@pytest.mark.docker
def test_connect_and_migrations(config, cmip6_data_catalog):
    database = Database.from_config(config)
    assert database.url.startswith("postgresql")
    assert database._engine.dialect.name == "postgresql"

    adapter = CMIP6DatasetAdapter()

    with database.session.begin():
        for instance_id, data_catalog_dataset in cmip6_data_catalog.groupby(adapter.slug_column):
            adapter.register_dataset(database, data_catalog_dataset)


@pytest.mark.docker
def test_obs4ref_enum_label(config):
    """Prove the ``obs4REF`` label actually landed on the Postgres ``sourcedatasettype`` enum.

    ``test_connect_and_migrations`` only proves the migration *chain* runs. The enum-label
    failure (``invalid input value for enum sourcedatasettype``) happens at INSERT, not at
    migration time, so a migrations-only run passes even if the ``ALTER TYPE ... ADD VALUE``
    was forgotten. This inserts an ``obs4REF`` row and reads it back to prove the label landed.
    """
    database = Database.from_config(config)
    assert database._engine.dialect.name == "postgresql"

    dataset = Obs4REFDataset(
        slug="obs4REF.TESTORG.TEST-SRC.mon.ts.100km.gn.v1",
        finalised=True,
        activity_id="obs4REF",
        frequency="mon",
        grid="native",
        grid_label="gn",
        institution_id="TESTORG",
        long_name="Test",
        nominal_resolution="100 km",
        realm="atmos",
        product="observations",
        source_id="TEST-SRC",
        source_type="satellite",
        units="K",
        variable_id="ts",
        variant_label="v1",
        version="v1",
        vertical_levels=1,
        source_version_number="1",
        instance_id="obs4REF.TESTORG.TEST-SRC.mon.ts.100km.gn.v1",
    )
    with database.session.begin():
        database.session.add(dataset)
    database.session.expire_all()

    with database.session.begin():
        fetched = database.session.query(Obs4REFDataset).filter_by(slug=dataset.slug).one()
        assert fetched.dataset_type is SourceDatasetType.obs4REF
        assert fetched.instance_id == "obs4REF.TESTORG.TEST-SRC.mon.ts.100km.gn.v1"


@pytest.mark.docker
def test_enum_add_value_survives_downgrade_upgrade_cycle(config):
    """Finding 1 regression guard: upgrade -> downgrade -> upgrade must not fail on Postgres.

    The migration that adds the ``obs4REF`` and ``ESMValToolReference`` labels only drops its
    tables on downgrade. It cannot remove the enum labels that ``ALTER TYPE sourcedatasettype
    ADD VALUE`` created, since Postgres has no ``DROP VALUE``. Without ``IF NOT EXISTS`` the
    second upgrade in this cycle fails with ``enum label "obs4REF" already exists``. This
    exercises exactly that cycle: it would fail against a plain ``ADD VALUE`` and only passes
    because the migration uses ``ADD VALUE IF NOT EXISTS``.
    """
    database = Database.from_config(config)
    alembic_cfg = database.alembic_config(config)

    alembic.command.upgrade(alembic_cfg, "head")
    # Downgrade past the enum-adding migration. The labels stay on the native Postgres enum type
    # even though the tables that used them are dropped.
    alembic.command.downgrade(alembic_cfg, "f6a7b8c9d0e1")
    # Must not raise "enum label ... already exists".
    alembic.command.upgrade(alembic_cfg, "head")

    # Confirm both labels are still genuinely usable after the second upgrade, not merely present.
    obs4ref_dataset = Obs4REFDataset(
        slug="obs4REF.TESTORG.CYCLE-SRC.mon.ts.100km.gn.v1",
        finalised=True,
        activity_id="obs4REF",
        frequency="mon",
        grid="native",
        grid_label="gn",
        institution_id="TESTORG",
        long_name="Test",
        nominal_resolution="100 km",
        realm="atmos",
        product="observations",
        source_id="CYCLE-SRC",
        source_type="satellite",
        units="K",
        variable_id="ts",
        variant_label="v1",
        version="v1",
        vertical_levels=1,
        source_version_number="1",
        instance_id="obs4REF.TESTORG.CYCLE-SRC.mon.ts.100km.gn.v1",
    )
    with database.session.begin():
        database.session.add(obs4ref_dataset)
    database.session.expire_all()

    with database.session.begin():
        fetched_obs4ref = database.session.query(Obs4REFDataset).filter_by(slug=obs4ref_dataset.slug).one()
        assert fetched_obs4ref.dataset_type is SourceDatasetType.obs4REF


@pytest.mark.docker
def test_check_up_to_date(config):
    database = Database.from_config(config)

    # Verify that the migrations match the codebase for postgres
    alembic.command.check(database.alembic_config(config))

    # Verify that we can go downgrade to an empty db
    alembic.command.downgrade(database.alembic_config(config), "base")
