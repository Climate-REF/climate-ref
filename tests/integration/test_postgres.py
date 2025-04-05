"""
Runs an integration test for the connecting to a Postgres DB

This runs the migrations and ingests some datasets as a test.

This test requires a running PostgreSQL server, which is started as a Docker container.
"""

import time

import psycopg2
from loguru import logger
from pytest_docker_tools import container, fetch, wrappers

from cmip_ref.database import Database
from cmip_ref.datasets.cmip6 import CMIP6DatasetAdapter

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


def test_connect_and_migrations(config, postgres_container, cmip6_data_catalog):
    config.db.database_url = postgres_container.connection_url()
    config.save()

    database = Database.from_config(config)
    assert database.url.startswith("postgresql")
    assert database._engine.dialect.name == "postgresql"

    adapter = CMIP6DatasetAdapter()

    with database.session.begin():
        for instance_id, data_catalog_dataset in cmip6_data_catalog.groupby(adapter.slug_column):
            adapter.register_dataset(config, database, data_catalog_dataset)
