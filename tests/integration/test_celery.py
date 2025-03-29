"""
Runs an integration test for the Celery executor with a Redis broker.

This test requires a running Redis server, which is started as a Docker container.
"""

import gc
import time
from pathlib import Path

import pytest
import redis
from cmip_ref_celery.app import create_celery_app
from cmip_ref_celery.tasks import register_celery_tasks
from cmip_ref_metrics_example import provider
from pytest_docker_tools import container, fetch, wrappers

from cmip_ref.database import Database
from cmip_ref.datasets.cmip6 import CMIP6DatasetAdapter
from cmip_ref.models import MetricExecutionResult
from cmip_ref.solver import solve_metrics

ROOT_DIR = Path(__file__).parents[2]


class RedisContainer(wrappers.Container):
    def ready(self):
        if super().ready() and len(self.ports["6379/tcp"]) > 0:
            port = self.ports["6379/tcp"][0]
            print(f"Redis using port: {port}")
            # Perform a simple ping to check if the server is ready
            r = redis.Redis(host="localhost", port=port)
            try:
                return r.ping()
            except redis.ConnectionError:
                print("Redis connection error, retrying...")
                time.sleep(2)  # Increase the sleep time
        return False

    def connection_url(self) -> str:
        port = self.ports["6379/tcp"][0]
        return f"redis://localhost:{port}/0"


redis_image = fetch(repository="redis:7")

redis_container = container(
    image="{redis_image.id}",
    ports={
        "6379/tcp": None,
    },
    wrapper_class=RedisContainer,
)


@pytest.fixture
def db_seeded(config, cmip6_data_catalog):
    database = Database.from_config(config)

    adapter = CMIP6DatasetAdapter()

    with database.session.begin():
        for instance_id, data_catalog_dataset in cmip6_data_catalog.groupby(adapter.slug_column):
            adapter.register_dataset(config, database, data_catalog_dataset)

    return database


@pytest.fixture()
def celery_app(redis_container, monkeypatch, config):
    """
    Fixture creating a Celery application instance.

    This celery app is a bit different from the default one,
    as it registers both to the "example" and "celery" queues.
    Typically, these are done on separate workers.
    """
    monkeypatch.setenv("CELERY_BROKER_URL", redis_container.connection_url())
    monkeypatch.setenv("CELERY_RESULT_BACKEND", redis_container.connection_url())

    app = create_celery_app("test")

    register_celery_tasks(app, provider)

    import cmip_ref_celery.worker_tasks  # noqa

    return app


@pytest.fixture()
def celery_worker_parameters():
    return {"queues": ("example", "celery"), "perform_ping_check": False}


def test_celery_solving(db_seeded, config, celery_worker, redis_container, monkeypatch):
    config.executor.executor = "cmip_ref_celery.executor.CeleryExecutor"
    monkeypatch.setenv("CELERY_BROKER_URL", redis_container.connection_url())
    monkeypatch.setenv("CELERY_RESULT_BACKEND", redis_container.connection_url())

    # Run the solver which executes the metrics
    solve_metrics(db_seeded, timeout=10, config=config)

    results = db_seeded.session.query(MetricExecutionResult).all()
    assert len(results)
    for result in results:
        assert result.successful, result.key
        assert (config.paths.results / result.output_fragment / result.path).exists()

    # Attempt to avoid a flakey outcome where the celery tasks aren't cleaned up properly
    time.sleep(3)
    gc.collect()
