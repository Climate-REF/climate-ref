"""
Runs an integration test for the Celery executor with a Redis broker.

This test requires a running Redis server, which is started as a Docker container.
"""

import gc
import time

import pytest
from cmip_ref_celery.app import create_celery_app
from cmip_ref_celery.tasks import register_celery_tasks
from cmip_ref_metrics_example import provider

from cmip_ref.database import Database
from cmip_ref.datasets.cmip6 import CMIP6DatasetAdapter
from cmip_ref.models import MetricExecutionResult
from cmip_ref.solver import solve_metrics


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
    celery_url = redis_container.connection_url()
    monkeypatch.setenv("CELERY_BROKER_URL", celery_url)
    monkeypatch.setenv("CELERY_RESULT_BACKEND", celery_url)

    app = create_celery_app("test")

    register_celery_tasks(app, provider)

    import cmip_ref_celery.worker_tasks  # noqa

    return app


@pytest.fixture()
def celery_worker_parameters():
    return {"queues": ("example", "celery"), "perform_ping_check": False}


def test_celery_solving(db_seeded, config, celery_worker, redis_container, monkeypatch):
    config.executor.executor = "cmip_ref_celery.executor.CeleryExecutor"
    celery_url = redis_container.connection_url()
    monkeypatch.setenv("CELERY_BROKER_URL", celery_url)
    monkeypatch.setenv("CELERY_RESULT_BACKEND", celery_url)

    # Run the solver which executes the metrics
    solve_metrics(db_seeded, timeout=10, config=config)

    results = db_seeded.session.query(MetricExecutionResult).all()
    assert len(results)
    for result in results:
        assert result.successful
        assert (config.paths.results / result.output_fragment / result.path).exists()

    # Attempt to avoid a flakey outcome where the celery tasks aren't cleaned up properly
    time.sleep(1)
    gc.collect()
