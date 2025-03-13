"""
Runs an integration test for the Celery executor with a Redis broker.

This test requires a running Redis server, which is started as a Docker container.
"""

import gc
import platform
import time
from pathlib import Path

import cmip_ref_metrics_esmvaltool
import cmip_ref_metrics_ilamb
import cmip_ref_metrics_pmp
import pandas as pd
import pytest
from cmip_ref_celery.app import create_celery_app
from cmip_ref_celery.tasks import register_celery_tasks

from cmip_ref.config import default_metric_providers
from cmip_ref.database import Database
from cmip_ref.models import MetricExecution
from cmip_ref.solver import solve_metrics


@pytest.fixture
def config(config, monkeypatch):
    config.metric_providers = default_metric_providers()

    # Put the conda environments in a shared location
    # ROOT / .ref / software
    config.paths.software = Path(__file__).parents[3] / ".ref" / "software"
    config.save()

    return config


@pytest.fixture
def celery_app(redis_container, config, monkeypatch):
    """
    Fixture creating a Celery application instance.

    This celery app is a bit different from the default one,
    as it registers both to the "example" and "celery" queues.
    Typically, these are done on separate workers.
    """
    monkeypatch.setenv("CELERY_BROKER_URL", redis_container.connection_url(1))
    monkeypatch.setenv("CELERY_RESULT_BACKEND", redis_container.connection_url(1))

    app = create_celery_app("test")

    register_celery_tasks(app, cmip_ref_metrics_ilamb.provider)
    register_celery_tasks(app, cmip_ref_metrics_pmp.provider)
    register_celery_tasks(app, cmip_ref_metrics_esmvaltool.provider)

    import cmip_ref_celery.worker_tasks  # noqa

    return app


@pytest.fixture()
def celery_worker_parameters():
    return {"queues": ("esmvaltool", "pmp", "ilamb", "celery"), "perform_ping_check": False}


def create_execution_dataframe(executions):
    data = []
    for execution in executions:
        assert len(execution.results) == 1
        result = execution.results[0]

        data.append(
            {
                "metric": execution.metric.slug,
                "provider": execution.metric.provider.slug,
                "execution_id": execution.id,
                "result_id": result.id,
                "execution_key": execution.key,
                "successful": result.successful,
            }
        )

    df = pd.DataFrame(data)
    return df


@pytest.skip(reason="Fails on GitHub Actions")
@pytest.mark.slow
def test_solve_ar7_ft(
    sample_data_dir,
    config,
    invoke_cli,
    monkeypatch,
):
    # Arm-based MacOS users will need to set the environment variable `MAMBA_PLATFORM=osx-64`
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        monkeypatch.setenv("MAMBA_PLATFORM", "osx-64")

    assert len(config.metric_providers) == 3

    db = Database.from_config(config)

    # Ingest the sample data
    invoke_cli(["datasets", "ingest", "--source-type", "cmip6", str(sample_data_dir / "CMIP6")])
    invoke_cli(["datasets", "ingest", "--source-type", "obs4mips", str(sample_data_dir / "obs4MIPs")])

    # Solve
    # This will also create conda environments for the metric providers
    invoke_cli(["--verbose", "solve", "--timeout", f"{60 * 60}"])

    executions = db.session.query(MetricExecution).all()
    df = create_execution_dataframe(executions)

    print(df)

    assert len(df["provider"].unique()) == 3
    assert df["successful"].all(), df[["metric", "successful"]]


@pytest.skip(reason="Fails on GitHub Actions")
@pytest.mark.slow
def test_solve_celery_ar7_ft(
    sample_data_dir, config, invoke_cli, monkeypatch, celery_worker, redis_container
):
    monkeypatch.setenv("CELERY_BROKER_URL", redis_container.connection_url(1))
    monkeypatch.setenv("CELERY_RESULT_BACKEND", redis_container.connection_url(1))

    config.executor.executor = "cmip_ref_celery.executor.CeleryExecutor"
    config.save()

    # Arm-based MacOS users will need to set the environment variable `MAMBA_PLATFORM=osx-64`
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        monkeypatch.setenv("MAMBA_PLATFORM", "osx-64")

    assert len(config.metric_providers) == 3

    db = Database.from_config(config)

    # Ingest the sample data
    invoke_cli(["datasets", "ingest", "--source-type", "cmip6", str(sample_data_dir / "CMIP6")])
    invoke_cli(["datasets", "ingest", "--source-type", "obs4mips", str(sample_data_dir / "obs4MIPs")])

    # Solve
    # This will also create conda environments for the metric providers
    solve_metrics(db, timeout=60 * 60, config=config)

    executions = db.session.query(MetricExecution).all()
    df = create_execution_dataframe(executions)

    print(df)

    assert len(df["provider"].unique()) == 3
    assert df["successful"].all(), df[["metric", "successful"]]

    # Attempt to avoid a flakey outcome where the celery tasks aren't cleaned up properly
    time.sleep(2)
    gc.collect()
