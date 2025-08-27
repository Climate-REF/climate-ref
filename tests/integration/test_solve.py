import pytest

from climate_ref.database import Database
from climate_ref.models import Dataset, Execution, ExecutionGroup


@pytest.fixture
def config(config):
    """
    Overwrite the default test config to use the diagnostic providers for CMIP7 Assessment Fast Track
    """
    # Use the local executor to parallelise the executions
    config.executor.executor = "climate_ref.executor.LocalExecutor"
    config.executor.config = {"n": 1}

    # Write the config to disk so it is used by the CLI
    # This overwrites the default config
    config.save()

    return config


def test_solve(sample_data_dir, cmip6_data_catalog, config, invoke_cli):
    num_expected_datasets = cmip6_data_catalog["instance_id"].nunique()
    num_expected_metrics = 13

    db = Database.from_config(config)
    invoke_cli(["datasets", "ingest", "--source-type", "cmip6", str(sample_data_dir / "CMIP6")])
    assert db.session.query(Dataset).count() == num_expected_datasets

    result = invoke_cli(["--verbose", "solve"])
    key = "cmip6_ssp126_ACCESS-ESM1-5_rsut_r1i1p1f1"
    expected_execution_slug = f"example/global-mean-timeseries/{key}"
    assert f"Created new execution group: {expected_execution_slug!r}" in result.stderr
    assert f"Running new execution for execution group: {expected_execution_slug!r}" in result.stderr
    assert db.session.query(ExecutionGroup).count() == num_expected_metrics

    # Running solve again should not trigger any new diagnostic executions
    result = invoke_cli(["--verbose", "solve"])
    assert f"Created new execution group {expected_execution_slug!r}" not in result.stderr
    assert db.session.query(ExecutionGroup).count() == num_expected_metrics
    execution = db.session.query(ExecutionGroup).filter_by(key=key).one()

    assert len(execution.executions[0].datasets) == 2
    assert (
        execution.executions[0].datasets[0].instance_id
        == "CMIP6.ScenarioMIP.CSIRO.ACCESS-ESM1-5.ssp126.r1i1p1f1.Amon.rsut.gn.v20210318"
    )
    assert (
        execution.executions[0].datasets[1].instance_id
        == "CMIP6.ScenarioMIP.CSIRO.ACCESS-ESM1-5.ssp126.r1i1p1f1.fx.areacella.gn.v20210318"
    )

    executions = db.session.query(Execution).all()
    for ex in executions:
        assert ex.successful is True, ex
