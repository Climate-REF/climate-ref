import cmip_ref.solver
from cmip_ref.database import Database
from cmip_ref.models import Dataset, MetricExecution
from cmip_ref.provider_registry import ProviderRegistry, _register_provider


class ExampleProviderRegistry(ProviderRegistry):
    def build_from_db(db: Database) -> "ExampleProviderRegistry":
        """
        Create a ProviderRegistry instance containing only the Example provider.

        Parameters
        ----------
        db
            Database instance

        Returns
        -------
        :
            A new ProviderRegistry instance
        """
        # TODO: We don't yet have any tables to represent metrics providers
        from cmip_ref_metrics_example import provider as example_provider

        with db.session.begin_nested():
            _register_provider(db, example_provider)
        return ProviderRegistry(providers=[example_provider])


def test_solve(sample_data_dir, cmip6_data_catalog, config, invoke_cli, monkeypatch):
    num_expected_datasets = cmip6_data_catalog["instance_id"].nunique()
    num_expected_metrics = 9

    db = Database.from_config(config)
    monkeypatch.setattr(cmip_ref.solver, "ProviderRegistry", ExampleProviderRegistry)
    invoke_cli(["datasets", "ingest", "--source-type", "cmip6", str(sample_data_dir)])
    assert db.session.query(Dataset).count() == num_expected_datasets

    result = invoke_cli(["--verbose", "solve"])
    expected_metric_execution_name = "dataset1_ACCESS-ESM1-5_rsut_ssp126_r1i1p1f1"
    assert f"Created metric execution {expected_metric_execution_name}" in result.stderr
    assert "Running metric" in result.stderr
    assert db.session.query(MetricExecution).count() == num_expected_metrics

    # Running solve again should not trigger any new metric executions
    result = invoke_cli(["--verbose", "solve"])
    assert f"Created metric execution {expected_metric_execution_name}" not in result.stderr
    assert db.session.query(MetricExecution).count() == num_expected_metrics
    execution = db.session.query(MetricExecution).filter_by(key=expected_metric_execution_name).one()

    assert len(execution.results[0].datasets) == 1
    assert (
        execution.results[0].datasets[0].instance_id
        == "CMIP6.ScenarioMIP.CSIRO.ACCESS-ESM1-5.ssp126.r1i1p1f1.Amon.rsut.gn.v20210318"
    )
