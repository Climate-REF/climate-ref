import pytest
from cmip_ref_metrics_pmp import provider
from cmip_ref_metrics_pmp.variability_modes import ExtratropicalModesOfVariability

from cmip_ref.database import Database
from cmip_ref.executor import handle_execution_result
from cmip_ref.models import MetricExecutionResult as MetricExecutionResultModel
from cmip_ref.solver import solve_metric_executions
from cmip_ref_core.metrics import MetricExecutionResult
from cmip_ref_core.pycmec.metric import CMECMetric
from cmip_ref_core.pycmec.output import CMECOutput

variability_metrics = [
    pytest.param(metric, id=metric.slug)
    for metric in provider.metrics()
    if isinstance(metric, ExtratropicalModesOfVariability)
]


def validate_result(config, result: MetricExecutionResult):
    database = Database.from_config(config)
    metric_execution_result = MetricExecutionResultModel(
        metric_execution_group_id=1,
        dataset_hash=result.definition.metric_dataset.hash,
        output_fragment=str(result.definition.output_fragment()),
    )
    database.session.add(metric_execution_result)
    database.session.flush()

    assert result.successful

    # Validate bundles
    CMECMetric.load_from_json(result.to_output_path(result.metric_bundle_filename))
    CMECOutput.load_from_json(result.to_output_path(result.output_bundle_filename))

    # This checks if the bundles are valid
    handle_execution_result(
        config, database=database, metric_execution_result=metric_execution_result, result=result
    )


@pytest.mark.slow
@pytest.mark.parametrize("metric", variability_metrics)
def test_variability_modes(metric: ExtratropicalModesOfVariability, data_catalog, tmp_path, config, mocker):
    mocker.patch.object(MetricExecutionResultModel, "metric_execution_group")

    # Ensure the conda prefix is set
    provider.configure(config)

    if metric.mode_id in ExtratropicalModesOfVariability.psl_modes:
        pytest.xfail("Missing PSL sample data")

    # Get the first match from the data catalog
    execution = next(
        solve_metric_executions(
            data_catalog=data_catalog,
            metric=metric,
            provider=provider,
        )
    )

    # Run the metric
    definition = execution.build_metric_execution_info(output_root=config.paths.scratch)
    result = metric.run(definition)

    # Check the result
    validate_result(config, result)
