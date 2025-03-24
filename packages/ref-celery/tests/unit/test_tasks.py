from unittest.mock import Mock

from celery import Celery
from cmip_ref_celery.tasks import _metric_task_factory, register_celery_tasks

from cmip_ref_core.metrics import MetricExecutionDefinition
from cmip_ref_core.providers import MetricsProvider


def test_metric_task_factory():
    # Mock Metric and MetricExecutionDefinition
    mock_metric = Mock()

    definition = MetricExecutionDefinition(
        dataset_key="test", metric_dataset=None, output_directory=None, root_directory=None
    )

    # Create task using factory
    task = _metric_task_factory(mock_metric)

    # Run task and check result
    result = task(definition)
    assert result == mock_metric.run.return_value
    mock_metric.run.assert_called_once_with(definition)


def test_register_celery_tasks(mocker):
    mock_task_factory = mocker.patch("cmip_ref_celery.tasks._metric_task_factory")
    # Mock Celery app and MetricsProvider
    mock_app = Mock(spec=Celery)
    mock_provider = Mock(spec=MetricsProvider)
    mock_provider.slug = "test_provider"
    mock_provider.metrics.return_value = [Mock(), Mock()]
    mock_provider.metrics.return_value[0].slug = "metric1"
    mock_provider.metrics.return_value[1].slug = "metric2"

    # Register tasks
    register_celery_tasks(mock_app, mock_provider)

    # Check that tasks are registered
    assert mock_app.task.call_count == 2
    mock_app.task.assert_any_call(
        mock_task_factory(mock_provider.metrics()[0]), name="test_provider.metric1", queue="test_provider"
    )
    mock_app.task.assert_any_call(
        mock_task_factory(mock_provider.metrics()[1]), name="test_provider.metric2", queue="test_provider"
    )
