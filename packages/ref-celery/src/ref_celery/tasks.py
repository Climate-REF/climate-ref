from collections.abc import Callable
from typing import Any

from celery import Celery
from loguru import logger
from ref_core.metrics import Configuration, Metric, MetricResult, TriggerInfo
from ref_core.providers import MetricsProvider


def metric_task_factory(metric: Metric) -> Callable:
    """
    Create a new task for the given metric
    """

    def task(configuration: Configuration, trigger: TriggerInfo, **kwargs: Any) -> MetricResult:
        """
        Task to run the metric
        """
        logger.info(f"Running metric {metric.name} with configuration {configuration} and trigger {trigger}")

        return metric.run(configuration, trigger)

    return task


def register_celery_tasks(app: Celery, provider: MetricsProvider):
    """
    Register all tasks for the given provider

    This is run on worker startup to register all tasks a given provider

    Parameters
    ----------
    app
        The Celery app to register the tasks with
    provider
        The provider to register tasks for
    """
    for metric in provider:
        print(f"Registering task for metric {metric.name}")
        app.task(metric_task_factory(metric), name=f"{provider.name}_{metric.name}", queue=provider.name)
