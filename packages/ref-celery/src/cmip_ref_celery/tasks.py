"""
Task generation and registration for Celery

This module provides a factory function to create Celery tasks for metrics.
These celery tasks are then registered with the Celery app to enable them to be run asynchronously.

Since the metric definition may be in a different virtual environment it is not possible to directly
import the provider and create the tasks in both the worker and the main process.

Instead, the tasks are registered only in the worker process.
The main process can then send tasks to the worker using the task name.
The main process is responsible for tracking what metrics have been registered
and to respond to new workers coming online.
"""

from collections.abc import Callable

from celery import Celery
from loguru import logger

from cmip_ref_core.metrics import Metric, MetricExecutionDefinition, MetricExecutionResult
from cmip_ref_core.providers import MetricsProvider


def generate_task_name(provider: MetricsProvider, metric: Metric) -> str:
    """
    Generate the name of the task for the given provider and metric
    """
    return f"{provider.slug}.{metric.slug}"


def _metric_task_factory(
    metric: Metric,
) -> Callable[
    [MetricExecutionDefinition],
    MetricExecutionResult,
]:
    """
    Create a new task for the given metric
    """

    def task(definition: MetricExecutionDefinition) -> MetricExecutionResult:
        """
        Task to run the metric
        """
        logger.info(f"Running metric {metric.name} with definition {definition}")

        return metric.run(definition)

    return task


def register_celery_tasks(app: Celery, provider: MetricsProvider) -> None:
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
    for metric in provider.metrics():
        print(f"Registering task for metric {metric.name}")
        app.task(  # type: ignore
            _metric_task_factory(metric),
            name=generate_task_name(provider, metric),
            queue=provider.slug,
        )
