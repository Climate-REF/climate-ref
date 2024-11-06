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
from typing import Any

from celery import Celery
from loguru import logger
from mypy_extensions import Arg, KwArg
from ref_core.metrics import Configuration, Metric, MetricResult, TriggerInfo
from ref_core.providers import MetricsProvider


def metric_task_factory(
    metric: Metric,
) -> Callable[[Arg(Configuration, "configuration"), Arg(TriggerInfo, "trigger"), KwArg(Any)], MetricResult]:
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
    for metric in provider:
        print(f"Registering task for metric {metric.name}")
        app.task(metric_task_factory(metric), name=f"{provider.name}_{metric.name}", queue=provider.name)
