from collections.abc import Callable

from celery import Celery
from ref_core.metrics import Metric
from ref_core.providers import MetricsProvider


def metric_task_factory(metric: Metric) -> Callable:
    """
    Create a new task for the given metric
    """

    def task():
        """
        Task to run the metric
        """
        return metric.name

    # def task(configuration: Configuration, trigger: TriggerInfo):
    # metric.run(configuration, trigger)

    return task


def register_celery_tasks(app: Celery, provider: MetricsProvider):
    """
    Register all tasks for the given provider

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
