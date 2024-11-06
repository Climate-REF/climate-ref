"""
Example script for running a set of metrics asynchronously
"""

import pathlib
from pathlib import Path

from celery.exceptions import NotRegistered
from loguru import logger
from ref_core.env import env
from ref_core.metrics import Configuration, TriggerInfo

from ref_celery.app import create_celery_app

config = {
    "providers": [
        {
            "name": "example",
            "metrics": ["annual-global-mean-timeseries"],
        }
    ]
}


def _get_changed_dataset() -> Path:
    """
    Get a file that has been changed

    This is a placeholder implementation for now
    """
    return (
        pathlib.Path("CMIP6")
        / "ScenarioMIP"
        / "CSIRO"
        / "ACCESS-ESM1-5"
        / "ssp126"
        / "r1i1p1f1"
        / "Amon"
        / "tas"
        / "gn"
        / "v20210318"
    )


def main():
    """
    Execute a set of celery tasks
    """
    app = create_celery_app("ref_celery")

    # Inquire what tasks are available
    i = app.control.inspect()
    if i.registered() is None:
        logger.error("No tasks are registered by any workers. Check that workers are running")
        return

    tasks = []

    # Create the configuration and trigger objects
    root_output_dir = env.path("REF_OUTPUT_ROOT")
    logger.info(f"Using output directory {root_output_dir}")

    trigger = TriggerInfo(dataset=_get_changed_dataset())

    # Create a task for each metric in each provider
    for provider in config["providers"]:
        for metric in provider["metrics"]:
            metric_name = f"{provider['name']}_{metric}"

            configuration = Configuration(
                output_fragment=Path(metric_name),
            )

            res = app.send_task(
                metric_name,
                kwargs=dict(
                    configuration=configuration,
                    trigger=trigger,
                ),
            )
            tasks.append(res)

    # Wait for all tasks to complete
    for task in tasks:
        try:
            print(task.get(timeout=10))
        except NotRegistered:
            i = app.control.inspect()

            logger.error(f"Task {task.name} is not registered by any workers")
            logger.info(f"Available tasks are: {i.registered()}")
            raise


if __name__ == "__main__":
    main()
