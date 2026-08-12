"""
Celery app creation
"""

import os
from typing import Any

from celery import Celery
from celery.signals import setup_logging, worker_ready
from loguru import logger
from rich.pretty import pretty_repr

from climate_ref.config import Config
from climate_ref_celery.serialisation import register_serialisation
from climate_ref_core.logging import initialise_logging

os.environ.setdefault("CELERY_CONFIG_MODULE", "climate_ref_celery.celeryconf.dev")


def create_celery_app(name: str) -> Celery:
    """
    Create a Celery app

    This function creates a new Celery app with the given name and configuration module.
    The configuration module is loaded from the environment variable `CELERY_CONFIG_MODULE`
    which defaults to `climate_ref_celery.celeryconf.dev` if not set.
    """
    register_serialisation()

    app = Celery(name)
    app.config_from_envvar("CELERY_CONFIG_MODULE")

    return app


@setup_logging.connect
def setup_logging_handler(loglevel: int, **kwargs: Any) -> None:  # pragma: no cover
    """Set up logging for the Celery worker using the celery signal"""
    # We ignore the format passed by Celery instead using our own configuration
    config = Config.default()
    msg_format = config.log_format

    initialise_logging(level=loglevel, format=msg_format, log_directory=config.paths.log)


@worker_ready.connect
def worker_ready_handler(sender: Any = None, **kwargs: Any) -> None:  # pragma: no cover
    """
    Log a message when the worker is ready
    """
    config = Config.default()
    logger.info(f"Worker ready with config {pretty_repr(config)}")

    max_tasks = sender.app.conf.worker_max_tasks_per_child if sender is not None else None
    if config.executor.measure_resources and max_tasks != 1:
        # The bias is not recorded on the row, so it has to be said out loud here.
        logger.warning(
            "CELERY_WORKER_MAX_TASKS_PER_CHILD is not 1, "
            "so worker processes are reused across executions. "
            "A recorded peak memory can include the footprint of an earlier execution "
            "in the same worker."
        )


app = create_celery_app("climate_ref_celery")
