"""
Celery app creation
"""

import os

from celery import Celery

os.environ.setdefault("CELERY_CONFIG_MODULE", "ref_celery.celeryconf.dev")

app = Celery()


def create_celery_app(name: str) -> Celery:
    """
    Create a Celery app

    This function creates a new Celery app with the given name and configuration module.
    The configuration module is loaded from the environment variable `CELERY_CONFIG_MODULE`
    which defaults to `ref_celery.celeryconf.dev` if not set.
    """
    app = Celery(name)
    app.config_from_envvar("CELERY_CONFIG_MODULE")

    return app
