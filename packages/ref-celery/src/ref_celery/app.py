import os

from celery import Celery

os.environ.setdefault("CELERY_CONFIG_MODULE", "ref_celery.celeryconf.dev")

app = Celery()


def create_celery_app(name: str) -> Celery:
    """
    Create a Celery app
    """
    app = Celery(name)
    app.config_from_envvar("CELERY_CONFIG_MODULE")

    return app
