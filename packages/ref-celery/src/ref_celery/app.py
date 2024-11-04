from celery import Celery

app = Celery()


class Config:
    """Celery configuration"""

    broker_url = "redis://localhost:6379/1"
    result_backend = "redis://localhost:6379/1"


def create_celery_app(name: str) -> Celery:
    """
    Create a Celery app
    """
    app = Celery(name)
    app.config_from_object(Config)

    return app
