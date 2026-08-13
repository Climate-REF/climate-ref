import pytest
from celery import Celery
from climate_ref_celery.app import create_celery_app


def test_create_celery_app(monkeypatch):
    app = create_celery_app("test")

    assert isinstance(app, Celery)
    assert app.main == "test"

    assert app.configured is False
    assert app.conf["broker_url"] == "redis://localhost:6379/1"
    assert app.conf["task_serializer"] == "ref-json"
    assert app.configured


def test_create_celery_app_does_not_accept_pickle():
    app = create_celery_app("test")

    assert "pickle" not in app.conf["accept_content"]


def test_create_celery_app_registers_the_serialiser():
    from kombu.serialization import dumps, loads  # noqa: PLC0415

    create_celery_app("test")

    content_type, content_encoding, body = dumps({"a": 1}, serializer="ref-json")

    assert loads(body, content_type, content_encoding) == {"a": 1}


def test_create_celery_app_invalid_config(monkeypatch):
    monkeypatch.setenv("CELERY_CONFIG_MODULE", "unknown")
    app = create_celery_app("test")

    with pytest.raises(ImportError):
        # Celery only loads the configuration when it is accessed
        app.conf["task_serializer"]
