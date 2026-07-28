import importlib

import pytest


@pytest.fixture
def load_base(monkeypatch):
    def _load(**env):
        for key, value in env.items():
            monkeypatch.setenv(key, value)
        return importlib.reload(importlib.import_module("climate_ref_celery.celeryconf.base"))

    yield _load

    # The module is process-global, so restore the unpatched values for later tests
    monkeypatch.undo()
    importlib.reload(importlib.import_module("climate_ref_celery.celeryconf.base"))


def test_accept_content_reads_a_comma_separated_list(load_base):
    base = load_base(CELERY_ACCEPT_CONTENT="json,ref-json,pickle")

    assert base.accept_content == ["json", "ref-json", "pickle"]


def test_compression_defaults_to_gzip(load_base):
    base = load_base()

    assert base.task_compression == "gzip"
    assert base.result_compression == "gzip"


def test_compression_can_be_overridden(load_base):
    base = load_base(CELERY_TASK_COMPRESSION="bzip2", CELERY_RESULT_COMPRESSION="bzip2")

    assert base.task_compression == "bzip2"
    assert base.result_compression == "bzip2"


def test_compression_empty_string_disables(load_base):
    base = load_base(CELERY_TASK_COMPRESSION="", CELERY_RESULT_COMPRESSION="")

    assert base.task_compression is None
    assert base.result_compression is None
