from concurrent.futures import ThreadPoolExecutor

import pytest

from climate_ref.provider_registry import _register_provider


@pytest.fixture
def db_with_provider(db, provider):
    """A database with the mock provider and its diagnostics persisted."""
    with db.session.begin():
        _register_provider(db, provider)
    return db


@pytest.fixture
def thread_pool():
    pool = ThreadPoolExecutor(max_workers=1)
    yield pool
    pool.shutdown(wait=False)
