from contextlib import contextmanager

import pytest
from sqlalchemy import event


@pytest.fixture
def count_queries():
    """Return a context manager yielding a list of every SQL statement a database executes."""

    @contextmanager
    def _count(db):
        statements: list[str] = []
        engine = db.session.get_bind()

        def _record(_conn, _cursor, statement, *_args):
            statements.append(statement)

        event.listen(engine, "before_cursor_execute", _record)
        try:
            yield statements
        finally:
            event.remove(engine, "before_cursor_execute", _record)

    return _count
