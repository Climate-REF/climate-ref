"""
Paging helpers for the ``Select`` statements the builders return.
"""

from typing import Any

from sqlalchemy import Select, func, select
from sqlalchemy.orm import Session


def count(session: Session, stmt: Select[Any]) -> int:
    """Count the rows ``stmt`` returns, ignoring any ordering, offset and limit. Used for ``total_count``."""
    return session.execute(select(func.count()).select_from(stmt.order_by(None).subquery())).scalar_one()


def paginate(stmt: Select[Any], offset: int, limit: int | None) -> Select[Any]:
    """Apply ``offset``/``limit`` to a ``Select``, leaving it untouched when neither is set."""
    if limit is not None:
        return stmt.offset(offset).limit(limit)
    if offset:
        return stmt.offset(offset)
    return stmt
