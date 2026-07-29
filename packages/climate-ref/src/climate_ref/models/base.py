import math
from typing import Any, TypeVar

from sqlalchemy import JSON, Dialect, MetaData
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.types import TypeDecorator


def replace_non_finite(value: Any) -> Any:
    """
    Replace NaN and infinities with None, recursing into lists and dicts

    JSON has no literal for either.
    `json.dumps` writes the bare tokens `NaN`, `Infinity` and `-Infinity` by default,
    which Python reads back but PostgreSQL rejects when it validates the value on insert.

    Parameters
    ----------
    value
        Value about to be serialised as JSON.

    Returns
    -------
    :
        The value with every non-finite float replaced by None.
    """
    if hasattr(value, "item"):
        # numpy scalars are not JSON serialisable,
        # and a non-finite float32 must be caught like any other float
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: replace_non_finite(item) for key, item in value.items()}
    if isinstance(value, list):
        return [replace_non_finite(item) for item in value]
    return value


class SanitisedJSON(TypeDecorator[Any]):
    """
    JSON column that maps non-finite floats to null.

    A missing value in a metric series is entirely ordinary,
    so a series that contains one must still be storable.
    Mapping to null keeps the series the same length,
    which is what a consumer plotting it needs.
    """

    impl = JSON
    cache_ok = True

    # JSON stores a Python None as the JSON literal null rather than SQL NULL.
    should_evaluate_none = True

    def process_bind_param(self, value: Any, dialect: Dialect) -> Any:
        """Sanitise the value on its way into the database."""
        return replace_non_finite(value)


class Base(DeclarativeBase):
    """
    Base class for all models
    """

    type_annotation_map = {  # noqa: RUF012
        dict[str, Any]: SanitisedJSON,
        list[float | int]: SanitisedJSON,
        list[float | int | str]: SanitisedJSON,
    }
    metadata = MetaData(
        # Enforce a common naming convention for constraints
        # https://alembic.sqlalchemy.org/en/latest/naming.html
        naming_convention={
            "ix": "ix_%(column_0_label)s",
            "uq": "uq_%(table_name)s_%(column_0_name)s",
            "ck": "ck_%(table_name)s_`%(constraint_name)s`",
            "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
            "pk": "pk_%(table_name)s",
        }
    )


Table = TypeVar("Table", bound=Base)
