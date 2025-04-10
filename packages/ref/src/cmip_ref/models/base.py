import datetime
from typing import Any

from sqlalchemy import JSON, MetaData, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """
    Base class for all models
    """

    type_annotation_map = {  # noqa: RUF012
        dict[str, Any]: JSON,
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


class CreatedUpdatedMixin:
    """
    Mixin for models that have a created_at and updated_at fields
    """

    created_at: Mapped[datetime.datetime] = mapped_column(server_default=func.now())
    """
    When the dataset was added to the database
    """

    updated_at: Mapped[datetime.datetime] = mapped_column(server_default=func.now(), onupdate=func.now())
    """
    When the dataset was updated.
    """
