import datetime
from typing import Any

from sqlalchemy import JSON, func
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """
    Base class for all models
    """

    type_annotation_map = {  # noqa: RUF012
        dict[str, Any]: JSON,
    }


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
