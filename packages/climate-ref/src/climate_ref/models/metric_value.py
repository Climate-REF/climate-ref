import enum
import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar

from sqlalchemy import ForeignKey, event, select
from sqlalchemy.orm import Mapped, Session, mapped_column, relationship

from climate_ref.models.base import Base
from climate_ref.models.mixins import CreatedUpdatedMixin, DimensionMixin

if TYPE_CHECKING:
    from climate_ref.models.execution import Execution


def _content_hash(payload: list[Any]) -> str:
    """
    Hash a JSON-serialisable payload into a stable content digest.

    The serialisation is fixed (compact separators, ``ensure_ascii=False``) so the digest
    is reproducible across platforms and runs. Keep it stable: the same serialisation is
    relied on by the series-index migration backfill.
    """
    serialised = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(serialised.encode()).hexdigest()


class MetricValueType(enum.Enum):
    """
    Type of metric value

    This is used to determine how the metric value should be interpreted.
    """

    # The value is a single number
    SCALAR = "scalar"

    # The value is a list of numbers
    SERIES = "series"


class MetricValue(DimensionMixin, CreatedUpdatedMixin, Base):
    """
    Represents a single metric value

    This is a base class for different types of metric values (e.g. scalar, series) which
    are stored in a single table using single table inheritance.

    This value has a number of dimensions which are used to query the diagnostic values.
    These dimensions describe aspects such as the type of statistic being measured,
    the region of interest or the model from which the statistic is being measured.

    The columns in this table are not known statically because the REF can track an arbitrary
    set of dimensions depending on the controlled vocabulary that will be used.
    A call to `register_cv_dimensions` must be made before using this class.
    """

    __tablename__ = "metric_value"

    __mapper_args__: ClassVar[Mapping[str, str]] = {  # type: ignore
        "polymorphic_on": "type",
    }

    _cv_dimensions: ClassVar[list[str]] = []

    id: Mapped[int] = mapped_column(primary_key=True)
    execution_id: Mapped[int] = mapped_column(ForeignKey("execution.id"), index=True)

    attributes: Mapped[dict[str, Any]] = mapped_column()

    execution: Mapped["Execution"] = relationship(back_populates="values")

    type: Mapped[MetricValueType] = mapped_column(index=True)
    """
    Type of metric value

    This value is used to determine how the metric value should be interpreted.
    """

    def __repr__(self) -> str:
        return f"<MetricValue id={self.id} execution={self.execution} dimensions={self.dimensions}>"


class ScalarMetricValue(MetricValue):
    """
    A scalar value with an associated dimensions

    This is a subclass of MetricValue that is used to represent a scalar value.
    """

    __mapper_args__: ClassVar[Mapping[str, Any]] = {  # type: ignore
        "polymorphic_identity": MetricValueType.SCALAR,
    }

    # This is a scalar value
    value: Mapped[float] = mapped_column(nullable=True)

    def __repr__(self) -> str:
        return (
            f"<ScalarMetricValue "
            f"id={self.id} execution={self.execution} dimensions={self.dimensions} value={self.value}>"
        )

    @classmethod
    def build(
        cls,
        *,
        execution_id: int,
        value: float,
        dimensions: dict[str, str],
        attributes: dict[str, Any] | None,
    ) -> "MetricValue":
        """
        Build a MetricValue from a collection of dimensions and a value

        This is a helper method that validates the dimensions supplied and provides an interface
        similar to [climate_ref_core.metric_values.ScalarMetricValue][].

        Parameters
        ----------
        execution_id
            Execution that created the diagnostic value
        value
            The value of the diagnostic
        dimensions
            Dimensions that describe the diagnostic execution result
        attributes
            Optional additional attributes to describe the value,
            but are not in the controlled vocabulary.

        Raises
        ------
        KeyError
            If an unknown dimension was supplied.

            Dimensions must exist in the controlled vocabulary.

        Returns
        -------
            Newly created MetricValue
        """
        for k in dimensions:
            if k not in cls._cv_dimensions:
                raise KeyError(f"Unknown dimension column '{k}'")

        return ScalarMetricValue(
            execution_id=execution_id,
            value=value,
            attributes=attributes,
            **dimensions,
        )


class SeriesIndex(Base):
    """
    A shared 1-d index axis for series metric values

    Many series share the same index (for example a common monthly time axis),
    so the index is stored once here and referenced by
    [SeriesMetricValue.index_id][climate_ref.models.metric_value.SeriesMetricValue]
    rather than duplicated on every row. Axes are deduplicated by ``hash``.
    """

    __tablename__ = "index_axis"

    id: Mapped[int] = mapped_column(primary_key=True)

    hash: Mapped[str] = mapped_column(unique=True, index=True)
    """
    Content hash of ``(name, values)``.

    The axes are deduplicated by this hash, so identical axes will share the same row and be referenced by id.
    """

    name: Mapped[str] = mapped_column(nullable=True)
    """Name of the index (e.g. ``"time"``). Used for presentation."""

    values: Mapped[list[float | int | str]] = mapped_column()
    """The 1-d array of index values."""

    length: Mapped[int] = mapped_column()
    """Number of points in the index; used to validate series lengths."""

    def __repr__(self) -> str:
        return f"<SeriesIndex id={self.id} name={self.name} length={self.length}>"

    @staticmethod
    def compute_hash(name: str | None, values: Sequence[float | int | str]) -> str:
        """
        Compute the content hash used to deduplicate identical axes.

        The hash covers both the name and the ordered values,
        so two axes are only shared when they are genuinely identical.
        """
        return _content_hash([name, list(values)])

    @classmethod
    def get_or_create(
        cls, session: Session, name: str | None, values: Sequence[float | int | str]
    ) -> "SeriesIndex":
        """
        Return the existing axis with this content, or create and flush a new one.

        Parameters
        ----------
        session
            Active database session.
        name
            Name of the index.
        values
            1-d array of index values.

        Returns
        -------
            The shared [SeriesIndex][climate_ref.models.metric_value.SeriesIndex] row.
        """
        digest = cls.compute_hash(name, values)
        existing = session.execute(select(cls).where(cls.hash == digest)).scalar_one_or_none()
        if existing is not None:
            return existing
        axis = cls(hash=digest, name=name, values=list(values), length=len(values))
        session.add(axis)
        session.flush()
        return axis


class SeriesMetricValue(MetricValue):
    """
    A 1d series with associated dimensions

    This is a subclass of MetricValue that is used to represent a series.
    This can be used to represent time series, vertical profiles or other 1d data.
    """

    __mapper_args__: ClassVar[Mapping[str, Any]] = {  # type: ignore
        "polymorphic_identity": MetricValueType.SERIES,
    }

    values: Mapped[list[float | int]] = mapped_column(nullable=True)

    index_id: Mapped[int | None] = mapped_column(ForeignKey("index_axis.id"), nullable=True, index=True)
    index_axis: Mapped["SeriesIndex | None"] = relationship(lazy="joined")

    reference_id: Mapped[str | None] = mapped_column(nullable=True, index=True)
    """
    Content hash of the reference payload, for reference (observation) series only.

    Two reference series with an identical payload share the same ``reference_id``,
    so observations can be deduplicated deterministically across executions.
    It is ``None`` for model series. See
    [compute_reference_id][climate_ref.models.metric_value.SeriesMetricValue.compute_reference_id].
    """

    @staticmethod
    def compute_reference_id(
        values: Sequence[float | int],
        index: Sequence[float | int | str] | None,
        reference_source_id: str | None,
    ) -> str:
        """
        Compute the content hash that deduplicates an identical reference payload.

        The hash covers the values, the index and the reference source,
        so two reference series are only treated as the same observation
        when their payloads are genuinely identical.
        Keep this payload stable: it is the deduplication key used downstream.
        """
        return _content_hash([list(values), list(index) if index is not None else None, reference_source_id])

    @property
    def index(self) -> list[float | int | str] | None:
        """The 1-d index values, resolved from the shared axis."""
        return self.index_axis.values if self.index_axis is not None else None

    @property
    def index_name(self) -> str | None:
        """The name of the index, resolved from the shared axis."""
        return self.index_axis.name if self.index_axis is not None else None

    def __repr__(self) -> str:
        return (
            f"<SeriesMetricValue id={self.id} execution={self.execution} "
            f"dimensions={self.dimensions} index_name={self.index_name}>"
        )

    @classmethod
    def build(
        cls,
        *,
        execution_id: int,
        values: list[float | int],
        index_axis: "SeriesIndex",
        dimensions: dict[str, str],
        attributes: dict[str, Any] | None,
    ) -> "MetricValue":
        """
        Build a database object from a series

        Parameters
        ----------
        execution_id
            Execution that created the diagnostic value
        values
            1-d array of values
        index_axis
            The shared index axis for this series, obtained via
            [SeriesIndex.get_or_create][climate_ref.models.metric_value.SeriesIndex.get_or_create]
        dimensions
            Dimensions that describe the diagnostic execution result
        attributes
            Optional additional attributes to describe the value,
            but are not in the controlled vocabulary.

        Raises
        ------
        KeyError
            If an unknown dimension was supplied.

            Dimensions must exist in the controlled vocabulary.
        ValueError
            If the length of values and index do not match.

        Returns
        -------
            Newly created MetricValue
        """
        for k in dimensions:
            if k not in cls._cv_dimensions:
                raise KeyError(f"Unknown dimension column '{k}'")

        if len(values) != index_axis.length:
            raise ValueError(f"Index length ({index_axis.length}) must match values length ({len(values)})")

        return SeriesMetricValue(
            execution_id=execution_id,
            values=values,
            index_axis=index_axis,
            attributes=attributes,
            **dimensions,
        )


@event.listens_for(SeriesMetricValue, "before_insert")
@event.listens_for(SeriesMetricValue, "before_update")
def validate_series_lengths(mapper: Any, connection: Any, target: SeriesMetricValue) -> None:
    """
    Validate that values and the referenced index axis have matching lengths

    This is done on insert and update to ensure that the database is consistent.
    """
    axis = target.index_axis
    if target.values is not None and axis is not None and len(target.values) != axis.length:
        raise ValueError(f"Index length ({axis.length}) must match values length ({len(target.values)})")
