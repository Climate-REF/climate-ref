from typing import TYPE_CHECKING, Any, ClassVar

from loguru import logger
from sqlalchemy import Column, ForeignKey, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from cmip_ref.models.base import Base, CreatedUpdatedMixin
from cmip_ref_core.pycmec.controlled_vocabulary import CV, Dimension

if TYPE_CHECKING:
    from cmip_ref.models.metric_execution import MetricExecutionResult


class MetricValue(CreatedUpdatedMixin, Base):
    """
    Represents a single metric value

    This value has a number of dimensions which are used to query the metric value.
    These dimensions describe aspects such as the type of statistic being measured,
    the region of interest or the model from which the statistic is being measured.

    The columns in this table are not known statically because the REF can track an arbitrary
    set of dimensions depending on the controlled vocabulary that will be used.
    A call to `register_cv_dimensions` must be made before using this class.
    """

    __tablename__ = "metric_value"

    id: Mapped[int] = mapped_column(primary_key=True)
    metric_execution_result_id: Mapped[int] = mapped_column(ForeignKey("metric_execution_result.id"))

    value: Mapped[float] = mapped_column()
    attributes: Mapped[dict[str, Any]] = mapped_column()

    metric_execution_result: Mapped["MetricExecutionResult"] = relationship(back_populates="values")

    _cv_dimensions: ClassVar[list[str]] = []

    @property
    def dimensions(self) -> dict[str, str]:
        """
        Get the non-null dimensions and their values

        Any changes to the resulting dictionary are not reflected in the object

        Returns
        -------
            Collection of dimensions names and their values
        """
        dims = {}
        for key in self._cv_dimensions:
            value = getattr(self, key)
            if value is not None:
                dims[key] = value
        return dims

    def __repr__(self) -> str:
        return (
            f"<MetricValue "
            f"id={self.id} "
            f"metric_execution={self.metric_execution_result} "
            f"value={self.value} "
            f"dimensions={self.dimensions}>"
        )

    @staticmethod
    def build_dimension_column(dimension: Dimension) -> Column[str]:
        """
        Create a column representing a CV dimension

        Any changes to this functionality likely require a manual database migration
        of the existing columns.

        Parameters
        ----------
        dimension
            Dimension definition to create the column for.

            Currently only the "name" field is being used.

        Returns
        -------
            An instance of a sqlalchemy Column

            This doesn't create the column in the database,
            but enables the ORM to access it.

        """
        return Column(dimension.name, Text, index=True, nullable=True)

    @classmethod
    def register_cv_dimensions(cls, cv: CV) -> None:
        """
        Register the dimensions supplied in the controlled vocabulary

        This has to be done at run-time to support custom CVs.
        Any extra columns already in the database, but not in the CV are ignored.

        Parameters
        ----------
        cv
            Controlled vocabulary being used by the application.
            This controlled vocabulary contains the definitions of the dimensions that can be used.
        """
        for dimension in cv.dimensions:
            target_attribute = dimension.name
            if target_attribute in cls._cv_dimensions:
                continue

            cls._cv_dimensions.append(target_attribute)
            logger.debug(f"Registered MetricValue dimension: {target_attribute}")

            if hasattr(cls, target_attribute):
                # This should only occur in test suite as we don't support removing dimensions at runtime
                logger.warning("Column attribute already exists on MetricValue. Ignoring")
            else:
                setattr(cls, target_attribute, cls.build_dimension_column(dimension))

            # TODO: Check if the underlying table already contains columns

    @classmethod
    def _reset_cv_dimensions(cls) -> None:
        """
        Remove any previously registered dimensions

        Used by the test suite and should not be called at runtime.

        This doesn't remove any previous column definitions due to a limitation that columns in
        declarative classes cannot be removed.
        This means that `hasattr(MetricValue, "old_attribute")`
        will still return True after resetting, but the values will not be included in any results.
        """
        logger.warning(f"Removing MetricValue dimensions: {cls._cv_dimensions}")

        keys = list(cls._cv_dimensions)
        for key in keys:
            cls._cv_dimensions.remove(key)

        assert not len(cls._cv_dimensions)  # noqa

    @classmethod
    def build(
        cls,
        *,
        metric_execution_result_id: int,
        value: float,
        dimensions: dict[str, str],
        attributes: dict[str, Any] | None,
    ) -> "MetricValue":
        """
        Build a MetricValue from a collection of dimensions and a value

        This is a helper method that validates the dimensions supplied and provides an interface
        similar to [cmip_ref_core.pycmec.metrics.MetricValue][].

        Parameters
        ----------
        metric_execution_result_id
            Execution result that created the metric value
        value
            The value of the metric
        dimensions
            Dimensions that describe the metric execution result
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

        return MetricValue(
            metric_execution_result_id=metric_execution_result_id,
            value=value,
            attributes=attributes,
            **dimensions,
        )
