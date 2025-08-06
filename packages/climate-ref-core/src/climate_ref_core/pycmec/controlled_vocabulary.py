import pathlib
from collections.abc import Iterable, Sequence
from typing import Any

from attrs import field, frozen, validators
from cattrs import Converter, transform_error
from loguru import logger
from yaml import safe_load

from climate_ref_core.exceptions import ResultValidationError
from climate_ref_core.metric_values import ScalarMetricValue, SeriesMetricValue
from climate_ref_core.pycmec.metric import CMECMetric

RESERVED_DIMENSION_NAMES = {"attributes", "json_structure", "created_at", "updated_at", "value", "id"}
"""
These names are reserved for internal use and should not be used as dimension names.

These names have other meanings that would conflict with the controlled vocabulary.
"""


@frozen
class DimensionValue:
    """
    An allowed value for a dimension
    """

    name: str
    long_name: str
    description: str | None = None
    units: str | None = None


@frozen
class Dimension:
    """
    Description of a dimension in a diagnostic bundle

    This information is also used by the frontend for presentation purposes.
    """

    name: str = field(validator=validators.not_(validators.in_(RESERVED_DIMENSION_NAMES)))
    """
    A short identifier of the dimension.

    This is used as a key in the diagnostic bundle and must be unique.
    """
    long_name: str
    """
    A longer name used for presentation
    """
    description: str
    """
    A short description of the dimension.

    This is used for presentation
    """
    allow_extra_values: bool
    """
    If True, additional non-controlled values are allowed.
    This is used for dimensions where not all the values are known at run time,'
    for example, the model dimension.
    """
    required: bool
    """
    If True, this dimension is required to be specified in the executions.
    """
    values: list[DimensionValue] = field(factory=list)
    """
    The list of controlled values for a given dimension.

    If `allow_extra_values` is False,
    then only these values are valid for the dimension.
    """


@frozen
class CV:
    """
    A collection of controlled dimensions and values used to validate executions.

    A diagnostic bundle does not have to specify all dimensions,
    but any dimensions not in the CV are not permitted.
    """

    # TODO: There might be some additional fields in future if this CV is project-specific

    dimensions: tuple[Dimension, ...] = field()

    @dimensions.validator
    def _validate_dimensions(self, _: Any, value: tuple[Dimension, ...]) -> None:
        """
        Validate that all dimension names are unique and do not conflict with reserved names
        """
        seen = set()
        for dim in value:
            if dim.name in seen:
                raise ValueError(f"Duplicate dimension name: {dim.name}")
            if dim.name in RESERVED_DIMENSION_NAMES:
                raise ValueError(f"Reserved dimension name: {dim.name}")
            seen.add(dim.name)

    def get_dimension_by_name(self, name: str) -> Dimension:
        """
        Get a dimension by name

        Parameters
        ----------
        name
            The name of the dimension

        Returns
        -------
        Dimension
            The dimension with the given name

        Raises
        ------
        KeyError
            If the dimension is not found
        """
        for dim in self.dimensions:
            if dim.name == name:
                return dim
        raise KeyError(f"Dimension {name} not found")

    def _validate_value(self, metric_value: ScalarMetricValue | SeriesMetricValue) -> None:
        """
        Validate a single metric value against the CV
        """
        for k, v in metric_value.dimensions.items():
            try:
                dimension = self.get_dimension_by_name(k)
            except KeyError:
                raise ResultValidationError(f"Unknown dimension: {k!r}")
            if not dimension.allow_extra_values:
                if v not in [dv.name for dv in dimension.values]:
                    raise ResultValidationError(f"Unknown value {v!r} for dimension {k!r}")

        if hasattr(metric_value, "value") and not isinstance(metric_value.value, float):  # pragma: no cover
            # This may not be possible with the current CMECMetric implementation
            raise ResultValidationError(f"Unexpected value: {metric_value.value!r}")

    def validate_metrics(self, metric_value_collection: CMECMetric | Sequence[SeriesMetricValue]) -> None:
        """
        Validate a set of metric values (either scalar or series) against a CV

        The CV describes the accepted dimensions and values within a bundle

        Parameters
        ----------
        metric_value_collection
            A collection of metric values to validate.

        This can be a CMECMetric instance or a sequence of SeriesMetricValue instances.

        Raises
        ------
        ResultValidationError
            If the validation of the dimensions or values fails
        """
        generator: Iterable[SeriesMetricValue | ScalarMetricValue]
        if isinstance(metric_value_collection, CMECMetric):
            generator = metric_value_collection.iter_results()
        else:
            generator = iter(metric_value_collection)

        for result in generator:
            self._validate_value(result)

    @staticmethod
    def load_from_file(filename: pathlib.Path | str) -> "CV":
        """
        Load a CV from disk

        Returns
        -------
            A new CV instance

        """
        convertor = Converter(forbid_extra_keys=True)

        try:
            contents = safe_load(pathlib.Path(filename).read_text(encoding="utf-8"))
            return convertor.structure(contents, CV)
        except Exception as exc:
            logger.error(f"Error loading CV from {filename}")
            for error in transform_error(exc):
                logger.error(error)
            raise
