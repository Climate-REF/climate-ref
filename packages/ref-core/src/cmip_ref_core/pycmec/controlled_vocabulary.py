import pathlib

from attrs import field, frozen
from cattrs import Converter, transform_error
from loguru import logger
from ruamel.yaml import YAML

from cmip_ref_core.exceptions import ResultValidationError
from cmip_ref_core.pycmec.metric import CMECMetric

yaml = YAML()


@frozen
class DimensionValue:
    """
    An allowed value for a dimension
    """

    name: str
    long_name: str
    description: str | None
    units: str


@frozen
class Dimension:
    """
    Description of a dimension in a metric bundle

    This information is also used by the frontend for presentation purposes.
    """

    name: str
    """
    A short idenfifier of the dimension.

    This is used as a key in the metric bundle.
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
    If True, this dimension is required to be specified in the results.
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
    A collection of controlled dimensions and values used to validate results.

    A metric bundle does not have to specify all dimensions,
    but any dimensions not in the CV are not permitted.
    """

    # TODO: There might be some additional fields in future if this CV is project-specific

    dimensions: tuple[Dimension, ...]

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

    def validate_metrics(self, metric_bundle: CMECMetric) -> None:
        """
        Validate a metric bundle against a CV

        The CV describes the accepted dimensions and values within a bundle

        Parameters
        ----------
        metric_bundle

        Raises
        ------
        ResultValidationError
            If the validation of the dimensions or values fails
        """
        for result in metric_bundle.iter_results():
            for k, v in result.dimensions.items():
                try:
                    dimension = self.get_dimension_by_name(k)
                except KeyError:
                    raise ResultValidationError(f"Unknown dimension: {k!r}")
                if not dimension.allow_extra_values:
                    if v not in [dv.name for dv in dimension.values]:
                        raise ResultValidationError(f"Unknown value {v!r} for dimension {k!r}")
            if not isinstance(result.value, float):  # pragma: no cover
                # This may not be possible with the current CMECMetric implementation
                raise ResultValidationError(f"Unexpected value: {result.value!r}")

    @staticmethod
    def load_from_file(filename: pathlib.Path | str) -> "CV":
        """
        Load a CV from disk

        Returns
        -------
            A new CV instance

        """
        convertor = Converter(forbid_extra_keys=True)
        contents = yaml.load(pathlib.Path(filename))

        try:
            return convertor.structure(contents, CV)
        except Exception as exc:
            logger.error(f"Error loading CV from {filename}")
            for error in transform_error(exc):
                logger.error(error)
            raise
