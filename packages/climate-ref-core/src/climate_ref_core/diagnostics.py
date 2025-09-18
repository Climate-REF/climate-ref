"""
Diagnostic interface
"""

from __future__ import annotations

import pathlib
from collections.abc import Iterable, Sequence
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import pandas as pd
from attrs import field, frozen

from climate_ref_core.constraints import GroupConstraint
from climate_ref_core.datasets import ExecutionDatasetCollection, FacetFilter, SourceDatasetType
from climate_ref_core.metric_values import SeriesMetricValue
from climate_ref_core.metric_values.typing import SeriesDefinition
from climate_ref_core.pycmec.metric import CMECMetric
from climate_ref_core.pycmec.output import CMECOutput

if TYPE_CHECKING:
    from climate_ref_core.providers import CommandLineDiagnosticProvider, DiagnosticProvider


def ensure_relative_path(path: pathlib.Path | str, root_directory: pathlib.Path) -> pathlib.Path:
    """
    Ensure that a path is relative to a root directory

    If a path is an absolute path, but not relative to the root directory, a ValueError is raised.

    Parameters
    ----------
    path
        The path to check
    root_directory
        The root directory that the path should be relative to

    Raises
    ------
    ValueError
        If the path is not relative to the root directory

    Returns
    -------
        The path relative to the root directory
    """
    path = pathlib.Path(path)
    try:
        return path.relative_to(root_directory)
    except ValueError:
        if path.is_absolute():
            raise
    return path


@frozen
class ExecutionDefinition:
    """
    Definition of an execution of a diagnostic

    This represents the information needed by a diagnostic to perform an execution
    for a specific set of datasets fulfilling the requirements.
    """

    diagnostic: Diagnostic
    """
    The diagnostic that is being executed
    """

    key: str
    """
    The unique identifier for the datasets in the diagnostic execution group.

    The key is derived from the datasets in the group using facet values.
    New datasets which match the same group by facet values will result in the same
    key.
    """

    datasets: ExecutionDatasetCollection
    """
    Collection of datasets required for the diagnostic execution
    """

    output_directory: pathlib.Path
    """
    Output directory to store the output of the diagnostic execution
    """

    _root_directory: pathlib.Path
    """
    Root directory for storing the output of the diagnostic execution
    """

    def execution_slug(self) -> str:
        """
        Get a slug for the execution
        """
        return f"{self.diagnostic.full_slug()}/{self.key}"

    def to_output_path(self, filename: pathlib.Path | str | None) -> pathlib.Path:
        """
        Get the absolute path for a file in the output directory

        Parameters
        ----------
        filename
            Name of the file to get the full path for

        Returns
        -------
        :
            Full path to the file in the output directory
        """
        if filename is None:
            return self.output_directory
        else:
            return self.output_directory / filename

    def as_relative_path(self, filename: pathlib.Path | str) -> pathlib.Path:
        """
        Get the relative path of a file in the output directory

        Parameters
        ----------
        filename
            Path to a file in the output directory

            If this is an absolute path, it will be converted to a relative path within the output directory.

        Returns
        -------
        :
            Relative path to the file in the output directory
        """
        return ensure_relative_path(filename, self.output_directory)

    def output_fragment(self) -> pathlib.Path:
        """
        Get the relative path of the output directory to the root output directory

        Returns
        -------
        :
            Relative path to the output directory
        """
        return self.output_directory.relative_to(self._root_directory)


@frozen
class ExecutionResult:
    """
    The result of executing a diagnostic.

    This execution may or may not be successful.

    The content of the result follows the Earth System Metrics and Diagnostics Standards
    ([EMDS](https://github.com/Earth-System-Diagnostics-Standards/EMDS/blob/main/standards.md)).
    """

    definition: ExecutionDefinition
    """
    The definition of the diagnostic execution that produced this result.
    """

    output_bundle_filename: pathlib.Path | None = None
    """
    Filename of the output bundle file relative to the execution directory.

    The contents of this file are defined by
    [EMDS standard](https://github.com/Earth-System-Diagnostics-Standards/EMDS/blob/main/standards.md#common-output-bundle-format-)
    """

    metric_bundle_filename: pathlib.Path | None = None
    """
    Filename of the diagnostic bundle file relative to the execution directory.

    The contents of this file are defined by
    [EMDS standard](https://github.com/Earth-System-Diagnostics-Standards/EMDS/blob/main/standards.md#common-metric-output-format-)
    """

    successful: bool = False
    """
    Whether the diagnostic execution ran successfully.
    """

    series_filename: pathlib.Path | None = None
    """
    A collection of series metric values that were extracted from the execution.

    These are written to a CSV file in the output directory.
    """

    @staticmethod
    def build_from_output_bundle(
        definition: ExecutionDefinition,
        *,
        cmec_output_bundle: CMECOutput | dict[str, Any],
        cmec_metric_bundle: CMECMetric | dict[str, Any],
        series: Sequence[SeriesMetricValue] = tuple(),
    ) -> ExecutionResult:
        """
        Build a ExecutionResult from a CMEC output bundle.

        Parameters
        ----------
        definition
            The execution definition.
        cmec_output_bundle
            An output bundle in the CMEC format.
        cmec_metric_bundle
            An diagnostic bundle in the CMEC format.
        series
            Series metric values extracted from the execution.

        Returns
        -------
        :
            A prepared ExecutionResult object.
            The output bundle will be written to the output directory.
        """
        if isinstance(cmec_output_bundle, dict):
            cmec_output = CMECOutput.model_validate(cmec_output_bundle)
        else:
            cmec_output = cmec_output_bundle

        if isinstance(cmec_metric_bundle, dict):
            cmec_metric = CMECMetric.model_validate(cmec_metric_bundle)
        else:
            cmec_metric = cmec_metric_bundle

        definition.to_output_path(filename=None).mkdir(parents=True, exist_ok=True)

        output_filename = "output.json"
        metric_filename = "diagnostic.json"
        series_filename = "series.json"

        cmec_output.dump_to_json(definition.to_output_path(output_filename))
        cmec_metric.dump_to_json(definition.to_output_path(metric_filename))
        SeriesMetricValue.dump_to_json(definition.to_output_path(series_filename), series)

        # We are using relative paths for the output files for portability of the results
        return ExecutionResult(
            definition=definition,
            output_bundle_filename=pathlib.Path(output_filename),
            metric_bundle_filename=pathlib.Path(metric_filename),
            series_filename=pathlib.Path(series_filename),
            successful=True,
        )

    @staticmethod
    def build_from_failure(definition: ExecutionDefinition) -> ExecutionResult:
        """
        Build a failed diagnostic result.

        This is a placeholder.
        Additional log information should still be captured in the output bundle.
        """
        return ExecutionResult(
            output_bundle_filename=None, metric_bundle_filename=None, successful=False, definition=definition
        )

    def to_output_path(self, filename: str | pathlib.Path | None) -> pathlib.Path:
        """
        Get the absolute path for a file in the output directory

        Parameters
        ----------
        filename
            Name of the file to get the full path for

            If None the path to the output bundle will be returned

        Returns
        -------
        :
            Full path to the file in the output directory
        """
        return self.definition.to_output_path(filename)

    def as_relative_path(self, filename: pathlib.Path | str) -> pathlib.Path:
        """
        Get the relative path of a file in the output directory

        Parameters
        ----------
        filename
            Path to a file in the output directory

            If this is an absolute path, it will be converted to a relative path within the output directory.

        Returns
        -------
        :
            Relative path to the file in the output directory
        """
        return self.definition.as_relative_path(filename)


@frozen(hash=True)
class DataRequirement:
    """
    Definition of the input datasets that a diagnostic requires to run.

    This is used to create groups of datasets.
    Each group will result in an execution of the diagnostic
    and defines the input data for that execution.

    The data catalog is filtered according to the `filters` field,
    then grouped according to the `group_by` field,
    and then each group is checked that it satisfies the `constraints`.
    Each such group will be processed as a separate execution of the diagnostic.
    """

    source_type: SourceDatasetType
    """
    Type of the source dataset (CMIP6, CMIP7 etc)
    """

    filters: tuple[FacetFilter, ...]
    """
    Filters to apply to the data catalog of datasets.

    This is used to reduce the set of datasets to only those that are required by the diagnostic.
    The filters are applied iteratively to reduce the set of datasets.
    """

    group_by: tuple[str, ...] | None
    """
    The fields to group the datasets by.

    This group by operation is performed after the data catalog is filtered according to `filters`.
    Each group will contain a unique combination of values from the metadata fields,
    and will result in a separate execution of the diagnostic.
    If `group_by=None`, all datasets will be processed together as a single execution.

    The unique values of the group by fields are used to create a unique key for the diagnostic execution.
    Changing the value of `group_by` may invalidate all previous diagnostic executions.
    """

    constraints: tuple[GroupConstraint, ...] = field(factory=tuple)
    """
    Constraints that must be satisfied when executing a given diagnostic run

    All of the constraints must be satisfied for a given group to be run.
    Each filter is applied iterative to a set of datasets to reduce the set of datasets.
    This is effectively an AND operation.
    """

    def apply_filters(self, data_catalog: pd.DataFrame) -> pd.DataFrame:
        """
        Apply filters to a DataFrame-based data catalog.

        Parameters
        ----------
        data_catalog
            DataFrame to filter.
            Each column contains a facet

        Returns
        -------
        :
            Filtered data catalog
        """
        for facet_filter in self.filters:
            data_catalog = facet_filter.apply(data_catalog)
        return data_catalog


@runtime_checkable
class AbstractDiagnostic(Protocol):
    """
    Interface for the calculation of a diagnostic.

    This is a very high-level interface to provide maximum scope for the diagnostic packages
    to have differing assumptions about how they work.
    The configuration and output of the diagnostic should follow the
    Earth System Metrics and Diagnostics Standards formats as much as possible.

    A diagnostic can be executed multiple times,
    each time targeting a different group of input data.
    The groups are determined using the grouping the data catalog according to the `group_by` field
    in the `DataRequirement` object using one or more metadata fields.
    Each group must conform with a set of constraints,
    to ensure that the correct data is available to run the diagnostic.
    Each group will then be processed as a separate execution of the diagnostic.

    See (cmip_ref_example.example.ExampleDiagnostic)[] for an example implementation.
    """

    name: str
    """
    Name of the diagnostic being run

    This should be unique for a given provider,
    but multiple providers can implement the same diagnostic.
    """

    slug: str
    """
    Unique identifier for the diagnostic.

    Defaults to the name of the diagnostic in lowercase with spaces replaced by hyphens.
    """

    data_requirements: Sequence[DataRequirement] | Sequence[Sequence[DataRequirement]]
    """
    Description of the required datasets for the current diagnostic

    This information is used to filter the a data catalog of both model and/or observation datasets
    that are required by the diagnostic.

    A diagnostic may specify either a single set of requirements (i.e. a list of `DataRequirement`'s),
    or multiple sets of requirements (i.e. a list of lists of `DataRequirement`'s).
    Each of these sets of requirements will be processed separately which is effectively an OR operation
    across the sets of requirements.

    Any modifications to the input data will new diagnostic calculation.
    """

    facets: tuple[str, ...]
    """
    Facets that are used to describe the values produced by this metric.

    These facets represent the dimensions that can be used to uniquely identify a metric value.
    Each metric value should have a unique set of keys for the dimension (this isn't checked).
    A faceted search can then be performed on these facets.

    These facets must be present in the controlled vocabulary otherwise a `KeyError` exception
    is raised.
    """

    series: Sequence[SeriesDefinition]
    """
    Definition of the series that are produced by the diagnostic.
    """

    provider: DiagnosticProvider
    """
    The provider that provides the diagnostic.
    """

    def execute(self, definition: ExecutionDefinition) -> None:
        """
        Execute the diagnostic on the given configuration.

        The implementation of this method is left to the diagnostic providers.
        The results should be written to the output directory of the execution definition.
        These are later used to build the output bundle and the diagnostic bundle.

        This may occur in a separate process (or python environment in the case of a `CommandLineDiagnostic`).

        Parameters
        ----------
        definition
            The configuration to run the diagnostic on.
        """
        ...

    def build_execution_result(self, definition: ExecutionDefinition) -> ExecutionResult:
        """
        Build the result from running the diagnostic on the given configuration.

        This can be replayed later to build the result from the output execution.

        Parameters
        ----------
        definition
            The configuration to run the diagnostic on.

        Returns
        -------
        :
            The result of running the diagnostic.
        """
        ...


class Diagnostic(AbstractDiagnostic):
    """
    Interface for the calculation of a diagnostic.

    This is a very high-level interface to provide maximum scope for the diagnostic packages
    to have differing assumptions.
    The configuration and output of the diagnostic should follow the
    Earth System Metrics and Diagnostics Standards formats as much as possible.

    A diagnostic can be executed multiple times,
    each time targeting a different group of input data.
    The groups are determined using the grouping the data catalog according to the `group_by` field
    in the `DataRequirement` object using one or more metadata fields.
    Each group must conform with a set of constraints,
    to ensure that the correct data is available to run the diagnostic.
    Each group will then be processed as a separate execution of the diagnostic.

    See (climate_ref_example.example.ExampleDiagnostic)[] for an example implementation.
    """

    series: Sequence[SeriesDefinition] = tuple()

    def __init__(self) -> None:
        super().__init__()
        self._provider: DiagnosticProvider | None = None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"

    def full_slug(self) -> str:
        """
        Full slug that describes the diagnostic

        This is a combination of the provider slug and the diagnostic slug.
        """
        return f"{self.provider.slug}/{self.slug}"

    @property
    def provider(self) -> DiagnosticProvider:
        """
        The provider that provides the diagnostic.
        """
        if self._provider is None:
            msg = f"Please register {self} with a DiagnosticProvider before using it."
            raise ValueError(msg)
        return self._provider

    @provider.setter
    def provider(self, value: DiagnosticProvider) -> None:
        self._provider = value

    def run(self, definition: ExecutionDefinition) -> ExecutionResult:
        """
        Run the diagnostic on the given configuration.

        This executes the diagnostic and builds the result from the output bundle.

        Parameters
        ----------
        definition
            The configuration to run the diagnostic on.
        """
        # Execute the diagnostic
        # This may be run in a separate process (or python environment)
        self.execute(definition)

        # Build the result from the output bundle
        return self.build_execution_result(definition)


class CommandLineDiagnostic(Diagnostic):
    """
    Diagnostic that can be run from the command line.
    """

    provider: CommandLineDiagnosticProvider

    def build_cmd(self, definition: ExecutionDefinition) -> Iterable[str]:
        """
        Build the command to run the diagnostic on the given configuration.

        Parameters
        ----------
        definition
            The configuration to run the diagnostic on.

        Returns
        -------
        :
            A command that can be run with :func:`subprocess.run`.
        """
        return []

    def execute(self, definition: ExecutionDefinition) -> None:
        """
        Run the diagnostic on the given configuration.

        Parameters
        ----------
        definition
            The configuration to run the diagnostic on.

        Returns
        -------
        :
            The result of running the diagnostic.
        """
        cmd = self.build_cmd(definition)
        self.provider.run(cmd)
