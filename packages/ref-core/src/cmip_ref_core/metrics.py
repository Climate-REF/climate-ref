from __future__ import annotations

import pathlib
from abc import abstractmethod
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import pandas as pd
from attrs import field, frozen

from cmip_ref_core.constraints import GroupConstraint
from cmip_ref_core.datasets import FacetFilter, MetricDataset, SourceDatasetType
from cmip_ref_core.pycmec.metric import CMECMetric
from cmip_ref_core.pycmec.output import CMECOutput

if TYPE_CHECKING:
    from cmip_ref_core.providers import CommandLineMetricsProvider, MetricsProvider


def ensure_relative_path(path: pathlib.Path | str, root_directory: pathlib.Path) -> pathlib.Path:
    """
    Ensure that a path is relative to a root directory

    If a path an absolute path, but not relative to the root directory, a ValueError is raised.

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
    if path.is_absolute():
        return path.relative_to(root_directory)
    return path


@frozen
class MetricExecutionDefinition:
    """
    Definition of a metric execution.

    This represents the information needed by a metric to perform a single execution of the metric
    """

    key: str
    """
    A unique identifier for the metric execution

    The key is a hash of the group by values for the datasets used in the metric execution.
    Duplicate keys will occur when new datasets are available that match the same group by values.
    """

    metric_dataset: MetricDataset
    """
    Collection of datasets required for the metric execution
    """

    output_directory: pathlib.Path
    """
    Output directory to store the output of the metric execution
    """

    _root_directory: pathlib.Path
    """
    Root directory for storing the output of the metric execution
    """

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
class MetricResult:
    """
    The result of running a metric.

    The content of the result follows the Earth System Metrics and Diagnostics Standards
    ([EMDS](https://github.com/Earth-System-Diagnostics-Standards/EMDS/blob/main/standards.md)).
    """

    # Do we want to load a serialised version of the output bundle here or just a file path?

    definition: MetricExecutionDefinition
    """
    The definition of the metric execution that produced this result.
    """

    output_bundle_filename: pathlib.Path | None = None
    """
    Filename of the output bundle file relative to the execution directory.

    The contents of this file are defined by
    [EMDS standard](https://github.com/Earth-System-Diagnostics-Standards/EMDS/blob/main/standards.md#common-output-bundle-format-)
    """

    metric_bundle_filename: pathlib.Path | None = None
    """
    Filename of the metric bundle file relative to the execution directory.

    The contents of this file are defined by
    [EMDS standard](https://github.com/Earth-System-Diagnostics-Standards/EMDS/blob/main/standards.md#common-metric-output-format-)
    """

    successful: bool = False
    """
    Whether the metric ran successfully.
    """
    # Log info is in the output bundle file already, but is definitely useful

    @staticmethod
    def build_from_output_bundle(
        definition: MetricExecutionDefinition,
        *,
        cmec_output_bundle: CMECOutput | dict[str, Any],
        cmec_metric_bundle: CMECMetric | dict[str, Any],
    ) -> MetricResult:
        """
        Build a MetricResult from a CMEC output bundle.

        Parameters
        ----------
        definition
            The execution defintion.
        cmec_output_bundle
            An output bundle in the CMEC format.
        cmec_metric_bundle
            An metric bundle in the CMEC format.

        Returns
        -------
        :
            A prepared MetricResult object.
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
        bundle_path = definition.to_output_path("output.json")
        cmec_output.dump_to_json(bundle_path)

        definition.to_output_path(filename=None).mkdir(parents=True, exist_ok=True)
        bundle_path = definition.to_output_path("metric.json")
        cmec_metric.dump_to_json(bundle_path)

        return MetricResult(
            definition=definition,
            output_bundle_filename=pathlib.Path("output.json"),
            metric_bundle_filename=pathlib.Path("metric.json"),
            successful=True,
        )

    @staticmethod
    def build_from_failure(definition: MetricExecutionDefinition) -> MetricResult:
        """
        Build a failed metric result.

        This is a placeholder.
        Additional log information should still be captured in the output bundle.
        """
        return MetricResult(
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
    Definition of the input datasets that a metric requires to run.

    This is used to create groups of datasets.
    Each group will result in an execution of the metric
    and defines the input data for that execution.

    The data catalog is filtered according to the `filters` field,
    then grouped according to the `group_by` field,
    and then each group is checked that it satisfies the `constraints`.
    Each such group will be processed as a separate execution of the metric.
    """

    source_type: SourceDatasetType
    """
    Type of the source dataset (CMIP6, CMIP7 etc)
    """

    filters: tuple[FacetFilter, ...]
    """
    Filters to apply to the data catalog of datasets.

    This is used to reduce the set of datasets to only those that are required by the metric.
    The filters are applied iteratively to reduce the set of datasets.
    """

    group_by: tuple[str, ...] | None
    """
    The fields to group the datasets by.

    This group by operation is performed after the data catalog is filtered according to `filters`.
    Each group will contain a unique combination of values from the metadata fields,
    and will result in a separate execution of the metric.
    If `group_by=None`, all datasets will be processed together as a single execution.

    The unique values of the group by fields are used to create a unique key for the metric execution.
    Changing the value of `group_by` may invalidate all previous metric executions.
    """

    constraints: tuple[GroupConstraint, ...] = field(factory=tuple)
    """
    Constraints that must be satisfied when executing a given metric run

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
            for facet, value in facet_filter.facets.items():
                clean_value = value if isinstance(value, tuple) else (value,)

                if facet not in data_catalog.columns:
                    raise KeyError(
                        f"Facet {facet!r} not in data catalog columns: {data_catalog.columns.to_list()}"
                    )

                mask = data_catalog[facet].isin(clean_value)
                if not facet_filter.keep:
                    mask = ~mask

                data_catalog = data_catalog[mask]
        return data_catalog


@runtime_checkable
class AbstractMetric(Protocol):
    """
    Interface for the calculation of a metric.

    This is a very high-level interface to provide maximum scope for the metrics packages
    to have differing assumptions.
    The configuration and output of the metric should follow the
    Earth System Metrics and Diagnostics Standards formats as much as possible.

    A metric can be executed multiple times,
    each time targeting a different group of input data.
    The groups are determined using the grouping the data catalog according to the `group_by` field
    in the `DataRequirement` object using one or more metadata fields.
    Each group must conform with a set of constraints,
    to ensure that the correct data is available to run the metric.
    Each group will then be processed as a separate execution of the metric.

    See (ref_example.example.ExampleMetric)[] for an example implementation.
    """

    name: str
    """
    Name of the metric being run

    This should be unique for a given provider,
    but multiple providers can implement the same metric.
    """

    slug: str
    """
    Unique identifier for the metric

    Defaults to the name of the metric in lowercase with spaces replaced by hyphens.
    """

    data_requirements: tuple[DataRequirement, ...]
    """
    Description of the required datasets for the current metric

    This information is used to filter the a data catalog of both CMIP and/or observation datasets
    that are required by the metric.

    Any modifications to the input data will new metric calculation.
    """

    provider: MetricsProvider
    """
    The provider that provides the metric.
    """

    def run(self, definition: MetricExecutionDefinition) -> MetricResult:
        """
        Run the metric on the given configuration.

        The implementation of this method method is left to the metrics providers.

        A CMEC-compatible package can use: TODO: Add link to CMEC metric wrapper

        Parameters
        ----------
        definition : MetricExecutionDefinition
            The configuration to run the metric on.

        Returns
        -------
        MetricResult
            The result of running the metric.
        """


class Metric(AbstractMetric):
    """
    Interface for the calculation of a metric.

    This is a very high-level interface to provide maximum scope for the metrics packages
    to have differing assumptions.
    The configuration and output of the metric should follow the
    Earth System Metrics and Diagnostics Standards formats as much as possible.

    A metric can be executed multiple times,
    each time targeting a different group of input data.
    The groups are determined using the grouping the data catalog according to the `group_by` field
    in the `DataRequirement` object using one or more metadata fields.
    Each group must conform with a set of constraints,
    to ensure that the correct data is available to run the metric.
    Each group will then be processed as a separate execution of the metric.

    See (ref_example.example.ExampleMetric)[] for an example implementation.
    """

    def __init__(self) -> None:
        super().__init__()
        self._provider: MetricsProvider | None = None

    @property
    def provider(self) -> MetricsProvider:
        """
        The provider that provides the metric.
        """
        if self._provider is None:
            msg = f"Please register {self} with a MetricsProvider before using it."
            raise ValueError(msg)
        return self._provider

    @provider.setter
    def provider(self, value: MetricsProvider) -> None:
        self._provider = value


class CommandLineMetric(Metric):
    """
    Metric that can be run from the command line.
    """

    provider: CommandLineMetricsProvider

    @abstractmethod
    def build_cmd(self, definition: MetricExecutionDefinition) -> Iterable[str]:
        """
        Build the command to run the metric on the given configuration.

        Parameters
        ----------
        definition : MetricExecutionDefinition
            The configuration to run the metric on.

        Returns
        -------
        :
            A command that can be run with :func:`subprocess.run`.
        """

    @abstractmethod
    def build_metric_result(self, definition: MetricExecutionDefinition) -> MetricResult:
        """
        Build the result from running the metric on the given configuration.

        Parameters
        ----------
        definition : MetricExecutionDefinition
            The configuration to run the metric on.

        Returns
        -------
        :
            The result of running the metric.
        """

    def run(self, definition: MetricExecutionDefinition) -> MetricResult:
        """
        Run the metric on the given configuration.

        Parameters
        ----------
        definition : MetricExecutionDefinition
            The configuration to run the metric on.

        Returns
        -------
        :
            The result of running the metric.
        """
        cmd = self.build_cmd(definition)
        self.provider.run(cmd)
        return self.build_metric_result(definition)
