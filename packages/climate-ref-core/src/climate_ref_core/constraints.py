"""
Dataset selection constraints
"""

import sys
import warnings
from collections import defaultdict
from collections.abc import Mapping
from datetime import datetime
from functools import total_ordering
from typing import Literal, Protocol, runtime_checkable

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

import numpy as np
import pandas as pd
from attrs import field, frozen
from loguru import logger

from climate_ref_core.datasets import SourceDatasetType


@runtime_checkable
class GroupConstraint(Protocol):
    """
    An operation to perform on a group of datasets resulting in a new group of datasets.

    This is applied to a group of datasets representing the inputs to a potential diagnostic execution.

    If the operation results in an empty group, the constraint is considered not satisfied.
    The group must satisfy all constraints to be processed.

    !! warning

        Operations should not mutate the input group, but instead return a new group.
        Mutating the input group may result in unexpected behaviour.
    """

    def apply(self, group: pd.DataFrame, data_catalog: pd.DataFrame) -> pd.DataFrame:
        """
        Perform an operation on the group of datasets.

        A new group of datasets should be returned if modifications are required,
        and the input group should not be modified. If no modifications are required,
        return the input group unchanged.
        If this operation fails, a ConstraintNotSatisfied exception should be raised.

        Parameters
        ----------
        group
            A group of datasets that is being validated.
        data_catalog
            The data catalog of datasets

        Raises
        ------
        ConstraintNotSatisfied
            The operation was not successful

        Returns
        -------
        :
            The updated group of datasets
        """
        ...


def apply_constraint(
    dataframe: pd.DataFrame,
    constraint: GroupConstraint,
    data_catalog: pd.DataFrame,
) -> pd.DataFrame | None:
    """
    Apply a constraint to a group of datasets

    Parameters
    ----------
    dataframe:
        The group of datasets to apply the constraint to.
    constraint
        The constraint to apply.
    data_catalog
        The data catalog of all datasets.

    Returns
    -------
    :
        The updated group of datasets or None if the constraint was not satisfied
    """
    updated_group = constraint.apply(dataframe, data_catalog)
    if updated_group.empty:
        logger.debug(
            "Constraint {} not satisfied for {} rows",
            constraint,
            len(dataframe),
        )
        return None

    return updated_group


def _to_tuple(value: None | str | tuple[str, ...]) -> tuple[str, ...]:
    """
    Normalize value to a tuple of strings.
    """
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(value)


def _to_tuple_dict(value: dict[str, str | tuple[str, ...]]) -> dict[str, tuple[str, ...]]:
    """
    Normalize value to a dict of tuples of strings.
    """
    return {k: _to_tuple(v) for k, v in value.items()}


@frozen
class RequireFacets:
    """
    A constraint that requires datasets to have certain facet values.
    """

    dimension: str
    """The name of the facet to filter on."""

    required_facets: tuple[str, ...] = field(converter=_to_tuple)
    "The required facet values."

    operator: Literal["all", "any"] = "all"
    """Whether all or any of the required facets must be present."""

    group_by: tuple[str, ...] | None = field(converter=_to_tuple, default=None)
    """
    The facets to group the datasets by.

    Each group created by `group_by` must contain at least one dataset where the
    value of the given dimension is in the list of required facet values.

    For example, if there are multiple models and variables in the selection,
    `group_by` can be used to make sure that only those models are selected that
    provide all required variables.
    """

    def apply(self, group: pd.DataFrame, data_catalog: pd.DataFrame) -> pd.DataFrame:
        """
        Filter out groups of datasets that do not provide the required facets
        """
        op = all if self.operator == "all" else any
        select = pd.Series(True, index=group.index)
        groups = [group] if not self.group_by else (g[1] for g in group.groupby(list(self.group_by)))
        for subgroup in groups:
            if not op(value in subgroup[self.dimension].values for value in self.required_facets):
                logger.debug(
                    f"Constraint {self} not satisfied because required facet values "
                    f"not found for group {', '.join(subgroup['path'])}"
                )
                select.loc[subgroup.index] = False
        return group[select]


@frozen
class IgnoreFacets:
    """
    A constraint that ignores certain facet values.

    Datasets with these facet values are removed from the selection.
    """

    facets: dict[str, str | tuple[str, ...]] = field(converter=_to_tuple_dict)
    """The facet values to ignore."""

    def apply(self, group: pd.DataFrame, data_catalog: pd.DataFrame) -> pd.DataFrame:
        """
        Filter out datasets with the ignored facets.
        """
        mask = group[list(self.facets)].isin(self.facets).all(axis="columns")
        if mask.any():
            logger.debug(f"Ignoring files {', '.join(group.loc[mask, 'path'])} becauseof {self}")
        return group[~mask]


@frozen
class AddSupplementaryDataset:
    """
    Include e.g. a cell measure or ancillary variable in the selection.
    """

    supplementary_facets: Mapping[str, str | tuple[str, ...]]
    """
    Facets describing the supplementary dataset.
    """

    matching_facets: tuple[str, ...]
    """
    Facets that must match with datasets in the selection.
    """

    optional_matching_facets: tuple[str, ...]
    """
    Select only the best matching datasets based on similarity with these facets.
    """

    def apply(
        self,
        group: pd.DataFrame,
        data_catalog: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Add a supplementary dataset to the group.
        """
        supplementary_facets: defaultdict[str, tuple[str, ...]] = defaultdict(tuple)
        for facet, values in self.supplementary_facets.items():
            supplementary_facets[facet] = values if isinstance(values, tuple) else (values,)

        for facet in self.matching_facets:
            values = tuple(group[facet].unique())
            supplementary_facets[facet] += values

        supplementary_group = data_catalog
        for facet, values in supplementary_facets.items():
            mask = supplementary_group[facet].isin(values)
            supplementary_group = supplementary_group[mask]
        if not supplementary_group.empty:
            matching_facets = list(self.matching_facets)
            facets = matching_facets + list(self.optional_matching_facets)
            datasets = group[facets].drop_duplicates()
            indices = set()
            for i in range(len(datasets)):
                dataset = datasets.iloc[i]
                # Restrict the supplementary datasets to those that match the main dataset.
                supplementaries = supplementary_group[
                    (supplementary_group[matching_facets] == dataset[matching_facets]).all(1)
                ]
                if not supplementaries.empty:
                    # Select the best matching supplementary dataset based on the optional matching facets.
                    scores = (supplementaries[facets] == dataset).sum(axis=1)
                    matches = supplementaries[scores == scores.max()]
                    if "version" in facets:
                        # Select the latest version if there are multiple matches
                        matches = matches[matches["version"] == matches["version"].max()]
                    # Select one match per dataset
                    indices.add(matches.index[0])

            supplementary_group = supplementary_group.loc[list(indices)].drop_duplicates()

        return pd.concat([group, supplementary_group])

    @classmethod
    def from_defaults(
        cls,
        variable: str,
        source_type: SourceDatasetType,
    ) -> Self:
        """
        Include e.g. a cell measure or ancillary variable in the selection.

        The constraint is created using the defaults for the source_type.

        Parameters
        ----------
        variable:
            The name of the variable to add.
        source_type:
            The source_type of the variable to add.

        Returns
        -------
        :
            A constraint to include a supplementary variable.

        """
        kwargs: dict[SourceDatasetType, dict[str, tuple[str, ...]]] = {
            SourceDatasetType.CMIP6: {
                "matching_facets": (
                    "source_id",
                    "grid_label",
                ),
                "optional_matching_facets": (
                    "table_id",
                    "experiment_id",
                    "member_id",
                    "version",
                ),
            },
            SourceDatasetType.CMIP7: {
                "matching_facets": (
                    "source_id",
                    "grid_label",
                ),
                "optional_matching_facets": (
                    "experiment_id",
                    "variant_label",  # CMIP7 uses variant_label instead of member_id
                ),
            },
        }
        variable_facet: dict[SourceDatasetType, str] = {
            SourceDatasetType.CMIP6: "variable_id",
            SourceDatasetType.CMIP7: "variable_id",
        }

        supplementary_facets = {variable_facet[source_type]: variable}
        source_kwargs = kwargs[source_type]
        return cls(
            supplementary_facets,
            matching_facets=source_kwargs["matching_facets"],
            optional_matching_facets=source_kwargs["optional_matching_facets"],
        )


@frozen
@total_ordering
class PartialDateTime:  # noqa: PLW1641
    """
    A partial datetime object that can be used to compare datetimes.

    Only the specified fields are used for comparison.
    """

    year: int | None = None
    month: int | None = None
    day: int | None = None
    hour: int | None = None
    minute: int | None = None
    second: int | None = None

    @property
    def _attrs(self) -> dict[str, int]:
        """The attributes that are set."""
        return {
            a: v
            for a in self.__slots__  # type: ignore[attr-defined]
            if not a.startswith("_") and (v := getattr(self, a)) is not None
        }

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({', '.join(f'{a}={v}' for a, v in self._attrs.items())})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, datetime):
            msg = (
                f"Can only compare PartialDateTime with `datetime.datetime` "
                f"objects, got object {other} of type {type(other)}"
            )
            raise TypeError(msg)

        for attr, value in self._attrs.items():
            other_value = getattr(other, attr)
            if value != other_value:
                return False
        return True

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, datetime):
            msg = (
                f"Can only compare PartialDateTime with `datetime.datetime` "
                f"objects, got object {other} of type {type(other)}"
            )
            raise TypeError(msg)

        for attr, value in self._attrs.items():
            other_value = getattr(other, attr)
            if value != other_value:
                return value < other_value  # type: ignore[no-any-return]
        return False


@frozen
class RequireTimerange:
    """
    A constraint that requires datasets to have a specific timerange.

    Specify the start and/or end of the required timerange using a precision
    that matches the frequency of the datasets.

    For example, to ensure that datasets at monthly frequency cover the period
    from 2000 to 2010, use start=PartialDateTime(year=2000, month=1) and
    end=PartialDateTime(year=2010, month=12).
    """

    group_by: tuple[str, ...]
    """
    The fields to group the datasets by. Groups that do not cover the timerange
    will be removed.
    """

    start: PartialDateTime | None = None
    """
    The start time of the required timerange. If None, no start time is required.
    """

    end: PartialDateTime | None = None
    """
    The end time of the required timerange. If None, no end time is required.
    """

    def apply(self, group: pd.DataFrame, data_catalog: pd.DataFrame) -> pd.DataFrame:
        """
        Check that all subgroups of the group have a contiguous timerange.
        """
        select = pd.Series(True, index=group.index)
        for _, subgroup in group.dropna(subset=["start_time", "end_time"]).groupby(list(self.group_by)):
            start = subgroup["start_time"].min()
            end = subgroup["end_time"].max()
            result = True
            if self.start is not None and start > self.start:
                logger.debug(
                    f"Constraint {self} not satisfied because start time {start} "
                    f"is after required start time for {', '.join(subgroup['path'])}"
                )
                result = False
            if self.end is not None and end < self.end:
                logger.debug(
                    f"Constraint {self} not satisfied because end time {end} "
                    f"is before required end time for {', '.join(subgroup['path'])}"
                )
                result = False
            if result:
                contiguous_subgroup = RequireContiguousTimerange(group_by=self.group_by).apply(
                    subgroup, data_catalog
                )
                result = len(contiguous_subgroup) == len(subgroup)
            if not result:
                select.loc[subgroup.index] = False
        return group[select]


@frozen
class RequireContiguousTimerange:
    """
    A constraint that requires datasets to have a contiguous timerange.
    """

    group_by: tuple[str, ...]
    """
    The fields to group the datasets by. Groups that are not be contiguous in time
    are removed.
    """

    def apply(self, group: pd.DataFrame, data_catalog: pd.DataFrame) -> pd.DataFrame:
        """
        Check that all subgroups of the group have a contiguous timerange.
        """
        # Maximum allowed time difference between the end of one file and the
        # start of the next file.
        max_timedelta = pd.Timedelta(
            days=31,  # Maximum number of days in a month.
            hours=1,  # Allow for potential rounding errors.
        )

        select = pd.Series(True, index=group.index)

        for _, subgroup in group.dropna(subset=["start_time", "end_time"]).groupby(list(self.group_by)):
            if len(subgroup) < 2:  # noqa: PLR2004
                continue
            sorted_group = subgroup.sort_values("start_time", kind="stable")
            start_series = sorted_group["start_time"]
            end_series = sorted_group["end_time"]
            # Sometimes the elements of start_series.values are of type datetime64[ns]
            # and sometimes its elements are of type datetime.datetime.
            # Convert both arrays to datetime.datetime objects to make sure they
            # can be subtracted.
            with warnings.catch_warnings():
                # We have already mitigated the future change in behaviour of DatetimeProperties.to_pydatetime
                warnings.simplefilter("ignore", FutureWarning)

                if hasattr(start_series, "dt"):
                    start_array = np.array(start_series.dt.to_pydatetime())
                else:
                    start_array = start_series.values  # type: ignore[assignment]
                if hasattr(end_series, "dt"):
                    end_array = np.array(end_series.dt.to_pydatetime())
                else:
                    end_array = end_series.values  # type: ignore[assignment]
            diff = start_array[1:] - end_array[:-1]
            gap_indices = diff > max_timedelta
            if gap_indices.any():
                paths = sorted_group["path"]
                for gap_idx in np.flatnonzero(gap_indices):
                    logger.debug(
                        f"Constraint {self} not satisfied because gap larger "
                        f"than {max_timedelta} found between "
                        f"{paths.iloc[gap_idx]} and {paths.iloc[gap_idx + 1]}"
                    )
                select.loc[subgroup.index] = False

        return group[select]


@frozen
class RequireOverlappingTimerange:
    """
    A constraint that requires datasets to have an overlapping timerange.
    """

    group_by: tuple[str, ...]
    """
    The fields to group the datasets by. There must be overlap in time between
    the groups to fulfill the constraint.
    """

    def apply(self, group: pd.DataFrame, data_catalog: pd.DataFrame) -> pd.DataFrame:
        """
        Check that all subgroups of the group have an overlapping timerange.
        """
        group_with_time = group.dropna(subset=["start_time", "end_time"])
        if len(group_with_time) < 2:  # noqa: PLR2004
            return group

        starts = group_with_time.groupby(list(self.group_by))["start_time"].min()
        ends = group_with_time.groupby(list(self.group_by))["end_time"].max()
        result = starts.max() < ends.min()
        if not result:
            logger.debug(
                f"Constraint {self} not satisfied because no overlapping timerange "
                f"found for groups in {', '.join(group['path'])}"
            )
            return group.loc[[]]
        return group


@frozen
class SelectParentExperiment:
    """
    Include a dataset's parent experiment in the selection
    """

    def apply(self, group: pd.DataFrame, data_catalog: pd.DataFrame) -> pd.DataFrame:
        """
        Include a dataset's parent experiment in the selection

        Not yet implemented
        """
        raise NotImplementedError("This is not implemented yet")  # pragma: no cover
