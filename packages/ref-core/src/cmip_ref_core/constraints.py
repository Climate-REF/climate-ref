import sys
from collections import defaultdict
from collections.abc import Mapping
from typing import Protocol, runtime_checkable

if sys.version_info < (3, 11):
    from typing_extensions import Self
else:
    from typing import Self

import numpy as np
import pandas as pd
from attrs import frozen
from loguru import logger

from cmip_ref_core.datasets import SourceDatasetType
from cmip_ref_core.exceptions import ConstraintNotSatisfied


@runtime_checkable
class GroupValidator(Protocol):
    """
    A constraint that must be satisfied when executing a given metric run.

    All constraints must be satisfied for a given group to be run.
    """

    def validate(self, group: pd.DataFrame) -> bool:
        """
        Validate if the constraint is satisfied by the dataset.

        This is executed after the apply method to determine if the constraint is satisfied.
        If the constraint is not satisfied, the group will not be executed.

        Parameters
        ----------
        group
            A group of datasets that is being validated.

        Returns
        -------
        :
            Whether the constraint is satisfied
        """
        ...


@runtime_checkable
class GroupOperation(Protocol):
    """
    An operation to perform on a group of datasets resulting in a new group of datasets.

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


GroupConstraint = GroupOperation | GroupValidator
"""
A constraint that must be satisfied when executing a given metric run.

This is applied to a group of datasets representing the inputs to a potential metric execution.
The group must satisfy all constraints to be processed.

This can include operations that are applied to a group of datasets which may modify the group,
but may also include validators that check if the group satisfies a certain condition.
"""


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
    try:
        updated_group = (
            constraint.apply(dataframe, data_catalog) if isinstance(constraint, GroupOperation) else dataframe
        )

        valid = constraint.validate(updated_group) if isinstance(constraint, GroupValidator) else True
        if not valid:
            logger.debug(f"Constraint {constraint} not satisfied for {dataframe}")
            raise ConstraintNotSatisfied(f"Constraint {constraint} not satisfied for {dataframe}")
    except ConstraintNotSatisfied:
        logger.debug(f"Constraint {constraint} not satisfied for {dataframe}")
        return None

    return updated_group


@frozen
class RequireFacets:
    """
    A constraint that requires a dataset to have certain facets.
    """

    dimension: str
    required_facets: tuple[str, ...]

    def validate(self: Self, group: pd.DataFrame) -> bool:
        """
        Check that the required facets are present in the group
        """
        if self.dimension not in group:
            logger.warning(f"Dimension {self.dimension} not present in group {group}")
            return False
        return all(value in group[self.dimension].values for value in self.required_facets)


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
        self: Self,
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

        if not supplementary_group.empty and self.optional_matching_facets:
            facets = list(self.matching_facets + self.optional_matching_facets)
            datasets = group[facets].drop_duplicates()
            indices = set()
            for i in range(len(datasets)):
                scores = (supplementary_group[facets] == datasets.iloc[i]).sum(axis=1)
                matches = supplementary_group[scores == scores.max()]
                # Select the latest version if there are multiple matches
                matches = matches[matches["version"] == matches["version"].max()]
                indices.add(matches.index[0])
            supplementary_group = supplementary_group.loc[list(indices)].drop_duplicates()

        return pd.concat([group, supplementary_group])

    @classmethod
    def from_defaults(
        cls: type[Self],
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
        kwargs = {
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
            }
        }
        variable_facet = {
            SourceDatasetType.CMIP6: "variable_id",
        }

        supplementary_facets = {variable_facet[source_type]: variable}
        return cls(supplementary_facets, **kwargs[source_type])


@frozen
class RequireContiguousTimerange:
    """
    A constraint that requires datasets to have a contiguous timerange.
    """

    group_by: tuple[str, ...]
    """
    The fields to group the datasets by. Each group must be contiguous in time
    to fulfill the constraint.
    """

    def validate(self: Self, group: pd.DataFrame) -> bool:
        """
        Check that all subgroups of the group have a contiguous timerange.
        """
        # Maximum allowed time difference between the end of one file and the
        # start of the next file.
        max_timedelta = pd.Timedelta(
            days=31,  # Maximum number of days in a month.
            hours=1,  # Allow for potential rounding errors.
        )
        group = group.dropna(subset=["start_time", "end_time"])
        if len(group) < 2:  # noqa: PLR2004
            return True

        for _, subgroup in group.groupby(list(self.group_by)):
            if len(subgroup) < 2:  # noqa: PLR2004
                continue
            sorted_group = subgroup.sort_values("start_time", kind="stable")
            start_series = sorted_group["start_time"]
            end_series = sorted_group["end_time"]
            # Sometimes the elements of start_series.values are of type datetime64[ns]
            # and sometimes its elements are of type datetime.datetime.
            # Convert both arrays to datetime.datetime objects to make sure they
            # can be subtracted.
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
                        f"Constraint {self.__class__.__name__} not satisfied "
                        f"because gap larger than {max_timedelta} found between "
                        f"{paths.iloc[gap_idx]} and {paths.iloc[gap_idx+1]}"
                    )
                return False
        return True


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

    def validate(self: Self, group: pd.DataFrame) -> bool:
        """
        Check that all subgroups of the group have an overlapping timerange.
        """
        group = group.dropna(subset=["start_time", "end_time"])
        if len(group) < 2:  # noqa: PLR2004
            return True

        starts = group.groupby(list(self.group_by))["start_time"].min()
        ends = group.groupby(list(self.group_by))["end_time"].max()
        return starts.max() < ends.min()  # type: ignore[no-any-return]


@frozen
class SelectParentExperiment:
    """
    Include a dataset's parent experiment in the selection
    """

    def apply(self: Self, group: pd.DataFrame, data_catalog: pd.DataFrame) -> pd.DataFrame:
        """
        Include a dataset's parent experiment in the selection

        Not yet implemented
        """
        raise NotImplementedError("This is not implemented yet")  # pragma: no cover
