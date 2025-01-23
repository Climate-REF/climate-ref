from datetime import datetime

import pandas as pd
import pytest

from cmip_ref_core.constraints import (
    AddSupplementaryDataset,
    GroupOperation,
    GroupValidator,
    RequireContiguousTimerange,
    RequireFacets,
    RequireOverlappingTimerange,
    SelectParentExperiment,
    apply_constraint,
)
from cmip_ref_core.datasets import SourceDatasetType
from cmip_ref_core.exceptions import ConstraintNotSatisfied


class TestRequireFacets:
    constraint = RequireFacets(dimension="variable_id", required_facets=["tas", "pr"])

    def test_is_group_validator(self):
        assert isinstance(self.constraint, GroupValidator)
        assert not isinstance(self.constraint, GroupOperation)

    @pytest.mark.parametrize(
        "data, expected",
        [
            (pd.DataFrame({}), False),
            (pd.DataFrame({"invalid": ["tas", "pr"]}), False),
            (pd.DataFrame({"variable_id": ["tas", "pr"]}), True),
            (pd.DataFrame({"variable_id": ["tas", "pr"], "extra": ["a", "b"]}), True),
            (pd.DataFrame({"variable_id": ["tas"]}), False),
            (pd.DataFrame({"variable_id": ["tas"], "extra": ["a"]}), False),
        ],
    )
    def test_validate(self, data, expected):
        assert self.constraint.validate(data) == expected


class TestAddSupplementaryDataset:
    constraint = AddSupplementaryDataset.from_defaults("areacella", SourceDatasetType.CMIP6)

    @pytest.mark.parametrize(
        "data_catalog, expected_rows",
        [
            (
                # Test that missing supplementary files are handled gracefully.
                pd.DataFrame(
                    {
                        "variable_id": ["tas"],
                        "source_id": ["ACCESS-ESM1-5"],
                        "grid_label": ["gn"],
                        "table_id": ["Amon"],
                        "experiment_id": ["historical"],
                        "member_id": ["r1i1p1f1"],
                        "version": ["v20210316"],
                    }
                ),
                [0],
            ),
            (
                # Test that the grid_label matches.
                pd.DataFrame(
                    {
                        "variable_id": ["tas", "areacella", "areacella"],
                        "source_id": ["ACCESS-ESM1-5"] * 3,
                        "grid_label": ["gn", "gn", "gr"],
                        "table_id": ["Amon", "fx", "fx"],
                        "experiment_id": ["historical"] * 3,
                        "member_id": ["r1i1p1f1", "r2i1p1f1", "r1i1p1f1"],
                        "version": ["v20210316"] * 3,
                    }
                ),
                [0, 1],
            ),
            (
                # Test that the source_id matches.
                pd.DataFrame(
                    {
                        "variable_id": ["tas", "areacella", "areacella"],
                        "source_id": ["ACCESS-ESM1-5", "X", "ACCESS-ESM1-5"],
                        "grid_label": ["gn"] * 3,
                        "table_id": ["Amon", "fx", "fx"],
                        "experiment_id": ["historical"] * 3,
                        "member_id": ["r1i1p1f1", "r1i1p1f1", "r2i1p1f1"],
                        "version": ["v20210316"] * 3,
                    }
                ),
                [0, 2],
            ),
            (
                # Test that the latest version is selected.
                pd.DataFrame(
                    {
                        "variable_id": ["tas", "areacella", "areacella"],
                        "source_id": ["ACCESS-ESM1-5"] * 3,
                        "grid_label": ["gn"] * 3,
                        "table_id": ["Amon", "fx", "fx"],
                        "experiment_id": ["historical"] * 3,
                        "member_id": ["r1i1p1f1"] * 3,
                        "version": ["v20210316", "v202200101", "v20230101"],
                    }
                ),
                [0, 2],
            ),
            (
                # Test that the best match for each "tas" dataset is selected.
                pd.DataFrame(
                    {
                        "variable_id": ["tas", "areacella", "tas", "areacella"],
                        "source_id": ["ACCESS-ESM1-5"] * 4,
                        "grid_label": ["gn"] * 4,
                        "table_id": ["Amon", "fx", "Amon", "fx"],
                        "experiment_id": ["historical", "historical", "ssp585", "ssp585"],
                        "member_id": ["r1i1p1f1", "r1i1p1f1", "r2i1p1f1", "r1i1p1f1"],
                        "version": ["v20210316", "v20220101", "v20210316", "v20210316"],
                    }
                ),
                [0, 2, 1, 3],
            ),
        ],
    )
    def test_apply(self, data_catalog, expected_rows):
        group = data_catalog[data_catalog["variable_id"] == "tas"]
        result = self.constraint.apply(group=group, data_catalog=data_catalog)
        expected = data_catalog.loc[expected_rows]
        assert (result == expected).all().all()


class TestContiguousTimerange:
    constraint = RequireContiguousTimerange(group_by=["variable_id"])

    @pytest.mark.parametrize(
        "data, expected",
        [
            (
                pd.DataFrame(
                    {
                        "variable_id": [],
                        "start_time": [],
                        "end_time": [],
                        "path": [],
                    }
                ),
                True,
            ),
            (
                pd.DataFrame(
                    {
                        "variable_id": ["tas", "tas"],
                        "start_time": [
                            datetime(2000, 1, 16, 12),
                            datetime(2001, 1, 16, 12),
                        ],
                        "end_time": [
                            datetime(2000, 12, 16, 12),
                            datetime(2001, 12, 16, 12),
                        ],
                        "path": [
                            "tas_Amon_ACCESS-ESM1-5_historical_r1i1p1f1_gn_200001-200012.nc",
                            "tas_Amon_ACCESS-ESM1-5_historical_r1i1p1f1_gn_200101-200112.nc",
                        ],
                    }
                ),
                True,
            ),
            (
                pd.DataFrame(
                    {
                        "variable_id": ["tas", "tas", "tas"],
                        "start_time": [
                            datetime(2000, 1, 16, 12),
                            datetime(2001, 1, 16, 12),
                            datetime(2003, 1, 16, 12),
                        ],
                        "end_time": [
                            datetime(2000, 12, 16, 12),
                            datetime(2001, 12, 16, 12),
                            datetime(2003, 12, 16, 12),
                        ],
                        "path": [
                            "tas_Amon_ACCESS-ESM1-5_historical_r1i1p1f1_gn_200001-200112.nc",
                            "tas_Amon_ACCESS-ESM1-5_historical_r1i1p1f1_gn_200101-200112.nc",
                            "tas_Amon_ACCESS-ESM1-5_historical_r1i1p1f1_gn_200301-200312.nc",
                        ],
                    }
                ),
                False,
            ),
            (
                pd.DataFrame(
                    {
                        "variable_id": ["tas", "tas", "areacella"],
                        "start_time": [
                            datetime(2000, 1, 16, 12),
                            datetime(2001, 1, 16, 12),
                            None,
                        ],
                        "end_time": [
                            datetime(2000, 12, 16, 12),
                            datetime(2001, 12, 16, 12),
                            None,
                        ],
                        "path": [
                            "tas_Amon_ACCESS-ESM1-5_historical_r1i1p1f1_gn_200001-200012.nc",
                            "tas_Amon_ACCESS-ESM1-5_historical_r1i1p1f1_gn_200101-200112.nc",
                            "areacella_fx_ACCESS-ESM1-5_historical_r1i1p1f1_gn.nc",
                        ],
                    }
                ),
                True,
            ),
            (
                pd.DataFrame(
                    {
                        "variable_id": ["pr", "tas", "tas"],
                        "start_time": [
                            datetime(2000, 1, 16, 12),
                            datetime(2000, 1, 16, 12),
                            datetime(2002, 1, 16, 12),
                        ],
                        "end_time": [
                            datetime(2000, 12, 16, 12),
                            datetime(2000, 12, 16, 12),
                            datetime(2002, 12, 16, 12),
                        ],
                        "path": [
                            "pr_Amon_ACCESS-ESM1-5_historical_r1i1p1f1_gn_200001-200012.nc",
                            "tas_Amon_ACCESS-ESM1-5_historical_r1i1p1f1_gn_200001-200012.nc",
                            "tas_Amon_ACCESS-ESM1-5_historical_r1i1p1f1_gn_200201-200212.nc",
                        ],
                    }
                ),
                False,
            ),
        ],
    )
    def test_validate(self, data, expected):
        assert self.constraint.validate(data) == expected


class TestOverlappingTimerange:
    constraint = RequireOverlappingTimerange(group_by=["variable_id"])

    @pytest.mark.parametrize(
        "data, expected",
        [
            (
                pd.DataFrame(
                    {
                        "variable_id": [],
                        "start_time": [],
                        "end_time": [],
                    }
                ),
                True,
            ),
            (
                pd.DataFrame(
                    {
                        "variable_id": ["tas", "tas", "pr"],
                        "start_time": [
                            datetime(2000, 1, 16, 12),
                            datetime(2001, 1, 16, 12),
                            datetime(2001, 1, 16, 12),
                        ],
                        "end_time": [
                            datetime(2001, 12, 16, 12),
                            datetime(2002, 12, 16, 12),
                            datetime(2014, 12, 16, 12),
                        ],
                    }
                ),
                True,
            ),
            (
                pd.DataFrame(
                    {
                        "variable_id": ["tas", "pr"],
                        "start_time": [
                            datetime(2000, 1, 16, 12),
                            datetime(2002, 1, 16, 12),
                        ],
                        "end_time": [
                            datetime(2001, 12, 16, 12),
                            datetime(2002, 12, 16, 12),
                        ],
                    }
                ),
                False,
            ),
            (
                pd.DataFrame(
                    {
                        "variable_id": ["tas", "tas", "areacella"],
                        "start_time": [
                            datetime(2000, 1, 16, 12),
                            datetime(2001, 1, 16, 12),
                            None,
                        ],
                        "end_time": [
                            datetime(2000, 12, 16, 12),
                            datetime(2001, 12, 16, 12),
                            None,
                        ],
                    }
                ),
                True,
            ),
        ],
    )
    def test_validate(self, data, expected):
        assert self.constraint.validate(data) == expected


class TestSelectParentExperiment:
    def test_is_group_constraint(self):
        constraint = SelectParentExperiment()

        assert isinstance(constraint, GroupOperation)
        assert not isinstance(constraint, GroupValidator)


@pytest.fixture
def data_catalog():
    return pd.DataFrame(
        {
            "variable": ["tas", "pr", "rsut", "tas", "tas"],
            "source_id": ["CESM2", "CESM2", "CESM2", "ACCESS", "CAS"],
        }
    )


def test_apply_constraint_operation(data_catalog):
    #  operation that appends the "rsut" variable to the group
    class ExampleOperation(GroupOperation):
        def apply(self, group: pd.DataFrame, data_catalog: pd.DataFrame) -> pd.DataFrame:
            return pd.concat([group, data_catalog[data_catalog["variable"] == "rsut"]])

    result = apply_constraint(
        data_catalog[data_catalog["variable"] == "tas"],
        ExampleOperation(),
        data_catalog,
    )

    pd.testing.assert_frame_equal(
        result,
        pd.DataFrame(
            {
                "variable": ["tas", "tas", "tas", "rsut"],
                "source_id": ["CESM2", "ACCESS", "CAS", "CESM2"],
            },
            index=[0, 3, 4, 2],
        ),
    )


def test_apply_constraint_operation_mutable(data_catalog):
    class MutableOperation(GroupOperation):
        def apply(self, group: pd.DataFrame, data_catalog: pd.DataFrame) -> pd.DataFrame:
            group["variable"] = "new"
            return group

    orig_data_catalog = data_catalog.copy()
    result = apply_constraint(
        data_catalog,
        MutableOperation(),
        None,
    )

    assert (result["variable"] == "new").all()

    # Mutating the group impacts the original data catalog
    with pytest.raises(AssertionError):
        pd.testing.assert_frame_equal(data_catalog, orig_data_catalog)


def test_apply_constraint_operation_raises():
    class RaisesOperation(GroupOperation):
        def apply(self, group: pd.DataFrame, data_catalog: pd.DataFrame) -> pd.DataFrame:
            raise ConstraintNotSatisfied("Test exception")

    assert (
        apply_constraint(
            pd.DataFrame(),
            RaisesOperation(),
            pd.DataFrame(),
        )
        is None
    )


def test_apply_constraint_empty():
    assert (
        apply_constraint(
            pd.DataFrame(),
            RequireFacets(dimension="variable_id", required_facets=["tas", "pr"]),
            pd.DataFrame(),
        )
        is None
    )


def test_apply_constraint_validate(data_catalog):
    result = apply_constraint(
        data_catalog,
        RequireFacets(dimension="variable", required_facets=["tas", "pr"]),
        pd.DataFrame(),
    )
    pd.testing.assert_frame_equal(result, data_catalog)


def test_apply_constraint_validate_invalid(data_catalog):
    assert (
        apply_constraint(
            data_catalog,
            RequireFacets(dimension="variable", required_facets=["missing", "pr"]),
            pd.DataFrame(),
        )
        is None
    )
