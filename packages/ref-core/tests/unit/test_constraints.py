from datetime import datetime

import pandas as pd
import pytest

from cmip_ref_core.constraints import (
    GroupOperation,
    GroupValidator,
    RequireContiguousTimerange,
    RequireFacets,
    RequireOverlappingTimerange,
    SelectParentExperiment,
    apply_constraint,
)
from cmip_ref_core.exceptions import ConstraintNotSatisfied


class TestRequireFacets:
    validator = RequireFacets(dimension="variable_id", required_facets=["tas", "pr"])

    def test_is_group_validator(self):
        assert isinstance(self.validator, GroupValidator)
        assert not isinstance(self.validator, GroupOperation)

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
        assert self.validator.validate(data) == expected


class TestContiguousTimerange:
    validator = RequireContiguousTimerange(group_by=["variable_id"])

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
        assert self.validator.validate(data) == expected


class TestOverlappingTimerange:
    validator = RequireOverlappingTimerange(group_by=["variable_id"])

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
        assert self.validator.validate(data) == expected


class TestSelectParentExperiment:
    def test_is_group_validator(self):
        validator = SelectParentExperiment()

        assert isinstance(validator, GroupOperation)
        assert not isinstance(validator, GroupValidator)


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
