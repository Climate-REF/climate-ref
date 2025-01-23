from unittest import mock

import pandas as pd
import pytest
from cmip_ref_metrics_example import provider

from cmip_ref.provider_registry import ProviderRegistry
from cmip_ref.solver import MetricSolver, extract_covered_datasets, solve_metrics
from cmip_ref_core.constraints import RequireFacets, SelectParentExperiment
from cmip_ref_core.datasets import SourceDatasetType
from cmip_ref_core.metrics import DataRequirement, FacetFilter


@pytest.fixture
def solver(db_seeded) -> MetricSolver:
    registry = ProviderRegistry(providers=[provider])
    # Use a fixed set of providers for the test suite until we can pull from the DB
    with db_seeded.session.begin():
        metric_solver = MetricSolver.build_from_db(db_seeded)
    metric_solver.provider_registry = registry

    return metric_solver


class TestMetricSolver:
    def test_solver_build_from_db(self, solver):
        assert isinstance(solver, MetricSolver)
        assert isinstance(solver.provider_registry, ProviderRegistry)
        assert SourceDatasetType.CMIP6 in solver.data_catalog
        assert isinstance(solver.data_catalog[SourceDatasetType.CMIP6], pd.DataFrame)
        assert len(solver.data_catalog[SourceDatasetType.CMIP6])


@pytest.mark.parametrize(
    "requirement,data_catalog,expected",
    [
        pytest.param(
            DataRequirement(
                source_type=SourceDatasetType.CMIP6,
                filters=(FacetFilter(facets={"variable_id": "missing"}),),
                group_by=("variable_id", "experiment_id"),
            ),
            pd.DataFrame(
                {
                    "variable_id": ["tas", "tas", "pr"],
                    "experiment_id": ["ssp119", "ssp126", "ssp119"],
                    "variant_label": ["r1i1p1f1", "r1i1p1f1", "r1i1p1f1"],
                }
            ),
            [],
            id="empty",
        ),
        pytest.param(
            DataRequirement(
                source_type=SourceDatasetType.CMIP6,
                filters=(FacetFilter(facets={"variable_id": "tas"}),),
                group_by=("variable_id", "experiment_id"),
            ),
            pd.DataFrame(
                {
                    "variable_id": ["tas", "tas", "pr"],
                    "experiment_id": ["ssp119", "ssp126", "ssp119"],
                    "variant_label": ["r1i1p1f1", "r1i1p1f1", "r1i1p1f1"],
                }
            ),
            [
                pd.DataFrame(
                    {
                        "variable_id": ["tas"],
                        "experiment_id": ["ssp119"],
                        "variant_label": ["r1i1p1f1"],
                    },
                    index=[0],
                ),
                pd.DataFrame(
                    {
                        "variable_id": ["tas"],
                        "experiment_id": ["ssp126"],
                        "variant_label": ["r1i1p1f1"],
                    },
                    index=[1],
                ),
            ],
            id="simple-filter",
        ),
        pytest.param(
            DataRequirement(
                source_type=SourceDatasetType.CMIP6,
                filters=(FacetFilter(facets={"variable_id": ("tas", "pr")}),),
                group_by=("experiment_id",),
            ),
            pd.DataFrame(
                {
                    "variable_id": ["tas", "tas", "pr"],
                    "experiment_id": ["ssp119", "ssp126", "ssp119"],
                }
            ),
            [
                pd.DataFrame(
                    {
                        "variable_id": ["tas", "pr"],
                        "experiment_id": ["ssp119", "ssp119"],
                    },
                    index=[0, 2],
                ),
                pd.DataFrame(
                    {
                        "variable_id": ["tas"],
                        "experiment_id": ["ssp126"],
                    },
                    index=[1],
                ),
            ],
            id="simple-or",
        ),
        pytest.param(
            DataRequirement(
                source_type=SourceDatasetType.CMIP6,
                filters=(FacetFilter(facets={"variable_id": ("tas", "pr")}),),
                constraints=(SelectParentExperiment(),),
                group_by=("variable_id", "experiment_id"),
            ),
            pd.DataFrame(
                {
                    "variable_id": ["tas", "tas"],
                    "experiment_id": ["ssp119", "historical"],
                    "parent_experiment_id": ["historical", "none"],
                }
            ),
            [
                pd.DataFrame(
                    {
                        "variable_id": ["tas", "tas"],
                        "experiment_id": ["historical", "ssp119"],
                    },
                    # The order of the rows is not guaranteed
                    index=[1, 0],
                ),
            ],
            marks=[pytest.mark.xfail(reason="Parent experiment not implemented")],
            id="parent",
        ),
        pytest.param(
            DataRequirement(
                source_type=SourceDatasetType.CMIP6,
                filters=(FacetFilter(facets={"variable_id": ("tas", "pr")}),),
                constraints=(RequireFacets(dimension="variable_id", required_facets=["tas", "pr"]),),
                group_by=("experiment_id",),
            ),
            pd.DataFrame(
                {
                    "variable_id": ["tas", "tas", "pr"],
                    "experiment_id": ["ssp119", "ssp126", "ssp119"],
                }
            ),
            [
                pd.DataFrame(
                    {
                        "variable_id": ["tas", "pr"],
                        "experiment_id": ["ssp119", "ssp119"],
                    },
                    index=[0, 2],
                ),
            ],
            id="simple-validation",
        ),
    ],
)
def test_data_coverage(requirement, data_catalog, expected):
    result = extract_covered_datasets(data_catalog, requirement)

    for res, exp in zip(result, expected):
        pd.testing.assert_frame_equal(res, exp)
    assert len(result) == len(expected)


@mock.patch("cmip_ref.solver.get_executor")
def test_solve_metrics_default_solver(mock_executor, db_seeded, solver):
    with db_seeded.session.begin():
        solve_metrics(db_seeded)

    # One of the datasets is incomplete due to slicing along the time axis
    # in the sample data.
    assert mock_executor.return_value.run_metric.call_count == 5


@mock.patch("cmip_ref.solver.get_executor")
def test_solve_metrics(mock_executor, db_seeded, solver):
    with db_seeded.session.begin():
        solve_metrics(db_seeded, dry_run=False, solver=solver)

    definitions = [call.kwargs["definition"] for call in mock_executor.return_value.run_metric.mock_calls]

    expected_instance_ids = [
        ["CMIP6.CMIP.CSIRO.ACCESS-ESM1-5.historical.r1i1p1f1.Amon.tas.gn.v20191115"],
        ["CMIP6.CMIP.CSIRO.ACCESS-ESM1-5.historical.r1i1p1f1.Amon.rsut.gn.v20191115"],
        ["CMIP6.ScenarioMIP.CSIRO.ACCESS-ESM1-5.ssp126.r1i1p1f1.Amon.tas.gn.v20210318"],
        ["CMIP6.ScenarioMIP.CSIRO.ACCESS-ESM1-5.ssp126.r1i1p1f1.Amon.rsut.gn.v20210318"],
    ]
    expected_keys = [
        "dataset1_ACCESS-ESM1-5_rsut_historical_r1i1p1f1",
        "dataset1_ACCESS-ESM1-5_tas_historical_r1i1p1f1",
        "dataset1_ACCESS-ESM1-5_rsut_ssp126_r1i1p1f1",
        "dataset1_ACCESS-ESM1-5_tas_ssp126_r1i1p1f1",
    ]

    assert mock_executor.return_value.run_metric.call_count == len(expected_instance_ids)

    for definition in definitions:
        assert definition.metric_dataset["cmip6"].instance_id.unique().tolist() in expected_instance_ids
        assert definition.key in expected_keys


@mock.patch("cmip_ref.solver.get_executor")
def test_solve_metrics_dry_run(mock_executor, db_seeded, solver):
    solve_metrics(db_seeded, dry_run=True, solver=solver)

    assert mock_executor.return_value.run_metric.call_count == 0

    # TODO: Check that no new metrics were added to the db
