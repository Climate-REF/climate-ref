from unittest import mock

import pandas as pd
import pytest
from cmip_ref_metrics_example import provider

from cmip_ref.provider_registry import ProviderRegistry
from cmip_ref.solver import MetricExecution, MetricSolver, extract_covered_datasets, solve_metrics
from cmip_ref_core.constraints import RequireFacets, SelectParentExperiment
from cmip_ref_core.datasets import SourceDatasetType
from cmip_ref_core.metrics import DataRequirement, FacetFilter, MetricExecutionDefinition


@pytest.fixture
def solver(db_seeded) -> MetricSolver:
    registry = ProviderRegistry(providers=[provider])
    # Use a fixed set of providers for the test suite until we can pull from the DB
    with db_seeded.session.begin():
        metric_solver = MetricSolver.build_from_db(db_seeded)
    metric_solver.provider_registry = registry

    return metric_solver


@pytest.fixture
def mock_metric_execution() -> MetricExecution:
    mock_execution = mock.MagicMock(spec=MetricExecution)
    mock_execution.provider = provider
    mock_execution.metric = provider.metrics()[0]

    mock_metric_dataset = mock.Mock(hash="123456", items=mock.Mock(return_value=[]))

    mock_execution.build_metric_execution_info.return_value = MetricExecutionDefinition(
        key="key",
        metric_dataset=mock_metric_dataset,
        output_fragment="output_fragment",
    )
    return mock_execution


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
@mock.patch.object(MetricSolver, "build_from_db")
def test_solve_metrics_default_solver(
    mock_build_solver, mock_executor, mock_metric_execution, db_seeded, solver
):
    # Create a mock solver that "solves" to create a single execution
    solver = mock.MagicMock(spec=MetricSolver)
    solver.solve.return_value = [mock_metric_execution]
    mock_build_solver.return_value = solver

    # Run with no solver specified
    with db_seeded.session.begin():
        solve_metrics(db_seeded)

    # Solver should be created
    assert mock_build_solver.call_count == 1
    # A single run would have been run
    assert mock_executor.return_value.run_metric.call_count == 1
    mock_executor.return_value.run_metric.assert_called_with(
        metric=mock_metric_execution.metric, definition=mock_metric_execution.build_metric_execution_info()
    )


@mock.patch("cmip_ref.solver.get_executor")
@mock.patch.object(MetricSolver, "build_from_db")
def test_solve_metrics(mock_build_solver, mock_executor, db_seeded, solver, data_regression):
    with db_seeded.session.begin():
        solve_metrics(db_seeded, dry_run=False, solver=solver)

    assert mock_build_solver.call_count == 0

    definitions = [call.kwargs["definition"] for call in mock_executor.return_value.run_metric.mock_calls]

    # Create a dictionary of the metric key and the source datasets that were used
    output = {}
    for definition in definitions:
        output[definition.key] = {
            str(source_type): ds_collection.instance_id.unique().tolist()
            for source_type, ds_collection in definition.metric_dataset.items()
        }

    # Write to a file for regression testing
    data_regression.check(output)


@mock.patch("cmip_ref.solver.get_executor")
def test_solve_metrics_dry_run(mock_executor, db_seeded, solver):
    solve_metrics(db_seeded, dry_run=True, solver=solver)

    assert mock_executor.return_value.run_metric.call_count == 0

    # TODO: Check that no new metrics were added to the db
