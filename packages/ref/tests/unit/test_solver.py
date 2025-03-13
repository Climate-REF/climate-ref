from unittest import mock

import pandas as pd
import pytest
from cmip_ref_metrics_example import provider

from cmip_ref.config import ExecutorConfig
from cmip_ref.models import MetricExecutionResult
from cmip_ref.provider_registry import ProviderRegistry
from cmip_ref.solver import MetricExecution, MetricSolver, extract_covered_datasets, solve_metrics
from cmip_ref_core.constraints import RequireFacets, SelectParentExperiment
from cmip_ref_core.datasets import SourceDatasetType
from cmip_ref_core.metrics import DataRequirement, FacetFilter
from cmip_ref_core.providers import MetricsProvider


@pytest.fixture
def solver(db_seeded, config) -> MetricSolver:
    registry = ProviderRegistry(providers=[provider])
    # Use a fixed set of providers for the test suite until we can pull from the DB
    with db_seeded.session.begin():
        metric_solver = MetricSolver.build_from_db(config, db_seeded)
    metric_solver.provider_registry = registry

    return metric_solver


@pytest.fixture
def mock_metric_execution(tmp_path, definition_factory) -> MetricExecution:
    mock_execution = mock.MagicMock(spec=MetricExecution)
    mock_execution.provider = provider
    mock_execution.metric = provider.metrics()[0]

    mock_metric_dataset = mock.Mock(hash="123456", items=mock.Mock(return_value=[]))

    mock_execution.build_metric_execution_info.return_value = definition_factory(
        metric_dataset=mock_metric_dataset
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
        pytest.param(
            DataRequirement(
                source_type=SourceDatasetType.obs4MIPs,
                filters=(FacetFilter(facets={"variable_id": "tas"}),),
                group_by=("variable_id", "source_id"),
            ),
            pd.DataFrame(
                {
                    "variable_id": ["tas", "tas", "pr"],
                    "source_id": ["ERA-5", "AIRX3STM-006", "GPCPMON-3-1"],
                    "frequency": ["mon", "mon", "mon"],
                }
            ),
            [
                pd.DataFrame(
                    {
                        "variable_id": ["tas"],
                        "source_id": ["AIRX3STM-006"],
                        "frequency": ["mon"],
                    },
                    index=[1],
                ),
                pd.DataFrame(
                    {
                        "variable_id": ["tas"],
                        "source_id": ["ERA-5"],
                        "frequency": ["mon"],
                    },
                    index=[0],
                ),
            ],
            id="simple-obs4MIPs",
        ),
    ],
)
def test_data_coverage(requirement, data_catalog, expected):
    result = extract_covered_datasets(data_catalog, requirement)

    for res, exp in zip(result, expected):
        pd.testing.assert_frame_equal(res, exp)
    assert len(result) == len(expected)


def test_solve_metrics_default_solver(mocker, mock_metric_execution, db_seeded, solver):
    mock_executor = mocker.patch.object(ExecutorConfig, "build")
    mock_build_solver = mocker.patch.object(MetricSolver, "build_from_db")

    # Create a mock solver that "solves" to create a single execution
    solver = mock.MagicMock(spec=MetricSolver)
    solver.solve.return_value = [mock_metric_execution]
    mock_build_solver.return_value = solver

    # Run with no solver specified
    with db_seeded.session.begin():
        solve_metrics(db_seeded)

    # Check that a result is created
    assert db_seeded.session.query(MetricExecutionResult).count() == 1
    execution_result = db_seeded.session.query(MetricExecutionResult).first()
    assert execution_result.output_fragment == "output_fragment"
    assert execution_result.dataset_hash == "123456"
    assert execution_result.metric_execution_group.dataset_key == "key"

    # Solver should be created
    assert mock_build_solver.call_count == 1
    # A single run would have been run
    assert mock_executor.return_value.run_metric.call_count == 1
    mock_executor.return_value.run_metric.assert_called_with(
        provider=mock_metric_execution.provider,
        metric=mock_metric_execution.metric,
        definition=mock_metric_execution.build_metric_execution_info(),
        metric_execution_result=execution_result,
    )


def test_solve_metrics(mocker, db_seeded, solver, data_regression):
    mock_executor = mocker.patch.object(ExecutorConfig, "build")
    mock_build_solver = mocker.patch.object(MetricSolver, "build_from_db")

    with db_seeded.session.begin():
        solve_metrics(db_seeded, dry_run=False, solver=solver)

    assert mock_build_solver.call_count == 0

    definitions = [call.kwargs["definition"] for call in mock_executor.return_value.run_metric.mock_calls]

    # Create a dictionary of the metric key and the source datasets that were used
    output = {}
    for definition in definitions:
        output[definition.dataset_key] = {
            str(source_type): ds_collection.instance_id.unique().tolist()
            for source_type, ds_collection in definition.metric_dataset.items()
        }

    # Write to a file for regression testing
    data_regression.check(output)


def test_solve_metrics_dry_run(mocker, db_seeded, config, solver):
    mock_executor = mocker.patch.object(ExecutorConfig, "build")

    solve_metrics(config=config, db=db_seeded, dry_run=True, solver=solver)

    assert mock_executor.return_value.run_metric.call_count == 0

    # TODO: Check that no new metrics were added to the db


@pytest.mark.parametrize("variable,expected", [("tas", 4), ("pr", 1), ("not_a_variable", 0)])
def test_solve_metric_executions(solver, mock_metric, variable, expected):
    metric = mock_metric
    metric.data_requirements = (
        DataRequirement(
            source_type=SourceDatasetType.obs4MIPs,
            filters=(FacetFilter(facets={"variable_id": variable}),),
            group_by=("variable_id", "source_id"),
        ),
        DataRequirement(
            source_type=SourceDatasetType.CMIP6,
            filters=(FacetFilter(facets={"variable_id": variable}),),
            group_by=("variable_id", "experiment_id"),
        ),
    )
    provider = MetricsProvider("mock_provider", "v0.1.0")
    provider.register(mock_metric)

    solver.data_catalog = {
        SourceDatasetType.obs4MIPs: pd.DataFrame(
            {
                "variable_id": ["tas", "tas", "pr"],
                "source_id": ["ERA-5", "AIRX3STM-006", "GPCPMON-3-1"],
                "frequency": ["mon", "mon", "mon"],
            }
        ),
        SourceDatasetType.CMIP6: pd.DataFrame(
            {
                "variable_id": ["tas", "tas", "pr"],
                "experiment_id": ["ssp119", "ssp126", "ssp119"],
                "variant_label": ["r1i1p1f1", "r1i1p1f1", "r1i1p1f1"],
            }
        ),
    }
    executions = solver.solve_metric_executions(metric, provider)
    assert len(list(executions)) == expected
