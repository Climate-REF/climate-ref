from copy import deepcopy
from typing import Any
from unittest import mock

import pandas as pd
import pytest
from climate_ref_esmvaltool import provider as esmvaltool_provider
from climate_ref_example import provider as example_provider
from climate_ref_ilamb import provider as ilamb_provider
from climate_ref_pmp import provider as pmp_provider

from climate_ref.config import ExecutorConfig
from climate_ref.data_catalog import DataCatalog
from climate_ref.database import Database
from climate_ref.datasets import CMIP6DatasetAdapter, Obs4MIPsDatasetAdapter, ingest_datasets
from climate_ref.models import Execution
from climate_ref.provider_registry import ProviderRegistry, _register_provider
from climate_ref.solver import (
    DiagnosticExecution,
    ExecutionSolver,
    SolveFilterOptions,
    extract_covered_datasets,
    matches_filter,
    solve_executions,
    solve_required_executions,
)
from climate_ref_core.constraints import AddParentDataset, AddSupplementaryDataset, RequireFacets
from climate_ref_core.datasets import SourceDatasetType
from climate_ref_core.diagnostics import DataRequirement, FacetFilter
from climate_ref_core.exceptions import InvalidDiagnosticException


@pytest.fixture
def solver(db_seeded, config) -> ExecutionSolver:
    registry = ProviderRegistry(providers=[example_provider])
    # Use a fixed set of providers for the test suite until we can pull from the DB
    metric_solver = ExecutionSolver.build_from_db(config, db_seeded)
    metric_solver.provider_registry = registry

    return metric_solver


@pytest.fixture
def aft_solver(db_seeded, config) -> ExecutionSolver:
    registry = ProviderRegistry(providers=[pmp_provider, esmvaltool_provider, ilamb_provider])
    metric_solver = ExecutionSolver.build_from_db(config, db_seeded)
    metric_solver.provider_registry = registry

    return metric_solver


@pytest.fixture
def mock_metric_execution(
    tmp_path, db_seeded, definition_factory, mock_diagnostic, provider
) -> DiagnosticExecution:
    with db_seeded.session.begin():
        _register_provider(db_seeded, provider)

    mock_execution = mock.MagicMock(spec=DiagnosticExecution)
    mock_execution.provider = provider
    mock_execution.diagnostic = provider.diagnostics()[0]
    mock_execution.selectors = {"cmip6": (("source_id", "Test"),)}

    mock_dataset_collection = mock.Mock(hash="123456", items=mock.Mock(return_value=[]))

    mock_execution.build_execution_definition.return_value = definition_factory(
        diagnostic=mock_diagnostic, execution_dataset_collection=mock_dataset_collection
    )
    return mock_execution


@pytest.fixture
def mock_executor(mocker):
    return mocker.patch.object(ExecutorConfig, "build")


class TestMetricSolver:
    def test_solver_build_from_db(self, solver):
        assert isinstance(solver, ExecutionSolver)
        assert isinstance(solver.provider_registry, ProviderRegistry)
        assert SourceDatasetType.CMIP6 in solver.data_catalog
        assert isinstance(solver.data_catalog[SourceDatasetType.CMIP6], DataCatalog)
        assert len(solver.data_catalog[SourceDatasetType.CMIP6].to_frame())


class TestExtractCoveredDatasetsWithDataCatalog:
    """Test that extract_covered_datasets triggers finalisation when given a DataCatalog."""

    def test_finalise_called_for_unfinalised_groups(self):
        """When a DataCatalog has unfinalised data, finalise() is called per group."""
        catalog_df = pd.DataFrame(
            {
                "variable_id": ["tas", "tas"],
                "experiment_id": ["historical", "historical"],
                "source_id": ["MODEL-A", "MODEL-A"],
                "finalised": [False, False],
            }
        )
        mock_catalog = mock.MagicMock(spec=DataCatalog)
        mock_catalog.to_frame.return_value = catalog_df
        mock_catalog.finalise.side_effect = lambda group: group.assign(finalised=True)

        requirement = DataRequirement(
            source_type=SourceDatasetType.CMIP6,
            filters=(),
            group_by=("source_id",),
        )

        result = extract_covered_datasets(mock_catalog, requirement)

        mock_catalog.finalise.assert_called()
        assert len(result) == 1
        # The finalised group should have finalised=True
        group_df = next(iter(result.values()))
        assert group_df["finalised"].all()

    def test_finalise_not_called_for_raw_dataframe(self):
        """When a raw DataFrame is passed, no finalisation is attempted."""
        catalog_df = pd.DataFrame(
            {
                "variable_id": ["tas"],
                "experiment_id": ["historical"],
                "source_id": ["MODEL-A"],
                "finalised": [False],
            }
        )

        requirement = DataRequirement(
            source_type=SourceDatasetType.CMIP6,
            filters=(),
            group_by=None,
        )

        # Should work without error - no finalise call attempted on a raw DataFrame
        result = extract_covered_datasets(catalog_df, requirement)
        assert len(result) == 1


@pytest.mark.parametrize(
    "requirement,data_catalog,expected",
    [
        pytest.param(
            DataRequirement(
                source_type=SourceDatasetType.CMIP6,
                filters=(),
                group_by=None,
            ),
            pd.DataFrame(
                {
                    "variable_id": ["tas", "tas", "pr"],
                    "experiment_id": ["ssp119", "ssp126", "ssp119"],
                    "variant_label": ["r1i1p1f1", "r1i1p1f1", "r1i1p1f1"],
                }
            ),
            {
                (): pd.DataFrame(
                    {
                        "variable_id": ["tas", "tas", "pr"],
                        "experiment_id": ["ssp119", "ssp126", "ssp119"],
                        "variant_label": ["r1i1p1f1", "r1i1p1f1", "r1i1p1f1"],
                    }
                )
            },
            id="group-by-none",
        ),
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
            {},
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
            {
                (("variable_id", "tas"), ("experiment_id", "ssp119")): pd.DataFrame(
                    {
                        "variable_id": ["tas"],
                        "experiment_id": ["ssp119"],
                        "variant_label": ["r1i1p1f1"],
                    },
                    index=[0],
                ),
                (("variable_id", "tas"), ("experiment_id", "ssp126")): pd.DataFrame(
                    {
                        "variable_id": ["tas"],
                        "experiment_id": ["ssp126"],
                        "variant_label": ["r1i1p1f1"],
                    },
                    index=[1],
                ),
            },
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
            {
                (("experiment_id", "ssp119"),): pd.DataFrame(
                    {
                        "variable_id": ["tas", "pr"],
                        "experiment_id": ["ssp119", "ssp119"],
                    },
                    index=[0, 2],
                ),
                (("experiment_id", "ssp126"),): pd.DataFrame(
                    {
                        "variable_id": ["tas"],
                        "experiment_id": ["ssp126"],
                    },
                    index=[1],
                ),
            },
            id="simple-or",
        ),
        pytest.param(
            DataRequirement(
                source_type=SourceDatasetType.CMIP6,
                filters=(FacetFilter(facets={"variable_id": ("tas", "pr")}),),
                constraints=(AddParentDataset.from_defaults(SourceDatasetType.CMIP6),),
                group_by=("variable_id", "experiment_id"),
            ),
            pd.DataFrame(
                {
                    "experiment_id": ["ssp119", "historical"],
                    "grid_label": ["gn", "gn"],
                    "parent_experiment_id": ["historical", "none"],
                    "parent_source_id": ["A", "A"],
                    "parent_variant_label": ["r1i1p1f1", "none"],
                    "source_id": ["A", "A"],
                    "table_id": ["Amon", "Amon"],
                    "variable_id": ["tas", "tas"],
                    "variant_label": ["r1i1p1f1", "r1i1p1f1"],
                    "version": ["v20210101", "v20220101"],
                }
            ),
            {
                (("variable_id", "tas"), ("experiment_id", "ssp119")): pd.DataFrame(
                    {
                        "experiment_id": ["ssp119", "historical"],
                        "grid_label": ["gn", "gn"],
                        "parent_experiment_id": ["historical", "none"],
                        "parent_source_id": ["A", "A"],
                        "parent_variant_label": ["r1i1p1f1", "none"],
                        "source_id": ["A", "A"],
                        "table_id": ["Amon", "Amon"],
                        "variable_id": ["tas", "tas"],
                        "variant_label": ["r1i1p1f1", "r1i1p1f1"],
                        "version": ["v20210101", "v20220101"],
                    },
                    index=[0, 1],
                ),
            },
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
            {
                (("experiment_id", "ssp119"),): pd.DataFrame(
                    {
                        "variable_id": ["tas", "pr"],
                        "experiment_id": ["ssp119", "ssp119"],
                    },
                    index=[0, 2],
                ),
            },
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
            {
                (("variable_id", "tas"), ("source_id", "AIRX3STM-006")): pd.DataFrame(
                    {
                        "variable_id": ["tas"],
                        "source_id": ["AIRX3STM-006"],
                        "frequency": ["mon"],
                    },
                    index=[1],
                ),
                (("variable_id", "tas"), ("source_id", "ERA-5")): pd.DataFrame(
                    {
                        "variable_id": ["tas"],
                        "source_id": ["ERA-5"],
                        "frequency": ["mon"],
                    },
                    index=[0],
                ),
            },
            id="simple-obs4MIPs",
        ),
    ],
)
def test_data_coverage(requirement, data_catalog, expected):
    def add_path(df: pd.DataFrame) -> pd.DataFrame:
        """Insert a path column into the DataFrame."""
        df["path"] = df.apply(lambda r: "_".join(map(str, r.tolist())) + ".nc", axis=1)

    add_path(data_catalog)
    for expected_value in expected.values():
        add_path(expected_value)
    result = extract_covered_datasets(data_catalog, requirement)

    for key, expected_value in expected.items():
        pd.testing.assert_frame_equal(result[key], expected_value)
    assert len(result) == len(expected)


def test_extract_no_groups():
    requirement = DataRequirement(
        source_type=SourceDatasetType.CMIP6,
        filters=(),
        group_by=(),
    )
    data_catalog = pd.DataFrame(
        {
            "variable_id": ["tas", "tas", "pr"],
        }
    )

    with pytest.raises(ValueError, match="No group keys passed!"):
        extract_covered_datasets(data_catalog, requirement)


def test_solver_solve_with_filters(aft_solver):
    def solve_filtered(**kwargs):
        """Helper function to solve with filters and return a DataFrame of results."""
        return pd.DataFrame(
            [
                {
                    "diagnostic": execution.diagnostic.slug,
                    "provider": execution.provider.slug,
                    "dataset_key": execution.dataset_key,
                }
                for execution in aft_solver.solve(filters=SolveFilterOptions(**kwargs))
            ]
        )

    # Empty filters should return all executions
    executions = solve_filtered()
    assert not executions.empty
    executions = solve_filtered(provider=None, diagnostic=None)
    assert not executions.empty
    executions = solve_filtered(provider=[], diagnostic=[])
    assert not executions.empty

    # ILAMB filter should only return ILAMB executions
    executions = solve_filtered(provider=["ilamb"])
    assert executions["provider"].unique().tolist() == ["ilamb"]
    assert executions["diagnostic"].nunique() > 1

    # Multiple provider filters
    executions = solve_filtered(provider=["ilamb", "pmp"])
    assert sorted(executions["provider"].unique().tolist()) == ["ilamb", "pmp"]

    # Partial diagnostic filter should return executions for that diagnostic
    # enso metrics exist in both pmp and esmvaltool providers
    executions = solve_filtered(diagnostic=["enso"])
    assert sorted(executions["provider"].unique().tolist()) == ["esmvaltool", "pmp"]

    # Adding in a provider filter as well should limit the results to that provider
    executions = solve_filtered(provider=["pmp"], diagnostic=["enso"])
    assert executions["provider"].unique().tolist() == ["pmp"]
    assert sorted(executions["diagnostic"].unique().tolist()) == ["enso_proc", "enso_tel"]

    # Check lowercase
    pd.testing.assert_frame_equal(executions, solve_filtered(provider=["PmP"], diagnostic=["enSo"]))

    # Missing provider should return no results
    assert not list(
        aft_solver.solve(
            filters=SolveFilterOptions(
                provider=["missing"],
            )
        )
    )

    # Missing diagnostic should return no results
    assert not list(
        aft_solver.solve(
            filters=SolveFilterOptions(
                diagnostic=["missing"],
            )
        )
    )


def test_solve_metrics_default_solver(mocker, mock_metric_execution, mock_executor, db_seeded, solver):
    mock_build_solver = mocker.patch.object(ExecutionSolver, "build_from_db")

    # Create a mock solver that "solves" to create a single execution
    solver = mock.MagicMock(spec=ExecutionSolver)
    solver.solve.return_value = [mock_metric_execution]
    mock_build_solver.return_value = solver

    # Run with no solver specified
    solve_required_executions(db_seeded)

    # Check that a result is created
    assert db_seeded.session.query(Execution).count() == 1
    execution_result = db_seeded.session.query(Execution).first()
    assert execution_result.output_fragment == "output_fragment"
    assert execution_result.dataset_hash == "123456"
    assert execution_result.execution_group.key == "key"
    # Nested tuples are converted into nested lists after going through the DB
    assert execution_result.execution_group.selectors == {
        "cmip6": [
            ["source_id", "Test"],
        ]
    }

    # Solver should be created
    assert mock_build_solver.call_count == 1
    # A single run would have been run
    assert mock_executor.return_value.run.call_count == 1
    mock_executor.return_value.run.assert_called_with(
        definition=mock_metric_execution.build_execution_definition(),
        execution=execution_result,
    )


def test_solve_metrics(mocker, db_seeded, solver, data_regression, mock_executor):
    mock_build_solver = mocker.patch.object(ExecutionSolver, "build_from_db")

    solve_required_executions(db_seeded, dry_run=False, solver=solver)

    assert mock_build_solver.call_count == 0

    definitions = [call.kwargs["definition"] for call in mock_executor.return_value.run.mock_calls]

    # Create a dictionary of the diagnostic key and the source datasets that were used
    output = {}
    for definition in definitions:
        output[definition.key] = {
            str(source_type): ds_collection.instance_id.unique().tolist()
            for source_type, ds_collection in definition.datasets.items()
        }

    # Write to a file for regression testing
    data_regression.check(output)


def test_solve_metrics_dry_run(db_seeded, config, solver, mock_executor):
    solve_required_executions(config=config, db=db_seeded, dry_run=True, solver=solver)

    assert mock_executor.return_value.run.call_count == 0


def test_solve_metric_executions_missing(mock_diagnostic, provider):
    mock_diagnostic.data_requirements = ()
    with pytest.raises(ValueError, match=f"Diagnostic {mock_diagnostic.slug!r} has no data requirements"):
        next(solve_executions({}, mock_diagnostic, provider))


def test_solve_metric_executions_mixed_data_requirements(mock_diagnostic, provider):
    mock_diagnostic.data_requirements = (
        DataRequirement(
            source_type=SourceDatasetType.CMIP6,
            filters=(),
            group_by=("variable_id", "source_id"),
        ),
        (
            DataRequirement(
                source_type=SourceDatasetType.CMIP6,
                filters=(),
                group_by=("variable_id", "experiment_id"),
            ),
        ),
    )
    data_catalog = {SourceDatasetType.CMIP6: pd.DataFrame()}

    with pytest.raises(TypeError, match="Expected a DataRequirement, got <class 'tuple'>"):
        next(solve_executions(data_catalog, mock_diagnostic, provider))

    mock_diagnostic.data_requirements = mock_diagnostic.data_requirements[::-1]
    with pytest.raises(
        TypeError,
        match=r"Expected a sequence of DataRequirement,"
        r" got <class 'climate_ref_core.diagnostics.DataRequirement'>",
    ):
        next(solve_executions(data_catalog, mock_diagnostic, provider))

    mock_diagnostic.data_requirements = ("test",)
    with pytest.raises(TypeError, match="Expected a DataRequirement, got <class 'str'>"):
        next(solve_executions(data_catalog, mock_diagnostic, provider))

    mock_diagnostic.data_requirements = (None,)
    with pytest.raises(TypeError, match="Expected a DataRequirement, got <class 'NoneType'>"):
        next(solve_executions(data_catalog, mock_diagnostic, provider))


@pytest.mark.parametrize("variable,expected", [("tas", 4), ("pr", 1), ("not_a_variable", 0)])
def test_solve_metric_executions(solver, mock_diagnostic, provider, variable, expected):
    metric = mock_diagnostic
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

    data_catalog = {
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
    executions = solve_executions(data_catalog, metric, provider)
    assert len(list(executions)) == expected


def test_solve_metric_executions_multiple_sets(solver, mock_diagnostic, provider):
    metric = mock_diagnostic
    metric.data_requirements = (
        (
            DataRequirement(
                source_type=SourceDatasetType.CMIP6,
                filters=(FacetFilter(facets={"variable_id": "tas"}),),
                group_by=("variable_id",),
            ),
        ),
        (
            DataRequirement(
                source_type=SourceDatasetType.CMIP6,
                filters=(FacetFilter(facets={"variable_id": "pr"}),),
                group_by=("variable_id", "experiment_id"),
            ),
        ),
    )

    data_catalog = {
        SourceDatasetType.CMIP6: pd.DataFrame(
            {
                "variable_id": ["tas", "tas", "pr"],
                "experiment_id": ["ssp119", "ssp126", "ssp119"],
                "variant_label": ["r1i1p1f1", "r1i1p1f1", "r1i1p1f1"],
            }
        ),
    }
    executions = list(solve_executions(data_catalog, metric, provider))
    assert len(executions) == 2

    assert executions[0].datasets[SourceDatasetType.CMIP6].selector == (("variable_id", "tas"),)

    assert executions[1].datasets[SourceDatasetType.CMIP6].selector == (
        ("experiment_id", "ssp119"),
        ("variable_id", "pr"),
    )


def test_solve_metric_executions_or_logic_missing_source_type(mock_diagnostic, provider):
    """Test OR logic when one requirement set has a missing source type."""
    metric = mock_diagnostic
    # First set requires CMIP7 (not available), second set requires CMIP6 (available)
    metric.data_requirements = (
        (
            DataRequirement(
                source_type=SourceDatasetType.CMIP7,
                filters=(FacetFilter(facets={"variable_id": "tas"}),),
                group_by=("variable_id",),
            ),
        ),
        (
            DataRequirement(
                source_type=SourceDatasetType.CMIP6,
                filters=(FacetFilter(facets={"variable_id": "tas"}),),
                group_by=("variable_id",),
            ),
        ),
    )

    # Only CMIP6 data available
    data_catalog = {
        SourceDatasetType.CMIP6: pd.DataFrame(
            {
                "variable_id": ["tas"],
                "experiment_id": ["historical"],
                "variant_label": ["r1i1p1f1"],
            }
        ),
    }
    executions = list(solve_executions(data_catalog, metric, provider))

    # Should fall back to CMIP6 requirement
    assert len(executions) == 1
    assert SourceDatasetType.CMIP6 in executions[0].datasets


def test_solve_metric_executions_or_logic_first_matches(mock_diagnostic, provider):
    """Test OR logic when first requirement set matches."""
    metric = mock_diagnostic
    # First set requires CMIP6 (available), second set also requires CMIP6
    metric.data_requirements = (
        (
            DataRequirement(
                source_type=SourceDatasetType.CMIP6,
                filters=(FacetFilter(facets={"variable_id": "tas"}),),
                group_by=("variable_id",),
            ),
        ),
        (
            DataRequirement(
                source_type=SourceDatasetType.CMIP6,
                filters=(FacetFilter(facets={"variable_id": "pr"}),),
                group_by=("variable_id",),
            ),
        ),
    )

    data_catalog = {
        SourceDatasetType.CMIP6: pd.DataFrame(
            {
                "variable_id": ["tas", "pr"],
                "experiment_id": ["historical", "historical"],
                "variant_label": ["r1i1p1f1", "r1i1p1f1"],
            }
        ),
    }
    executions = list(solve_executions(data_catalog, metric, provider))

    # Both requirement sets should produce executions
    assert len(executions) == 2


def test_solve_metric_executions_or_logic_no_matches(mock_diagnostic, provider):
    """Test OR logic when no requirement sets match."""
    metric = mock_diagnostic
    # Both sets require source types that are not available
    metric.data_requirements = (
        (
            DataRequirement(
                source_type=SourceDatasetType.CMIP7,
                filters=(FacetFilter(facets={"variable_id": "tas"}),),
                group_by=("variable_id",),
            ),
        ),
        (
            DataRequirement(
                source_type=SourceDatasetType.CMIP7,
                filters=(FacetFilter(facets={"variable_id": "pr"}),),
                group_by=("variable_id",),
            ),
        ),
    )

    # No matching data
    data_catalog = {
        SourceDatasetType.CMIP6: pd.DataFrame(
            {
                "variable_id": ["tas"],
                "experiment_id": ["historical"],
                "variant_label": ["r1i1p1f1"],
            }
        ),
    }

    with pytest.raises(InvalidDiagnosticException, match="No data catalog matches"):
        list(solve_executions(data_catalog, metric, provider))


def test_solve_metric_executions_or_logic_with_cmip7_available(mock_diagnostic, provider):
    """Test OR logic when CMIP7 data is available."""
    metric = mock_diagnostic
    # First set requires CMIP7 (available), second set requires CMIP6
    metric.data_requirements = (
        (
            DataRequirement(
                source_type=SourceDatasetType.CMIP7,
                filters=(FacetFilter(facets={"variable_id": "tas"}),),
                group_by=("variable_id",),
            ),
        ),
        (
            DataRequirement(
                source_type=SourceDatasetType.CMIP6,
                filters=(FacetFilter(facets={"variable_id": "tas"}),),
                group_by=("variable_id",),
            ),
        ),
    )

    # Both CMIP6 and CMIP7 data available
    data_catalog = {
        SourceDatasetType.CMIP6: pd.DataFrame(
            {
                "variable_id": ["tas"],
                "experiment_id": ["historical"],
                "variant_label": ["r1i1p1f1"],
            }
        ),
        SourceDatasetType.CMIP7: pd.DataFrame(
            {
                "variable_id": ["tas"],
                "experiment_id": ["historical"],
                "variant_label": ["r1i1p1f1"],
            }
        ),
    }
    executions = list(solve_executions(data_catalog, metric, provider))

    # Both sets should produce executions
    assert len(executions) == 2
    source_types = {next(iter(e.datasets.keys())) for e in executions}
    assert SourceDatasetType.CMIP6 in source_types
    assert SourceDatasetType.CMIP7 in source_types


def _prep_data_catalog(data_catalog: dict[str, Any]) -> pd.DataFrame:
    data_catalog_df = pd.DataFrame(data_catalog)
    data_catalog_df["instance_id"] = data_catalog_df.apply(
        lambda row: "CMIP6." + ".".join([row[item] for item in ["variable_id", "experiment_id"]]), axis=1
    )

    return data_catalog_df


def test_solve_with_new_datasets(obs4mips_data_catalog, mock_diagnostic, provider):
    expected_dataset_key = "cmip6_ACCESS-ESM1-5_tas"
    mock_diagnostic.data_requirements = (
        DataRequirement(
            source_type=SourceDatasetType.CMIP6,
            filters=(FacetFilter(facets={"variable_id": "tas"}),),
            group_by=("variable_id", "source_id"),
        ),
    )

    data_catalog = _prep_data_catalog(
        {
            "variable_id": ["tas", "pr"],
            "experiment_id": ["ssp119", "ssp119"],
            "source_id": "ACCESS-ESM1-5",
            "grid_label": "gn",
            "table_id": "AMon",
            "member_id": "r1i1pif1",
            "version": "v20210318",
        }
    )

    result_1 = next(
        solve_executions(
            {SourceDatasetType.CMIP6: data_catalog},
            mock_diagnostic,
            provider,
        )
    )
    assert result_1.dataset_key == expected_dataset_key

    data_catalog = _prep_data_catalog(
        {
            "variable_id": ["tas", "tas", "pr"],
            "experiment_id": ["ssp119", "ssp126", "ssp119"],
            "source_id": "ACCESS-ESM1-5",
            "grid_label": "gn",
            "table_id": "AMon",
            "member_id": "r1i1pif1",
            "version": "v20210318",
        }
    )

    result_2 = next(
        solve_executions(
            {SourceDatasetType.CMIP6: data_catalog},
            mock_diagnostic,
            provider,
        )
    )
    assert result_2.dataset_key == expected_dataset_key
    assert result_2.datasets.hash != result_1.datasets.hash


def test_solve_with_new_areacella(obs4mips_data_catalog, mock_diagnostic, provider):
    expected_dataset_key = "cmip6_ssp126_ACCESS-ESM1-5_tas__obs4mips_HadISST-1-1_ts"
    mock_diagnostic.data_requirements = (
        DataRequirement(
            source_type=SourceDatasetType.obs4MIPs,
            filters=(FacetFilter(facets={"variable_id": "ts", "source_id": "HadISST-1-1"}),),
            group_by=("variable_id", "source_id"),
        ),
        DataRequirement(
            source_type=SourceDatasetType.CMIP6,
            filters=(FacetFilter(facets={"variable_id": "tas", "experiment_id": "ssp126"}),),
            group_by=("variable_id", "experiment_id", "source_id"),
            constraints=(AddSupplementaryDataset.from_defaults("areacella", SourceDatasetType.CMIP6),),
        ),
    )

    cmip_data_catalog = _prep_data_catalog(
        {
            "variable_id": ["tas", "tas", "pr"],
            "experiment_id": ["ssp119", "ssp126", "ssp119"],
            "source_id": "ACCESS-ESM1-5",
            "grid_label": "gn",
            "table_id": "AMon",
            "member_id": "r1i1pif1",
            "version": "v20210318",
        }
    )

    result_1 = next(
        solve_executions(
            {
                SourceDatasetType.obs4MIPs: obs4mips_data_catalog,
                SourceDatasetType.CMIP6: cmip_data_catalog,
            },
            mock_diagnostic,
            provider,
        )
    )
    assert result_1.dataset_key == expected_dataset_key

    # areacella added
    # dataset key should remain the same
    cmip_data_catalog = _prep_data_catalog(
        {
            "variable_id": ["tas", "tas", "areacella", "pr"],
            "experiment_id": ["ssp119", "ssp126", "ssp126", "ssp119"],
            "source_id": "ACCESS-ESM1-5",
            "grid_label": "gn",
            "table_id": ["AMon", "AMon", "fx", "AMon"],
            "member_id": "r1i1pif1",
            "version": "v20210318",
        }
    )
    result_2 = next(
        solve_executions(
            {
                SourceDatasetType.obs4MIPs: obs4mips_data_catalog,
                SourceDatasetType.CMIP6: cmip_data_catalog,
            },
            mock_diagnostic,
            provider,
        )
    )
    assert result_2.dataset_key == expected_dataset_key
    assert result_2.datasets.hash != result_1.datasets.hash


def test_solve_with_one_per_provider(
    db_seeded, mock_metric_execution, mock_diagnostic, caplog, mock_executor
):
    mock_execution_2 = deepcopy(mock_metric_execution)
    mock_execution_2.diagnostic = mock_diagnostic.provider.get("failed")

    # Create a mock solver that "solves" to create multiple executions with the same provider,
    # but different diagnostics
    solver = mock.MagicMock(spec=ExecutionSolver)
    solver.solve.return_value = [mock_metric_execution, mock_execution_2]
    with caplog.at_level("INFO"):
        solve_required_executions(db_seeded, solver=solver, one_per_provider=True)

    assert "Skipping execution due to one-of check" in caplog.text

    # Check that only one result is created
    assert db_seeded.session.query(Execution).count() == 1
    execution_result = db_seeded.session.query(Execution).first()

    # A single run would have been run
    assert mock_executor.return_value.run.call_count == 1
    mock_executor.return_value.run.assert_called_with(
        definition=mock_metric_execution.build_execution_definition(),
        execution=execution_result,
    )


def test_solve_with_one_per_diagnostic(
    db_seeded, mock_metric_execution, mock_diagnostic, caplog, mock_executor
):
    # Create a mock solver that "solves" to create multiple executions with the same diagnostic.
    # The second execution has the same dataset hash as the first, so the in-progress guard
    # in should_run will skip it (the first execution has successful=None).
    solver = mock.MagicMock(spec=ExecutionSolver)
    solver.solve.return_value = [mock_metric_execution, mock_metric_execution]
    with caplog.at_level("DEBUG"):
        solve_required_executions(db_seeded, solver=solver, one_per_diagnostic=True)

    assert "already has an in-progress execution" in caplog.text

    # Check that a result is created
    assert db_seeded.session.query(Execution).count() == 1
    execution_result = db_seeded.session.query(Execution).first()

    # A single run would have been run
    assert mock_executor.return_value.run.call_count == 1
    mock_executor.return_value.run.assert_called_with(
        definition=mock_metric_execution.build_execution_definition(),
        execution=execution_result,
    )


def test_solve_with_one_per_diagnostic_different_diagnostics(
    db_seeded, mock_metric_execution, mock_diagnostic, mock_executor
):
    mock_execution_2 = deepcopy(mock_metric_execution)
    mock_execution_2.diagnostic = mock_diagnostic.provider.get("failed")

    # Create a mock solver that "solves" to create multiple executions with the different diagnostics
    solver = mock.MagicMock(spec=ExecutionSolver)
    solver.solve.return_value = [mock_metric_execution, mock_execution_2]

    solve_required_executions(db_seeded, solver=solver, one_per_diagnostic=True)

    # Check that multiple diagnostics are created
    assert db_seeded.session.query(Execution).count() == 2


class TestMatchesFilter:
    """Tests for the matches_filter function."""

    def test_no_filters_returns_true(self, mock_diagnostic):
        assert matches_filter(mock_diagnostic, None) is True

    def test_empty_filters_returns_true(self, mock_diagnostic):
        assert matches_filter(mock_diagnostic, SolveFilterOptions()) is True

    def test_empty_lists_returns_true(self, mock_diagnostic):
        assert matches_filter(mock_diagnostic, SolveFilterOptions(provider=[], diagnostic=[])) is True

    def test_matching_provider(self, mock_diagnostic):
        assert matches_filter(mock_diagnostic, SolveFilterOptions(provider=["mock_provider"])) is True

    def test_non_matching_provider(self, mock_diagnostic):
        assert matches_filter(mock_diagnostic, SolveFilterOptions(provider=["nonexistent"])) is False

    def test_matching_diagnostic(self, mock_diagnostic):
        assert matches_filter(mock_diagnostic, SolveFilterOptions(diagnostic=["mock"])) is True

    def test_non_matching_diagnostic(self, mock_diagnostic):
        assert matches_filter(mock_diagnostic, SolveFilterOptions(diagnostic=["nonexistent"])) is False

    def test_partial_match_provider(self, mock_diagnostic):
        # "mock" is contained in "mock_provider"
        assert matches_filter(mock_diagnostic, SolveFilterOptions(provider=["mock"])) is True

    def test_partial_match_diagnostic(self, mock_diagnostic):
        # "ock" is contained in "mock"
        assert matches_filter(mock_diagnostic, SolveFilterOptions(diagnostic=["ock"])) is True

    def test_case_insensitive_provider(self, mock_diagnostic):
        assert matches_filter(mock_diagnostic, SolveFilterOptions(provider=["MOCK_PROVIDER"])) is True

    def test_case_insensitive_diagnostic(self, mock_diagnostic):
        assert matches_filter(mock_diagnostic, SolveFilterOptions(diagnostic=["MOCK"])) is True

    def test_provider_and_diagnostic_both_match(self, mock_diagnostic):
        assert (
            matches_filter(
                mock_diagnostic, SolveFilterOptions(provider=["mock_provider"], diagnostic=["mock"])
            )
            is True
        )

    def test_provider_matches_diagnostic_does_not(self, mock_diagnostic):
        assert (
            matches_filter(
                mock_diagnostic, SolveFilterOptions(provider=["mock_provider"], diagnostic=["nonexistent"])
            )
            is False
        )

    def test_provider_does_not_match_diagnostic_matches(self, mock_diagnostic):
        assert (
            matches_filter(mock_diagnostic, SolveFilterOptions(provider=["nonexistent"], diagnostic=["mock"]))
            is False
        )


def test_solve_metric_executions_empty_dataframe(mock_diagnostic, provider):
    """Test solve_executions when data catalog has an empty DataFrame for the source type."""
    mock_diagnostic.data_requirements = (
        DataRequirement(
            source_type=SourceDatasetType.CMIP6,
            filters=(FacetFilter(facets={"variable_id": "tas"}),),
            group_by=("variable_id",),
        ),
    )

    data_catalog = {
        SourceDatasetType.CMIP6: pd.DataFrame(
            columns=["variable_id", "experiment_id", "variant_label", "path"]
        ),
    }
    executions = list(solve_executions(data_catalog, mock_diagnostic, provider))
    assert len(executions) == 0


def test_solve_required_executions_no_execute(mocker, mock_metric_execution, mock_executor, db_seeded):
    """Test solve_required_executions with execute=False still creates DB records."""
    solver = mock.MagicMock(spec=ExecutionSolver)
    solver.solve.return_value = [mock_metric_execution]

    solve_required_executions(db_seeded, solver=solver, execute=False)

    # DB record should be created
    assert db_seeded.session.query(Execution).count() == 1
    # But the executor's run method should NOT be called
    assert mock_executor.return_value.run.call_count == 0


def test_diagnostic_execution_slug(mock_diagnostic, provider):
    """Test DiagnosticExecution.execution_slug method."""
    data_catalog = {
        SourceDatasetType.CMIP6: pd.DataFrame(
            {
                "variable_id": ["tas"],
                "experiment_id": ["historical"],
                "variant_label": ["r1i1p1f1"],
            }
        ),
    }
    mock_diagnostic.data_requirements = (
        DataRequirement(
            source_type=SourceDatasetType.CMIP6,
            filters=(FacetFilter(facets={"variable_id": "tas"}),),
            group_by=("variable_id",),
        ),
    )
    executions = list(solve_executions(data_catalog, mock_diagnostic, provider))
    assert len(executions) == 1

    slug = executions[0].execution_slug()
    assert provider.slug in slug
    assert mock_diagnostic.slug in slug
    assert "cmip6" in slug


def test_diagnostic_execution_build_definition(mock_diagnostic, provider, tmp_path):
    """Test DiagnosticExecution.build_execution_definition method."""
    data_catalog = {
        SourceDatasetType.CMIP6: pd.DataFrame(
            {
                "variable_id": ["tas"],
                "experiment_id": ["historical"],
                "variant_label": ["r1i1p1f1"],
                "instance_id": ["CMIP6.test.tas"],
            }
        ),
    }
    mock_diagnostic.data_requirements = (
        DataRequirement(
            source_type=SourceDatasetType.CMIP6,
            filters=(FacetFilter(facets={"variable_id": "tas"}),),
            group_by=("variable_id",),
        ),
    )
    executions = list(solve_executions(data_catalog, mock_diagnostic, provider))
    assert len(executions) == 1

    definition = executions[0].build_execution_definition(output_root=tmp_path)
    assert definition.key == executions[0].dataset_key
    assert provider.slug in str(definition.output_directory)
    assert mock_diagnostic.slug in str(definition.output_directory)
    # output_directory should be under the resolved tmp_path
    assert str(tmp_path.resolve()) in str(definition.output_directory)


def test_diagnostic_execution_selectors(mock_diagnostic, provider):
    """Test DiagnosticExecution.selectors property."""
    data_catalog = {
        SourceDatasetType.CMIP6: pd.DataFrame(
            {
                "variable_id": ["tas"],
                "experiment_id": ["historical"],
                "variant_label": ["r1i1p1f1"],
            }
        ),
    }
    mock_diagnostic.data_requirements = (
        DataRequirement(
            source_type=SourceDatasetType.CMIP6,
            filters=(FacetFilter(facets={"variable_id": "tas"}),),
            group_by=("variable_id",),
        ),
    )
    executions = list(solve_executions(data_catalog, mock_diagnostic, provider))
    assert len(executions) == 1

    selectors = executions[0].selectors
    assert isinstance(selectors, dict)
    assert "cmip6" in selectors


def test_diagnostic_execution_dataset_key_multiple_source_types(mock_diagnostic, provider):
    """Test dataset_key with multiple source types produces a stable key."""
    mock_diagnostic.data_requirements = (
        DataRequirement(
            source_type=SourceDatasetType.obs4MIPs,
            filters=(FacetFilter(facets={"variable_id": "tas"}),),
            group_by=("variable_id", "source_id"),
        ),
        DataRequirement(
            source_type=SourceDatasetType.CMIP6,
            filters=(FacetFilter(facets={"variable_id": "tas"}),),
            group_by=("variable_id", "experiment_id"),
        ),
    )

    data_catalog = {
        SourceDatasetType.obs4MIPs: pd.DataFrame(
            {
                "variable_id": ["tas"],
                "source_id": ["ERA-5"],
                "frequency": ["mon"],
            }
        ),
        SourceDatasetType.CMIP6: pd.DataFrame(
            {
                "variable_id": ["tas"],
                "experiment_id": ["historical"],
                "variant_label": ["r1i1p1f1"],
            }
        ),
    }
    executions = list(solve_executions(data_catalog, mock_diagnostic, provider))
    assert len(executions) == 1

    key = executions[0].dataset_key
    # Key should contain both source types
    assert "cmip6" in key
    assert "obs4mips" in key
    # Key should be joined by "__"
    assert "__" in key


def test_extract_covered_datasets_empty_catalog():
    """Test extract_covered_datasets with empty data catalog."""
    requirement = DataRequirement(
        source_type=SourceDatasetType.CMIP6,
        filters=(FacetFilter(facets={"variable_id": "tas"}),),
        group_by=("variable_id",),
    )
    data_catalog = pd.DataFrame(columns=["variable_id", "experiment_id", "path"])
    result = extract_covered_datasets(data_catalog, requirement)
    assert result == {}


def test_extract_covered_datasets_no_matching_filter():
    """Test extract_covered_datasets when filter matches nothing."""
    requirement = DataRequirement(
        source_type=SourceDatasetType.CMIP6,
        filters=(FacetFilter(facets={"variable_id": "nonexistent"}),),
        group_by=("variable_id",),
    )
    data_catalog = pd.DataFrame(
        {
            "variable_id": ["tas", "pr"],
            "experiment_id": ["ssp119", "ssp126"],
            "path": ["tas.nc", "pr.nc"],
        }
    )
    result = extract_covered_datasets(data_catalog, requirement)
    assert result == {}


def _build_solver_for_parser(config, sample_data_dir, parser_type, tmp_path, providers):
    """Helper: ingest sample data with the given parser and return a solver."""
    config.cmip6_parser = parser_type

    db_path = tmp_path / f"bench_{parser_type}.db"
    db = Database(f"sqlite:///{db_path}")
    db.migrate(config)

    # Ingest CMIP6
    adapter = CMIP6DatasetAdapter(config=config)
    ingest_datasets(adapter, sample_data_dir / "CMIP6", db, skip_invalid=True)

    # Ingest obs4MIPs + obs4REF
    obs_adapter = Obs4MIPsDatasetAdapter()
    ingest_datasets(obs_adapter, sample_data_dir / "obs4MIPs", db, skip_invalid=True)
    ingest_datasets(obs_adapter, sample_data_dir / "obs4REF", db, skip_invalid=True)

    # Register providers
    with db.session.begin():
        for p in providers:
            _register_provider(db, p)

    solver = ExecutionSolver.build_from_db(config, db)
    solver.provider_registry = ProviderRegistry(providers=providers)
    return solver, db


def test_drs_and_complete_parsers_produce_same_executions(config, sample_data_dir, prepare_db, tmp_path):
    """DRS (lazy) and complete parsers must produce the same solver executions."""
    providers = [pmp_provider, esmvaltool_provider, ilamb_provider]

    solver_complete, db_complete = _build_solver_for_parser(
        config, sample_data_dir, "complete", tmp_path, providers
    )
    solver_drs, db_drs = _build_solver_for_parser(config, sample_data_dir, "drs", tmp_path, providers)

    try:
        complete_executions = sorted((e.diagnostic.slug, e.dataset_key) for e in solver_complete.solve())
        drs_executions = sorted((e.diagnostic.slug, e.dataset_key) for e in solver_drs.solve())

        assert len(complete_executions) == len(drs_executions), (
            f"Complete parser found {len(complete_executions)} executions "
            f"but DRS parser found {len(drs_executions)}.\n"
            f"Only in complete: {set(complete_executions) - set(drs_executions)}\n"
            f"Only in DRS: {set(drs_executions) - set(complete_executions)}"
        )
        assert complete_executions == drs_executions
    finally:
        db_complete.close()
        db_drs.close()
