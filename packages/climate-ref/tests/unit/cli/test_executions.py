import datetime
import pathlib
from unittest.mock import patch

import pytest
from climate_ref_esmvaltool import provider as esmvaltool_provider
from climate_ref_pmp import provider as pmp_provider
from rich.console import Console

from climate_ref.cli.executions import _results_directory_panel
from climate_ref.models import Execution, ExecutionGroup
from climate_ref.models.dataset import CMIP6Dataset
from climate_ref.models.diagnostic import Diagnostic
from climate_ref.models.execution import ExecutionOutput, ResultOutputType, execution_datasets
from climate_ref.models.metric_value import ScalarMetricValue
from climate_ref.provider_registry import _register_provider
from climate_ref_core.datasets import SourceDatasetType


@pytest.fixture
def db_with_groups(db_seeded):
    """Fixture to set up a database with various execution groups for testing filters."""
    with db_seeded.session.begin():
        _register_provider(db_seeded, pmp_provider)
        _register_provider(db_seeded, esmvaltool_provider)

        # Diagnostic 1, Provider 1, Facets: source_id=GFDL-ESM4, variable_id=tas
        diag_1 = (
            db_seeded.session.query(Diagnostic).filter_by(slug="enso_tel").first()
        )  # ENSO diagnostic from PMP
        eg1 = ExecutionGroup(
            key="key1",
            diagnostic_id=diag_1.id,
            selectors={"cmip6": [["source_id", "GFDL-ESM4"], ["variable_id", "tas"]]},
        )
        db_seeded.session.add(eg1)

        # Diagnostic 2, Provider 1, Facets: source_id=ACCESS-ESM1-5, variable_id=pr
        diag_2 = (
            db_seeded.session.query(Diagnostic)
            .filter_by(slug="extratropical-modes-of-variability-nao")
            .first()
        )  # Mode of variability diagnostic from PMP
        eg2 = ExecutionGroup(
            key="key2",
            diagnostic_id=diag_2.id,
            selectors={"cmip6": [["source_id", "ACCESS-ESM1-5"], ["variable_id", "pr"]]},
        )
        db_seeded.session.add(eg2)

        # Diagnostic 1, Provider 2, Facets: source_id=CNRM-CM6-1, variable_id=tas
        diag_3 = (
            db_seeded.session.query(Diagnostic).filter_by(slug="enso-characteristics").first()
        )  # ENSO diagnostic from ESMValTool
        eg3 = ExecutionGroup(
            key="key3",
            diagnostic_id=diag_3.id,
            selectors={"cmip6": [["source_id", "CNRM-CM6-1"], ["variable_id", "tas"]]},
        )
        db_seeded.session.add(eg3)

        # Diagnostic 4, Provider 2, No specific facets (or different ones)
        diag_4 = (
            db_seeded.session.query(Diagnostic).filter_by(slug="sea-ice-area-basic").first()
        )  # ENSO diagnostic from ESMValTool
        eg4 = ExecutionGroup(
            key="key4", diagnostic_id=diag_4.id, selectors={"cmip6": [["experiment_id", "historical"]]}
        )
        db_seeded.session.add(eg4)

        # Add some executions to avoid "not-started" status
        db_seeded.session.flush()
        db_seeded.session.add(
            Execution(
                execution_group_id=eg1.id, successful=True, output_fragment="out1", dataset_hash="hash1"
            )
        )
        db_seeded.session.add(
            Execution(
                execution_group_id=eg2.id, successful=True, output_fragment="out2", dataset_hash="hash2"
            )
        )
        db_seeded.session.add(
            Execution(
                execution_group_id=eg3.id, successful=False, output_fragment="out3", dataset_hash="hash3"
            )
        )
        db_seeded.session.add(
            Execution(
                execution_group_id=eg4.id, successful=True, output_fragment="out4", dataset_hash="hash4"
            )
        )

        # Add a dirty execution group
        eg5 = ExecutionGroup(
            key="key5",
            diagnostic_id=diag_4.id,
            selectors={"cmip6": [["experiment_id", "historical"]]},
            dirty=True,
        )
        db_seeded.session.add(eg5)
        db_seeded.session.flush()
        db_seeded.session.add(
            Execution(
                execution_group_id=eg5.id, successful=True, output_fragment="out5", dataset_hash="hash5"
            )
        )

        # Add an execution group with no executions (not-started)
        eg6 = ExecutionGroup(
            key="key6", diagnostic_id=diag_4.id, selectors={"cmip6": [["experiment_id", "ssp126"]]}
        )
        db_seeded.session.add(eg6)
    db_seeded.session.commit()
    return db_seeded


def test_execution_help(invoke_cli):
    result = invoke_cli(["executions", "--help"])

    assert "View execution groups" in result.stdout


class TestExecutionList:
    def _setup_db(self, db):
        with db.session.begin():
            db.session.add(ExecutionGroup(key="key1", diagnostic_id=1))
            db.session.add(ExecutionGroup(key="key2", diagnostic_id=1))

    def test_list(self, sample_data_dir, db_seeded, invoke_cli):
        self._setup_db(db_seeded)

        result = invoke_cli(["executions", "list-groups"])

        assert "key1" in result.stdout
        assert "key2" in result.stdout
        assert "dirty" in result.stdout

    def test_list_limit(self, sample_data_dir, db_seeded, invoke_cli):
        self._setup_db(db_seeded)

        result = invoke_cli(["executions", "list-groups", "--limit", "1"])

        assert "key1" in result.stdout
        assert "key2" not in result.stdout

    def test_list_columns(self, sample_data_dir, db_seeded, invoke_cli):
        self._setup_db(db_seeded)

        result = invoke_cli(["executions", "list-groups", "--column", "key", "--column", "diagnostic"])

        assert "key1" in result.stdout
        assert "diagnostic" in result.stdout
        assert "dirty" not in result.stdout

    def test_list_columns_missing(self, sample_data_dir, db_seeded, invoke_cli):
        self._setup_db(db_seeded)

        invoke_cli(
            ["executions", "list-groups", "--column", "key", "--column", "missing"], expected_exit_code=1
        )


class TestListGroupsFiltering:
    def test_filter_by_diagnostic(self, db_with_groups, invoke_cli):
        result = invoke_cli(["executions", "list-groups", "--diagnostic", "enso"])
        assert "enso" in result.stdout
        assert "extratropical-modes-of-variability-nao" not in result.stdout
        assert "sea-ice-area-basic" not in result.stdout

    def test_filter_by_provider(self, db_with_groups, invoke_cli):
        result = invoke_cli(["executions", "list-groups", "--provider", "pmp"])

        assert "pmp" in result.stdout
        assert "esmvaltool" not in result.stdout

    @pytest.mark.parametrize("filter_arg", ["source_id=GFDL-ESM4", "cmip6.source_id=GFDL-ESM4"])
    def test_filter_by_facet(self, db_with_groups, invoke_cli, filter_arg):
        result = invoke_cli(["executions", "list-groups", "--filter", filter_arg])
        assert "key1" in result.stdout
        assert "key2" not in result.stdout
        assert "key3" not in result.stdout

    def test_filter_combined(self, db_with_groups, invoke_cli):
        result = invoke_cli(
            [
                "executions",
                "list-groups",
                "--diagnostic",
                "enso",
                "--provider",
                "pmp",
                "--filter",
                "source_id=GFDL-ESM4",
                "--filter",
                "variable_id=tas",
            ]
        )
        assert "key1" in result.stdout
        assert "key2" not in result.stdout
        assert "key3" not in result.stdout
        assert "key4" not in result.stdout

    def test_filter_multiple_diagnostic_or(self, db_with_groups, invoke_cli):
        result = invoke_cli(
            [
                "executions",
                "list-groups",
                "--diagnostic",
                "enso",
                "--diagnostic",
                "extratropical-modes-of-variability-nao",
            ]
        )

        assert "enso" in result.stdout
        assert "extratropical-modes-of-variability-nao" in result.stdout
        assert "sea-ice-area-basic" not in result.stdout

    def test_filter_multiple_provider_or(self, db_with_groups, invoke_cli):
        result = invoke_cli(["executions", "list-groups", "--provider", "pmp", "--provider", "esmvaltool"])
        assert "pmp" in result.stdout
        assert "esmvaltool" in result.stdout

    def test_filter_multiple_facet_and(self, db_with_groups, invoke_cli):
        result = invoke_cli(
            [
                "executions",
                "list-groups",
                "--filter",
                "source_id=GFDL-ESM4",
                "--filter",
                "variable_id=tas",
            ]
        )
        assert "key1" in result.stdout
        assert "key2" not in result.stdout
        assert "key3" not in result.stdout
        assert "key4" not in result.stdout

    def test_filter_invalid_syntax(self, invoke_cli):
        result = invoke_cli(
            ["executions", "list-groups", "--filter", "invalid_no_equals"], expected_exit_code=1
        )
        assert "Invalid filter format" in result.stderr

    def test_filter_empty_results_warning(self, db_with_groups, invoke_cli):
        # Warn if no results after filtering
        result = invoke_cli(["executions", "list-groups", "--filter", "source_id=NONEXISTENT"])
        assert "No execution groups match the specified filters." in result.stderr
        assert "Total execution groups in database:" in result.stderr
        assert "Applied filters: facet filters: ['source_id=NONEXISTENT']" in result.stderr
        assert "id" in result.stdout  # Ensure empty table headers are still printed

    def test_facet_warning_multiple_same_key(self, db_with_groups, invoke_cli):
        # This functionality might be useful in future, but not today
        result = invoke_cli(
            [
                "executions",
                "list-groups",
                "--filter",
                "source_id=GFDL-ESM4",
                "--filter",
                "source_id=ACCESS-ESM1-5",
            ]
        )
        assert (
            "Filter key 'source_id' specified multiple times. Using last value: 'ACCESS-ESM1-5'"
            in result.stderr
        )
        assert "ACCESS-ESM1-5" in result.stdout
        assert "GFDL-ESM4" not in result.stdout

    def test_filter_successful(self, db_with_groups, invoke_cli):
        result = invoke_cli(["executions", "list-groups", "--successful"])
        # Should include key1, key2, key4, key5 (successful=True),
        # exclude key3 (successful=False), exclude key6 (no executions)
        assert "key1" in result.stdout
        assert "key2" in result.stdout
        assert "key3" not in result.stdout
        assert "key4" in result.stdout
        assert "key5" in result.stdout
        assert "key6" not in result.stdout

    def test_filter_not_successful(self, db_with_groups, invoke_cli):
        result = invoke_cli(["executions", "list-groups", "--not-successful"])
        # Should include key3 (successful=False) and key6 (no executions), exclude successful ones
        assert "key1" not in result.stdout
        assert "key2" not in result.stdout
        assert "key3" in result.stdout
        assert "key4" not in result.stdout
        assert "key5" not in result.stdout
        assert "key6" in result.stdout

    def test_filter_dirty(self, db_with_groups, invoke_cli):
        result = invoke_cli(["executions", "list-groups", "--dirty"])
        # Should include key5 (dirty=True), exclude others (dirty=False by default)
        assert "key1" not in result.stdout
        assert "key2" not in result.stdout
        assert "key3" not in result.stdout
        assert "key4" not in result.stdout
        assert "key5" in result.stdout
        assert "key6" not in result.stdout

    def test_filter_not_dirty(self, db_with_groups, invoke_cli):
        result = invoke_cli(["executions", "list-groups", "--not-dirty"])
        # Should include key1, key2, key3, key4, key6 (dirty=False), exclude key5 (dirty=True)
        assert "key1" in result.stdout
        assert "key2" in result.stdout
        assert "key3" in result.stdout
        assert "key4" in result.stdout
        assert "key5" not in result.stdout
        assert "key6" in result.stdout


class TestDeleteGroups:
    def test_delete_groups_with_confirmation(self, db_with_groups, invoke_cli):
        # Count before deletion
        initial_count = db_with_groups.session.query(ExecutionGroup).count()

        with patch("climate_ref.cli.executions.typer.confirm", return_value=True):
            result = invoke_cli(["executions", "delete-groups", "--diagnostic", "enso"])

        assert result.exit_code == 0
        assert "Successfully deleted" in result.stdout
        assert "Execution groups to be deleted:" in result.stdout

        # Verify deletion
        remaining_count = db_with_groups.session.query(ExecutionGroup).count()
        assert remaining_count < initial_count

    def test_delete_groups_cancellation(self, db_with_groups, invoke_cli):
        initial_count = db_with_groups.session.query(ExecutionGroup).count()

        with patch("climate_ref.cli.executions.typer.confirm", return_value=False):
            result = invoke_cli(["executions", "delete-groups", "--diagnostic", "enso"])

        assert result.exit_code == 0
        assert "Deletion cancelled." in result.stdout

        # Verify no deletion
        remaining_count = db_with_groups.session.query(ExecutionGroup).count()
        assert remaining_count == initial_count

    def test_delete_groups_force_flag(self, db_with_groups, invoke_cli):
        initial_count = db_with_groups.session.query(ExecutionGroup).count()

        result = invoke_cli(["executions", "delete-groups", "--diagnostic", "enso", "--force"])

        assert result.exit_code == 0
        assert "Successfully deleted" in result.stdout
        assert "Execution groups to be deleted:" in result.stdout

        # Verify deletion
        remaining_count = db_with_groups.session.query(ExecutionGroup).count()
        assert remaining_count < initial_count

    def test_delete_groups_no_deletion_on_decline(self, db_with_groups, invoke_cli):
        initial_count = db_with_groups.session.query(ExecutionGroup).count()

        with patch("climate_ref.cli.executions.typer.confirm", return_value=False):
            result = invoke_cli(["executions", "delete-groups", "--diagnostic", "enso"])

        assert result.exit_code == 0
        assert "Deletion cancelled." in result.stdout

        # Verify no deletion
        remaining_count = db_with_groups.session.query(ExecutionGroup).count()
        assert remaining_count == initial_count

    def test_delete_groups_filter_diagnostic(self, db_with_groups, invoke_cli):
        with patch("climate_ref.cli.executions.typer.confirm", return_value=True):
            result = invoke_cli(["executions", "delete-groups", "--diagnostic", "enso", "--force"])

        assert result.exit_code == 0
        assert "Successfully deleted" in result.stdout

    def test_delete_groups_filter_provider(self, db_with_groups, invoke_cli):
        with patch("climate_ref.cli.executions.typer.confirm", return_value=True):
            result = invoke_cli(["executions", "delete-groups", "--provider", "pmp", "--force"])

        assert result.exit_code == 0
        assert "Successfully deleted" in result.stdout

    def test_delete_groups_filter_facet(self, db_with_groups, invoke_cli):
        with patch("climate_ref.cli.executions.typer.confirm", return_value=True):
            result = invoke_cli(["executions", "delete-groups", "--filter", "source_id=GFDL-ESM4", "--force"])

        assert result.exit_code == 0
        assert "Successfully deleted" in result.stdout

    def test_delete_groups_filter_successful(self, db_with_groups, invoke_cli):
        with patch("climate_ref.cli.executions.typer.confirm", return_value=True):
            result = invoke_cli(["executions", "delete-groups", "--successful", "--force"])

        assert result.exit_code == 0
        assert "Successfully deleted" in result.stdout

    def test_delete_groups_filter_dirty(self, db_with_groups, invoke_cli):
        with patch("climate_ref.cli.executions.typer.confirm", return_value=True):
            result = invoke_cli(["executions", "delete-groups", "--dirty", "--force"])

        assert result.exit_code == 0
        assert "Successfully deleted" in result.stdout

    def test_delete_groups_multiple_filters(self, db_with_groups, invoke_cli):
        with patch("climate_ref.cli.executions.typer.confirm", return_value=True):
            result = invoke_cli(
                [
                    "executions",
                    "delete-groups",
                    "--diagnostic",
                    "enso",
                    "--provider",
                    "pmp",
                    "--filter",
                    "source_id=GFDL-ESM4",
                    "--force",
                ]
            )

        assert result.exit_code == 0
        assert "Successfully deleted" in result.stdout

    def test_delete_groups_no_filters_error(self, invoke_cli):
        with patch("climate_ref.cli.executions.typer.confirm", return_value=False):
            result = invoke_cli(["executions", "delete-groups"], expected_exit_code=1)

        assert "THIS WILL DELETE ALL EXECUTION GROUPS IN THE DATABASE" in result.stderr

    def test_delete_groups_no_results_warning(self, db_with_groups, invoke_cli):
        result = invoke_cli(["executions", "delete-groups", "--filter", "source_id=NONEXISTENT", "--force"])

        assert result.exit_code == 0
        assert "No execution groups match the specified filters." in result.stderr

    def test_delete_groups_cascade_deletes_all_related_models(self, db_with_groups, invoke_cli):
        """Test that delete-groups properly deletes ExecutionGroups, Executions,
        ExecutionOutputs, MetricValues, and execution_datasets associations."""

        # Get the execution groups that match "enso" diagnostic (eg1 and eg3)
        enso_groups = [
            eg
            for eg in db_with_groups.session.query(ExecutionGroup).all()
            if "enso" in eg.diagnostic.slug.lower()
        ]

        # Count datasets before creating a new one (db_seeded has existing datasets)
        initial_dataset_count_before_test = db_with_groups.session.query(CMIP6Dataset).count()

        # Create a shared Dataset for associations
        with db_with_groups.session.begin_nested():
            dataset = CMIP6Dataset(
                slug="test-cmip6-dataset",
                dataset_type=SourceDatasetType.CMIP6,
                activity_id="CMIP",
                experiment_id="historical",
                institution_id="TEST",
                source_id="TEST-MODEL",
                member_id="r1i1p1f1",
                table_id="Amon",
                variable_id="tas",
                grid_label="gn",
                version="v20200101",
                instance_id="CMIP.TEST.TEST-MODEL.historical.Amon.gn",
                variant_label="r1i1p1f1",
            )
            db_with_groups.session.add(dataset)
            db_with_groups.session.flush()

            # Add ExecutionOutputs, MetricValues, and Dataset associations
            for eg in enso_groups:
                for execution in eg.executions:
                    # Add ExecutionOutput
                    output = ExecutionOutput(
                        execution_id=execution.id,
                        output_type=ResultOutputType.Plot,
                        filename="test_plot.png",
                    )
                    db_with_groups.session.add(output)

                    # Add MetricValue
                    metric_value = ScalarMetricValue(
                        execution_id=execution.id,
                        value=42.0,
                        attributes={"test_attr": "test_value"},
                    )
                    db_with_groups.session.add(metric_value)

                    # Add Dataset association
                    execution.datasets.append(dataset)

        db_with_groups.session.commit()

        # Get initial counts before deletion
        initial_exec_count = db_with_groups.session.query(Execution).count()
        initial_output_count = db_with_groups.session.query(ExecutionOutput).count()
        initial_metric_count = db_with_groups.session.query(ScalarMetricValue).count()
        initial_dataset_count = db_with_groups.session.query(CMIP6Dataset).count()

        # Count execution_datasets associations before deletion
        initial_assoc_count = len(db_with_groups.session.execute(execution_datasets.select()).fetchall())

        # Verify we have created the related models
        assert initial_output_count > 0, "Should have ExecutionOutputs"
        assert initial_metric_count > 0, "Should have MetricValues"
        assert initial_assoc_count > 0, "Should have execution_datasets associations"
        assert initial_dataset_count == initial_dataset_count_before_test + 1, (
            "Should have one more dataset than before"
        )

        # Perform deletion
        with patch("climate_ref.cli.executions.typer.confirm", return_value=True):
            result = invoke_cli(["executions", "delete-groups", "--diagnostic", "enso", "--force"])

        assert result.exit_code == 0

        # Verify executions are deleted
        remaining_exec_count = db_with_groups.session.query(Execution).count()
        assert remaining_exec_count < initial_exec_count, "Executions should be deleted"

        # Verify ExecutionOutputs are deleted
        remaining_output_count = db_with_groups.session.query(ExecutionOutput).count()
        assert remaining_output_count < initial_output_count, "ExecutionOutputs should be deleted"

        # Verify MetricValues are deleted
        remaining_metric_count = db_with_groups.session.query(ScalarMetricValue).count()
        assert remaining_metric_count < initial_metric_count, "MetricValues should be deleted"

        # Verify execution_datasets associations are deleted
        remaining_assoc_count = len(db_with_groups.session.execute(execution_datasets.select()).fetchall())
        assert remaining_assoc_count < initial_assoc_count, (
            "execution_datasets associations should be deleted"
        )

        # Verify Datasets themselves are NOT deleted (just the association)
        remaining_dataset_count = db_with_groups.session.query(CMIP6Dataset).count()
        assert remaining_dataset_count == initial_dataset_count, (
            "Datasets should still exist (only associations removed)"
        )

    def test_delete_groups_removes_outputs(self, db_with_groups, tmp_path, invoke_cli, config):
        # Create actual output directories in tmp_path
        results_path = tmp_path / "results"
        results_path.mkdir()

        # Mock config.paths.results to use tmp_path
        config.paths.results = results_path
        config.save()

        # Create execution and its output directory
        eg = db_with_groups.session.query(ExecutionGroup).first()
        execution = eg.executions[0]
        output_dir = results_path / execution.output_fragment
        output_dir.mkdir(parents=True)

        # Verify directory exists before deletion
        assert output_dir.exists()

        # Run command with --remove-outputs
        with patch("climate_ref.cli.executions.typer.confirm", return_value=True):
            result = invoke_cli(
                [
                    "executions",
                    "delete-groups",
                    "--diagnostic",
                    "enso",
                    "--remove-outputs",
                    "--force",
                ]
            )

        # Assert success
        assert result.exit_code == 0

        # Verify output directory was removed
        assert not output_dir.exists()

        # Verify database records deleted (only enso diagnostics: eg1 and eg3)
        # Remaining: eg2, eg4, eg5, eg6 = 4 groups
        assert db_with_groups.session.query(ExecutionGroup).count() == 4

        # Verify success message includes output directories
        assert "and their output directories" in result.stdout

    def test_delete_groups_without_remove_outputs_flag(self, db_with_groups, tmp_path, invoke_cli, config):
        """Test that output directories are NOT removed when --remove-outputs flag is omitted"""
        # Create actual output directories in tmp_path
        results_path = tmp_path / "results"
        results_path.mkdir()

        # Mock config.paths.results to use tmp_path
        config.paths.results = results_path
        config.save()

        # Create execution and its output directory
        eg = db_with_groups.session.query(ExecutionGroup).first()
        execution = eg.executions[0]
        output_dir = results_path / execution.output_fragment
        output_dir.mkdir(parents=True)

        # Verify directory exists before deletion
        assert output_dir.exists()

        # Run command WITHOUT --remove-outputs
        with patch("climate_ref.cli.executions.typer.confirm", return_value=True):
            result = invoke_cli(["executions", "delete-groups", "--diagnostic", "enso", "--force"])

        # Assert success
        assert result.exit_code == 0

        # Verify output directory still exists
        assert output_dir.exists()

        # Verify database records deleted (only enso diagnostics: eg1 and eg3)
        # Remaining: eg2, eg4, eg5, eg6 = 4 groups
        assert db_with_groups.session.query(ExecutionGroup).count() == 4

        # Verify success message does NOT include output directories
        assert "and their output directories" not in result.stdout

    def test_delete_groups_remove_outputs_nonexistent_directory(
        self, db_with_groups, tmp_path, invoke_cli, config
    ):
        """Test graceful handling when output directory doesn't exist"""
        # Create results path but not the output directories
        results_path = tmp_path / "results"
        results_path.mkdir()

        # Mock config.paths.results to use tmp_path
        config.paths.results = results_path
        config.save()

        # Get execution with output_fragment (directories don't exist)
        eg = db_with_groups.session.query(ExecutionGroup).first()
        execution = eg.executions[0]
        output_dir = results_path / execution.output_fragment

        # Verify directory does NOT exist
        assert not output_dir.exists()

        # Run command with --remove-outputs
        with patch("climate_ref.cli.executions.typer.confirm", return_value=True):
            result = invoke_cli(
                [
                    "executions",
                    "delete-groups",
                    "--diagnostic",
                    "enso",
                    "--remove-outputs",
                    "--force",
                ]
            )

        # Assert success (no errors for missing directories)
        assert result.exit_code == 0

        # Verify database records deleted (only enso diagnostics: eg1 and eg3)
        # Remaining: eg2, eg4, eg5, eg6 = 4 groups
        assert db_with_groups.session.query(ExecutionGroup).count() == 4

    def test_delete_groups_remove_outputs_filesystem_error(
        self, db_with_groups, tmp_path, invoke_cli, config
    ):
        """Test error handling for filesystem failures during output removal"""
        # Create actual output directories in tmp_path
        results_path = tmp_path / "results"
        results_path.mkdir()

        # Mock config.paths.results to use tmp_path
        config.paths.results = results_path
        config.save()

        # Create execution and its output directory
        eg = db_with_groups.session.query(ExecutionGroup).first()
        execution = eg.executions[0]
        output_dir = results_path / execution.output_fragment
        output_dir.mkdir(parents=True)

        # Verify directory exists before deletion
        assert output_dir.exists()

        # Mock shutil.rmtree to raise an exception
        with patch("shutil.rmtree", side_effect=OSError("Permission denied")):
            with patch("climate_ref.cli.executions.typer.confirm", return_value=True):
                result = invoke_cli(
                    [
                        "executions",
                        "delete-groups",
                        "--diagnostic",
                        "enso",
                        "--remove-outputs",
                        "--force",
                    ]
                )

        # Assert success (command should not fail due to filesystem error)
        assert result.exit_code == 0

        # Verify database records are still deleted despite filesystem error (only enso diagnostics)
        # Remaining: eg2, eg4, eg5, eg6 = 4 groups
        assert db_with_groups.session.query(ExecutionGroup).count() == 4

        # Verify output directory still exists (since rmtree failed)
        assert output_dir.exists()


class TestExecutionInspect:
    def test_inspect(self, sample_data_dir, db_seeded, invoke_cli, file_regression, config):
        # Ensure the executions path is consistent
        config.paths.results = pathlib.Path("/results")
        config.save()

        # Create a diagnostic execution group with a result
        execution_group = ExecutionGroup(
            key="key1",
            diagnostic_id=1,
            # Ensure dates are consistent
            created_at=datetime.datetime(2021, 1, 1),
            updated_at=datetime.datetime(2021, 2, 1),
        )
        with db_seeded.session.begin():
            db_seeded.session.add(execution_group)
            db_seeded.session.flush()

            execution = Execution(
                execution_group_id=execution_group.id,
                successful=True,
                output_fragment="output",
                dataset_hash="hash",
            )
            db_seeded.session.add(execution)
            db_seeded.session.flush()
            db_seeded.session.execute(
                execution_datasets.insert(),
                [{"execution_id": execution.id, "dataset_id": idx} for idx in [1, 2]],
            )
        result = invoke_cli(["executions", "inspect", str(execution_group.id)])

        assert "Successful: True" in result.stdout
        file_regression.check(result.stdout)

    def test_inspect_failed(self, sample_data_dir, db_seeded, invoke_cli):
        # Create a diagnostic execution group with a result
        execution_group = ExecutionGroup(
            key="key1",
            diagnostic_id=1,
        )
        with db_seeded.session.begin():
            db_seeded.session.add(execution_group)
            db_seeded.session.flush()

            result = Execution(
                execution_group_id=execution_group.id,
                successful=False,
                output_fragment="output",
                dataset_hash="hash",
            )
            db_seeded.session.add(result)

        result = invoke_cli(["executions", "inspect", str(execution_group.id)])

        assert "Successful: False" in result.stdout

    def test_inspect_no_results(self, sample_data_dir, db_seeded, invoke_cli):
        metric_execution_group = ExecutionGroup(key="key1", diagnostic_id=1)
        with db_seeded.session.begin():
            db_seeded.session.add(metric_execution_group)

        result = invoke_cli(["executions", "inspect", str(metric_execution_group.id)])

        assert "not-started" in result.stdout

    def test_inspect_missing(self, invoke_cli):
        result = invoke_cli(["executions", "inspect", "999"], expected_exit_code=1)
        assert "Execution not found: 999" in result.stderr

    def test_results_directory_panel(self, tmp_path):
        tmp_path = tmp_path / "inner"
        tmp_path.mkdir()

        with open(tmp_path / "file1.txt", "w") as f:
            f.write("test")

        tmp_path.joinpath(".hidden").touch()

        inner_dir = tmp_path / "dir1"
        inner_dir.mkdir()
        inner_dir.joinpath("file2").touch()

        table = _results_directory_panel(tmp_path)

        console = Console()
        with console.capture() as capture:
            console.print(table)

        assert "file1.txt (4 bytes)" in capture.get()
        assert "┣━━ 📂 dir1" in capture.get()
        assert "hidden" not in capture.get()

    def test_flag_dirty(self, sample_data_dir, db_seeded, invoke_cli, config):
        config.paths.results = pathlib.Path("/results")
        config.save()
        execution_group = ExecutionGroup(
            key="key1",
            diagnostic_id=1,
            created_at=datetime.datetime(2021, 1, 1),
            updated_at=datetime.datetime(2021, 2, 1),
        )
        with db_seeded.session.begin():
            db_seeded.session.add(execution_group)
            db_seeded.session.flush()
            execution = Execution(
                execution_group_id=execution_group.id,
                successful=True,
                output_fragment="output",
                dataset_hash="hash",
            )
            db_seeded.session.add(execution)
            db_seeded.session.flush()
            db_seeded.session.execute(
                execution_datasets.insert(),
                [{"execution_id": execution.id, "dataset_id": idx} for idx in [1, 2]],
            )
        result = invoke_cli(["executions", "inspect", str(execution_group.id)])
        assert "Dirty: False" in result.stdout
        result = invoke_cli(["executions", "flag-dirty", str(execution_group.id)])
        assert "Dirty: True" in result.stdout

    def test_flag_dirty_missing(self, db_seeded, invoke_cli):
        invoke_cli(["executions", "flag-dirty", "123"], expected_exit_code=1)


class TestFailRunning:
    @pytest.fixture
    def db_with_running(self, db_seeded):
        """Fixture with running executions (successful=None) across different providers."""
        with db_seeded.session.begin():
            _register_provider(db_seeded, pmp_provider)
            _register_provider(db_seeded, esmvaltool_provider)

            diag_pmp = db_seeded.session.query(Diagnostic).filter_by(slug="enso_tel").first()
            diag_esmvaltool = (
                db_seeded.session.query(Diagnostic).filter_by(slug="enso-characteristics").first()
            )

            # Running execution for PMP diagnostic (created recently)
            eg1 = ExecutionGroup(
                key="running-pmp",
                diagnostic_id=diag_pmp.id,
                selectors={"cmip6": [["source_id", "GFDL-ESM4"]]},
            )
            db_seeded.session.add(eg1)

            # Running execution for ESMValTool diagnostic (created long ago)
            eg2 = ExecutionGroup(
                key="running-esmvaltool",
                diagnostic_id=diag_esmvaltool.id,
                selectors={"cmip6": [["source_id", "ACCESS-ESM1-5"]]},
            )
            db_seeded.session.add(eg2)

            # Successful execution (should not be touched)
            eg3 = ExecutionGroup(
                key="completed",
                diagnostic_id=diag_pmp.id,
                selectors={"cmip6": [["source_id", "CNRM-CM6-1"]]},
            )
            db_seeded.session.add(eg3)

            db_seeded.session.flush()

            # Recent running execution
            db_seeded.session.add(
                Execution(
                    execution_group_id=eg1.id,
                    successful=None,
                    output_fragment="out-running-1",
                    dataset_hash="hash-r1",
                )
            )

            # Old running execution (48 hours ago)
            old_time = datetime.datetime.now(tz=datetime.timezone.utc) - datetime.timedelta(hours=48)
            db_seeded.session.add(
                Execution(
                    execution_group_id=eg2.id,
                    successful=None,
                    output_fragment="out-running-2",
                    dataset_hash="hash-r2",
                    created_at=old_time,
                )
            )

            # Completed execution (should not be affected)
            db_seeded.session.add(
                Execution(
                    execution_group_id=eg3.id,
                    successful=True,
                    output_fragment="out-done",
                    dataset_hash="hash-done",
                )
            )

        db_seeded.session.commit()
        return db_seeded

    def test_fail_all_running(self, db_with_running, invoke_cli):
        result = invoke_cli(["executions", "fail-running", "--force"])

        assert result.exit_code == 0
        assert "Successfully marked 2 execution(s) as failed" in result.stdout

        # Verify both running executions are now failed
        session = db_with_running.session
        running = session.query(Execution).filter(Execution.successful.is_(None)).count()
        assert running == 0

        # Verify completed execution is untouched
        completed = session.query(Execution).filter(Execution.successful.is_(True)).count()
        assert completed == 1

    def test_fail_running_marks_groups_dirty(self, db_with_running, invoke_cli):
        result = invoke_cli(["executions", "fail-running", "--force"])

        assert result.exit_code == 0

        session = db_with_running.session
        eg = session.query(ExecutionGroup).filter_by(key="running-pmp").first()
        assert eg.dirty is True

        eg2 = session.query(ExecutionGroup).filter_by(key="running-esmvaltool").first()
        assert eg2.dirty is True

    def test_fail_running_older_than(self, db_with_running, invoke_cli):
        # Only fail executions older than 24 hours — should only catch the 48h-old one
        result = invoke_cli(["executions", "fail-running", "--older-than", "24", "--force"])

        assert result.exit_code == 0
        assert "Successfully marked 1 execution(s) as failed" in result.stdout

        session = db_with_running.session
        # The recent one should still be running
        running = session.query(Execution).filter(Execution.successful.is_(None)).count()
        assert running == 1

    def test_fail_running_older_than_none_match(self, db_with_running, invoke_cli):
        # Use a very large threshold — nothing should match
        result = invoke_cli(["executions", "fail-running", "--older-than", "9999", "--force"])

        assert result.exit_code == 0
        assert "No running executions found" in result.stdout

    def test_fail_running_filter_by_provider(self, db_with_running, invoke_cli):
        result = invoke_cli(["executions", "fail-running", "--provider", "pmp", "--force"])

        assert result.exit_code == 0
        assert "Successfully marked 1 execution(s) as failed" in result.stdout
        assert "enso_tel" in result.stdout

    def test_fail_running_filter_by_diagnostic(self, db_with_running, invoke_cli):
        result = invoke_cli(["executions", "fail-running", "--diagnostic", "enso-characteristics", "--force"])

        assert result.exit_code == 0
        assert "Successfully marked 1 execution(s) as failed" in result.stdout
        assert "enso-characteristics" in result.stdout

    def test_fail_running_no_running_executions(self, db_with_groups, invoke_cli):
        # db_with_groups has no running executions (all are True/False)
        result = invoke_cli(["executions", "fail-running", "--force"])

        assert result.exit_code == 0
        assert "No running executions found" in result.stdout

    def test_fail_running_cancelled(self, db_with_running, invoke_cli):
        with patch("climate_ref.cli.executions.typer.confirm", return_value=False):
            result = invoke_cli(["executions", "fail-running"])

        assert result.exit_code == 0
        assert "Cancelled." in result.stdout

        # Verify nothing was changed
        session = db_with_running.session
        running = session.query(Execution).filter(Execution.successful.is_(None)).count()
        assert running == 2

    def test_fail_running_with_confirmation(self, db_with_running, invoke_cli):
        with patch("climate_ref.cli.executions.typer.confirm", return_value=True):
            result = invoke_cli(["executions", "fail-running"])

        assert result.exit_code == 0
        assert "Successfully marked 2 execution(s) as failed" in result.stdout
