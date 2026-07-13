import datetime
import json
import pathlib
from typing import ClassVar
from unittest.mock import patch

import pytest
from climate_ref_esmvaltool import provider as esmvaltool_provider
from climate_ref_pmp import provider as pmp_provider
from rich.console import Console

from climate_ref.cli.executions import _outputs_panel, _results_directory_panel
from climate_ref.models import Execution, ExecutionGroup
from climate_ref.models.dataset import CMIP6Dataset
from climate_ref.models.diagnostic import Diagnostic
from climate_ref.models.execution import ExecutionOutput, ResultOutputType, execution_datasets
from climate_ref.models.metric_value import ScalarMetricValue, SeriesIndex, SeriesMetricValue
from climate_ref.provider_registry import _register_provider
from climate_ref.results import Reader
from climate_ref.results.executions import OutputView
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

    def test_list_default_columns_omit_verbose(self, sample_data_dir, db_seeded, invoke_cli):
        self._setup_db(db_seeded)

        result = invoke_cli(["executions", "list-groups"])

        # The default table shows a sane subset; verbose columns are omitted.
        assert "dirty" in result.stdout
        assert "created_at" in result.stdout
        assert "selectors" not in result.stdout
        assert "updated_at" not in result.stdout

    def test_list_json(self, sample_data_dir, db_seeded, invoke_cli):
        import json

        self._setup_db(db_seeded)

        result = invoke_cli(["executions", "list-groups", "--format", "json"])

        payload = json.loads(result.stdout)
        keys = {row["key"] for row in payload}
        assert {"key1", "key2"} <= keys

        # ``selectors`` must be emitted as structured JSON, not a double-encoded
        # string (i.e. ``{}`` not ``"{}"``), so callers don't have to parse twice.
        assert all(isinstance(row["selectors"], dict) for row in payload)

    def test_list_json_empty(self, sample_data_dir, db_seeded, invoke_cli):
        import json

        # No execution groups created: JSON output must still be valid (an empty list).
        result = invoke_cli(["executions", "list-groups", "--format", "json"])

        assert json.loads(result.stdout) == []


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

    def test_facet_multiple_same_key_returns_both(self, db_with_groups, invoke_cli):
        # Multiple values for the same key are ORed (both should appear)
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
        # key1 -> source_id=GFDL-ESM4, key2 -> source_id=ACCESS-ESM1-5.
        # selectors are omitted from the default table, so assert on the group keys.
        assert "key1" in result.stdout
        assert "key2" in result.stdout

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

    def test_delete_groups_backend_error_propagates(self, db_with_groups, invoke_cli):
        """A genuine backend error must surface, not be relabelled 'Error applying filters'."""

        def _boom(*args, **kwargs):
            raise RuntimeError("db exploded")

        with patch("climate_ref.cli.executions.get_execution_group_and_latest_filtered", _boom):
            result = invoke_cli(
                ["executions", "delete-groups", "--diagnostic", "enso", "--force"],
                expected_exit_code=1,
            )

        assert "Error applying filters" not in result.stdout
        assert "Error applying filters" not in result.stderr
        assert isinstance(result.exception, RuntimeError)
        assert str(result.exception) == "db exploded"

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

    def test_delete_groups_skips_escaping_output_fragment(self, db_with_groups, tmp_path, invoke_cli, config):
        """An output_fragment that escapes the results root via '..' must not be deleted."""
        results_path = tmp_path / "results"
        results_path.mkdir()

        # Mock config.paths.results to use tmp_path
        config.paths.results = results_path
        config.save()

        # A directory outside the results root that a malicious/corrupt fragment would target.
        outside_dir = tmp_path / "outside"
        outside_dir.mkdir()
        (outside_dir / "sentinel.txt").write_text("do not delete")

        # Point the execution's output_fragment outside the results root.
        eg = db_with_groups.session.query(ExecutionGroup).first()
        execution = eg.executions[0]
        execution.output_fragment = "../outside"
        db_with_groups.session.commit()

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

        # Assert success (the unsafe fragment is skipped, not fatal)
        assert result.exit_code == 0

        # Verify the directory outside the results root was NOT removed
        assert outside_dir.exists()
        assert (outside_dir / "sentinel.txt").exists()

        # Verify database records were still deleted (only enso diagnostics: eg1 and eg3)
        assert db_with_groups.session.query(ExecutionGroup).count() == 4

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


class TestExecutionStats:
    def test_stats_no_data(self, db_seeded, invoke_cli):
        result = invoke_cli(["executions", "stats"])
        assert "No execution groups found." in result.stdout

    def test_stats_basic(self, db_with_groups, invoke_cli):
        result = invoke_cli(["executions", "stats"])
        # db_with_groups has both pmp and esmvaltool providers
        assert "pmp" in result.stdout
        assert "esmvaltool" in result.stdout

    def test_stats_shows_status_columns(self, db_with_groups, invoke_cli):
        result = invoke_cli(["executions", "stats"])
        assert "diagnostic" in result.stdout
        assert "running" in result.stdout
        assert "failed" in result.stdout
        assert "successful" in result.stdout
        assert "not_started" in result.stdout
        assert "dirty" in result.stdout
        assert "total" in result.stdout

    def test_stats_includes_diagnostic_and_totals(self, db_with_groups, invoke_cli):
        """Verify that the output includes per-diagnostic rows and provider totals.

        The fixture creates diagnostics across pmp and esmvaltool providers.
        Each provider should have a (total) row.
        """
        result = invoke_cli(["executions", "stats"])
        assert "(total)" in result.stdout
        # Individual diagnostics should appear
        assert "enso_tel" in result.stdout
        assert "enso-characteristics" in result.stdout

    def test_stats_filter_by_provider(self, db_with_groups, invoke_cli):
        result = invoke_cli(["executions", "stats", "--provider", "pmp"])
        assert "pmp" in result.stdout
        assert "esmvaltool" not in result.stdout

    def test_stats_filter_by_diagnostic(self, db_with_groups, invoke_cli):
        result = invoke_cli(["executions", "stats", "--diagnostic", "enso"])
        # Only diagnostics matching "enso" should appear
        assert result.exit_code == 0

    def test_stats_filter_no_results(self, db_with_groups, invoke_cli):
        result = invoke_cli(["executions", "stats", "--provider", "nonexistent"])
        assert "No execution groups found." in result.stdout


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

    def test_inspect_outputs_panel(self, sample_data_dir, db_seeded, invoke_cli, config):
        # Real on-disk results root so the directory-tree / log panels keep their existing behaviour.
        results_path = config.paths.results
        results_path.mkdir(parents=True, exist_ok=True)

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

            output_dir = results_path / execution.output_fragment
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "plot.png").write_text("fake plot")

            db_seeded.session.add(
                ExecutionOutput(
                    execution_id=execution.id,
                    output_type=ResultOutputType.Plot,
                    filename="plot.png",
                    short_name="plot-short",
                    long_name="Plot long name",
                )
            )
            # A metadata-only row with no filename must not emit a bogus file link/path.
            db_seeded.session.add(
                ExecutionOutput(
                    execution_id=execution.id,
                    output_type=ResultOutputType.HTML,
                    filename=None,
                    short_name="html-short",
                    long_name="HTML long name",
                )
            )

        result = invoke_cli(["executions", "inspect", str(execution_group.id)])

        assert "Outputs" in result.stdout
        assert "plot-short" in result.stdout
        assert "plot.png" in result.stdout
        assert "html-short" in result.stdout

        # Directory-tree / log panels still render (regression).
        assert "File Tree" in result.stdout
        assert "Execution Logs" in result.stdout
        assert "plot.png (9 bytes)" in result.stdout

    def test_inspect_no_results_skips_outputs_panel(self, sample_data_dir, db_seeded, invoke_cli):
        metric_execution_group = ExecutionGroup(key="key1", diagnostic_id=1)
        with db_seeded.session.begin():
            db_seeded.session.add(metric_execution_group)

        result = invoke_cli(["executions", "inspect", str(metric_execution_group.id)])

        assert result.exit_code == 0
        assert "not-started" in result.stdout
        assert "Outputs" not in result.stdout

    def test_outputs_panel_no_outputs(self, db_seeded, tmp_path):
        reader = Reader(db_seeded, results=tmp_path)
        panel = _outputs_panel((), "frag", reader)

        console = Console()
        with console.capture() as capture:
            console.print(panel)

        assert "No registered outputs." in capture.get()

    def test_outputs_panel_links_filename_rows_only(self, db_seeded, tmp_path):
        outputs = (
            OutputView(
                execution_id=1,
                output_type="plot",
                filename="plot.png",
                short_name="plot-short",
                long_name="Plot long name",
                description=None,
                dimensions={},
            ),
            OutputView(
                execution_id=1,
                output_type="html",
                filename=None,
                short_name="html-short",
                long_name="HTML long name",
                description=None,
                dimensions={},
            ),
        )
        reader = Reader(db_seeded, results=tmp_path)
        panel = _outputs_panel(outputs, "frag", reader)

        # `force_terminal=True` so rich emits the OSC 8 hyperlink escape sequence to inspect.
        console = Console(force_terminal=True)
        with console.capture() as capture:
            console.print(panel)
        rendered = capture.get()

        assert f"file://{tmp_path / 'frag' / 'plot.png'}" in rendered
        assert "html-short" in rendered
        assert "file://" not in rendered.split("html-short")[1]

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
            old_time = datetime.datetime.now(tz=datetime.UTC) - datetime.timedelta(hours=48)
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


class TestReingestCLI:
    """Tests for the `executions reingest` CLI command."""

    @pytest.fixture
    def db_with_executions(self, db_seeded, config):
        """Create a DB with execution groups that have output directories on disk."""
        with db_seeded.session.begin():
            _register_provider(db_seeded, pmp_provider)
            _register_provider(db_seeded, esmvaltool_provider)

            diag_pmp = db_seeded.session.query(Diagnostic).filter_by(slug="enso_tel").first()
            diag_esm = db_seeded.session.query(Diagnostic).filter_by(slug="enso-characteristics").first()

            eg1 = ExecutionGroup(
                key="reingest-1",
                diagnostic_id=diag_pmp.id,
                selectors={"cmip6": [["source_id", "GFDL-ESM4"]]},
            )
            eg2 = ExecutionGroup(
                key="reingest-2",
                diagnostic_id=diag_esm.id,
                selectors={"cmip6": [["source_id", "ACCESS-ESM1-5"]]},
            )
            db_seeded.session.add_all([eg1, eg2])
            db_seeded.session.flush()

            ex1 = Execution(
                execution_group_id=eg1.id,
                successful=True,
                output_fragment="pmp/enso_tel/out1",
                dataset_hash="h1",
            )
            ex2 = Execution(
                execution_group_id=eg2.id,
                successful=False,
                output_fragment="esmvaltool/enso-char/out2",
                dataset_hash="h2",
            )
            db_seeded.session.add_all([ex1, ex2])

        db_seeded.session.commit()
        return db_seeded

    def test_reingest_no_filters_error(self, invoke_cli, db_seeded):
        """Calling reingest with no filters should exit with code 1."""
        result = invoke_cli(["executions", "reingest"], expected_exit_code=1)
        assert "At least one filter is required" in result.stderr

    def test_reingest_no_results(self, db_with_executions, invoke_cli):
        """When filters match nothing, should print a message and exit cleanly."""
        result = invoke_cli(["executions", "reingest", "--provider", "nonexistent", "--force"])
        assert result.exit_code == 0
        assert "No executions found" in result.stdout

    def test_reingest_dry_run(self, db_with_executions, invoke_cli):
        """Dry run should show preview without making changes."""
        result = invoke_cli(["executions", "reingest", "--provider", "pmp", "--dry-run", "--force"])
        assert result.exit_code == 0
        assert "Dry run" in result.stdout
        assert "enso_tel" in result.stdout

    def test_reingest_cancellation(self, db_with_executions, invoke_cli):
        """User declining confirmation should cancel reingest."""
        with patch("climate_ref.cli.executions.typer.confirm", return_value=False):
            result = invoke_cli(["executions", "reingest", "--provider", "pmp"])
        assert result.exit_code == 0
        assert "Reingest cancelled" in result.stdout

    def test_reingest_force_runs(self, db_with_executions, invoke_cli, config):
        """Force mode should skip confirmation. Even if reingest_execution returns False
        (no output dirs), we exercise the CLI loop and get skip counts."""
        result = invoke_cli(["executions", "reingest", "--provider", "pmp", "--force"])
        assert result.exit_code == 0
        assert "Reingest complete" in result.stdout
        # Output dir doesn't exist so reingest_execution returns False -> skipped
        assert "skipped" in result.stdout

    def test_reingest_by_group_ids(self, db_with_executions, invoke_cli):
        """Passing group IDs directly should work."""
        eg = db_with_executions.session.query(ExecutionGroup).filter_by(key="reingest-1").first()
        result = invoke_cli(["executions", "reingest", str(eg.id), "--dry-run"])
        assert result.exit_code == 0
        assert "Dry run" in result.stdout
        assert "reingest-1" in result.stdout

    def test_reingest_include_failed(self, db_with_executions, invoke_cli):
        """--include-failed should include failed executions in preview."""
        result = invoke_cli(
            [
                "executions",
                "reingest",
                "--provider",
                "esmvaltool",
                "--include-failed",
                "--dry-run",
            ]
        )
        assert result.exit_code == 0
        assert "enso-characteristics" in result.stdout

    def test_reingest_success_path(self, db_with_executions, invoke_cli, config):
        """When reingest_execution succeeds, success_count should increment."""
        # Create output directory so reingest_execution can find it
        eg = db_with_executions.session.query(ExecutionGroup).filter_by(key="reingest-1").first()
        ex = eg.executions[0]
        results_dir = config.paths.results / ex.output_fragment
        results_dir.mkdir(parents=True, exist_ok=True)

        # Mock reingest_execution to return True (success)
        with patch("climate_ref.executor.reingest.reingest_execution", return_value=True):
            result = invoke_cli(["executions", "reingest", str(eg.id), "--force"])
        assert result.exit_code == 0
        assert "1 succeeded" in result.stdout
        assert "0 skipped" in result.stdout


class TestExecutionValues:
    """The ``executions values`` command, backed by the shared ``climate_ref.results`` layer."""

    _MODELS: ClassVar[dict[str, float]] = {
        "ACCESS-CM2": 10.0,
        "CESM3": 10.5,
        "MIROC6": 9.5,
        "MPI-ESM": 10.2,
        "NorESM": 9.8,
    }

    def _setup(self, db):
        """Seed one group / execution with six scalar tas values (one wild) and one series."""
        with db.session.begin():
            diagnostic = db.session.query(Diagnostic).first()
            assert diagnostic is not None
            group = ExecutionGroup(key="valkey", diagnostic_id=diagnostic.id, selectors={})
            db.session.add(group)
            db.session.flush()
            execution = Execution(
                execution_group_id=group.id,
                output_fragment="frag",
                dataset_hash="valhash",
                successful=True,
            )
            db.session.add(execution)
            db.session.flush()

            for source_id, value in self._MODELS.items():
                db.session.add(
                    ScalarMetricValue.build(
                        execution_id=execution.id,
                        value=value,
                        attributes=None,
                        dimensions={"statistic": "mean", "metric": "tas", "source_id": source_id},
                    )
                )
            db.session.add(
                ScalarMetricValue.build(
                    execution_id=execution.id,
                    value=1000.0,  # wild outlier
                    attributes=None,
                    dimensions={"statistic": "mean", "metric": "tas", "source_id": "WILD"},
                )
            )

            axis = SeriesIndex.get_or_create(db.session, "time", [0, 1, 2])
            db.session.add(
                SeriesMetricValue.build(
                    execution_id=execution.id,
                    values=[1.0, 2.0, 3.0],
                    index_axis=axis,
                    dimensions={"source_id": "ACCESS-CM2", "metric": "tas"},
                    attributes=None,
                )
            )
            db.session.flush()
            group_id = group.id
            execution_id = execution.id
        return group_id, execution_id

    def test_scalar_table(self, db_seeded, invoke_cli):
        group_id, _ = self._setup(db_seeded)

        result = invoke_cli(["executions", "values", str(group_id)])

        assert result.exit_code == 0
        # Dimension columns and the raw values are rendered; outliers shown by default.
        assert "source_id" in result.stdout
        assert "WILD" in result.stdout
        assert "1000.0" in result.stdout

    def test_scalar_json(self, db_seeded, invoke_cli):
        group_id, _ = self._setup(db_seeded)

        result = invoke_cli(["executions", "values", str(group_id), "--format", "json"])

        assert result.exit_code == 0
        records = json.loads(result.stdout)
        assert len(records) == len(self._MODELS) + 1  # models + wild
        assert any(r["value"] == 1000.0 for r in records)

    def test_scalar_outliers_hidden(self, db_seeded, invoke_cli):
        group_id, _ = self._setup(db_seeded)

        result = invoke_cli(["executions", "values", str(group_id), "--outliers"])

        assert result.exit_code == 0
        assert "WILD" not in result.stdout
        assert "flagged as outliers" in result.stderr

    def test_scalar_outliers_shown(self, db_seeded, invoke_cli):
        group_id, _ = self._setup(db_seeded)

        result = invoke_cli(["executions", "values", str(group_id), "--outliers", "--include-unverified"])

        assert result.exit_code == 0
        assert "WILD" in result.stdout

    def test_scalar_all_outliers_hidden_reports_outlier_count(self, db_seeded, invoke_cli):
        """When every value is flagged as an outlier the empty-page message must still surface
        that outliers exist, instead of silently reporting "No scalar values found."."""
        with db_seeded.session.begin():
            diagnostic = db_seeded.session.query(Diagnostic).first()
            assert diagnostic is not None
            group = ExecutionGroup(key="allwild", diagnostic_id=diagnostic.id, selectors={})
            db_seeded.session.add(group)
            db_seeded.session.flush()
            execution = Execution(
                execution_group_id=group.id,
                output_fragment="frag-wild",
                dataset_hash="wildhash",
                successful=True,
            )
            db_seeded.session.add(execution)
            db_seeded.session.flush()
            # A non-finite value is always flagged as an outlier, regardless of group size.
            db_seeded.session.add(
                ScalarMetricValue.build(
                    execution_id=execution.id,
                    value=float("nan"),
                    attributes=None,
                    dimensions={"statistic": "mean", "metric": "tas", "source_id": "ONLY-MODEL"},
                )
            )
            db_seeded.session.flush()
            group_id = group.id

        result = invoke_cli(["executions", "values", str(group_id), "--outliers"])

        assert result.exit_code == 0
        assert "No scalar values found" in result.stderr
        assert "1" in result.stderr
        assert "outlier" in result.stderr
        assert "--include-unverified" in result.stderr

    def test_values_default_limit_caps_output(self, db_seeded, invoke_cli):
        """Test that limit defaults to 100 and caps output when more rows exist."""
        with db_seeded.session.begin():
            diagnostic = db_seeded.session.query(Diagnostic).first()
            assert diagnostic is not None
            group = ExecutionGroup(key="biggroup", diagnostic_id=diagnostic.id, selectors={})
            db_seeded.session.add(group)
            db_seeded.session.flush()
            execution = Execution(
                execution_group_id=group.id,
                output_fragment="frag-big",
                dataset_hash="bighash",
                successful=True,
            )
            db_seeded.session.add(execution)
            db_seeded.session.flush()
            # Create 150 scalar values to exceed default limit of 100
            for i in range(150):
                db_seeded.session.add(
                    ScalarMetricValue.build(
                        execution_id=execution.id,
                        value=float(i),
                        attributes=None,
                        dimensions={"statistic": "mean", "metric": "tas", "source_id": f"MODEL-{i}"},
                    )
                )
            db_seeded.session.flush()
            group_id = group.id

        result = invoke_cli(["executions", "values", str(group_id)])

        assert result.exit_code == 0
        # Should display 100 values by default
        assert "Displaying 100 of 150" in result.stderr
        assert "--limit / --offset" in result.stderr

    def test_series_with_outlier_flags_warns(self, db_seeded, invoke_cli):
        """Test that --outliers and --include-unverified warn when used with --kind series."""
        group_id, _ = self._setup(db_seeded)

        result = invoke_cli(["executions", "values", str(group_id), "--kind", "series", "--outliers"])

        assert result.exit_code == 0
        assert "--outliers only apply to scalar values and were ignored." in result.stderr

    def test_series_with_include_unverified_warns(self, db_seeded, invoke_cli):
        """Test that --include-unverified warns when used with --kind series."""
        group_id, _ = self._setup(db_seeded)

        result = invoke_cli(
            ["executions", "values", str(group_id), "--kind", "series", "--include-unverified"]
        )

        assert result.exit_code == 0
        assert "--include-unverified only apply to scalar values and were ignored." in result.stderr

    def test_series_with_both_flags_warns(self, db_seeded, invoke_cli):
        """Test that both flags warn together when used with --kind series."""
        group_id, _ = self._setup(db_seeded)

        result = invoke_cli(
            [
                "executions",
                "values",
                str(group_id),
                "--kind",
                "series",
                "--outliers",
                "--include-unverified",
            ]
        )

        assert result.exit_code == 0
        assert "--outliers/--include-unverified only apply to scalar values and were ignored." in (
            result.stderr
        )

    def test_dimension_filter(self, db_seeded, invoke_cli):
        group_id, _ = self._setup(db_seeded)

        result = invoke_cli(["executions", "values", str(group_id), "-d", "source_id=WILD"])

        assert result.exit_code == 0
        assert "WILD" in result.stdout
        assert "ACCESS-CM2" not in result.stdout

    def test_unknown_dimension(self, db_seeded, invoke_cli):
        group_id, _ = self._setup(db_seeded)

        result = invoke_cli(
            ["executions", "values", str(group_id), "-d", "not_a_dim=1"], expected_exit_code=1
        )

        assert "Unknown dimension" in result.stderr

    def test_series(self, db_seeded, invoke_cli):
        group_id, _ = self._setup(db_seeded)

        result = invoke_cli(["executions", "values", str(group_id), "--kind", "series"])

        assert result.exit_code == 0
        # The terminal shows a compact per-series summary rather than raw arrays.
        assert "n_points" in result.stdout
        assert "3" in result.stdout

    def test_series_json(self, db_seeded, invoke_cli):
        group_id, _ = self._setup(db_seeded)

        result = invoke_cli(["executions", "values", str(group_id), "--kind", "series", "--format", "json"])

        assert result.exit_code == 0
        # JSON output is the long-form, exploded shape: one record per (series, index point).
        records = json.loads(result.stdout)
        assert len(records) == 3
        assert all(r["source_id"] == "ACCESS-CM2" for r in records)
        assert [r["value"] for r in records] == [1.0, 2.0, 3.0]

    def test_specific_execution_wrong_group(self, db_seeded, invoke_cli):
        _group_id, execution_id = self._setup(db_seeded)
        with db_seeded.session.begin():
            other = ExecutionGroup(key="othergroup", diagnostic_id=1)
            db_seeded.session.add(other)
            db_seeded.session.flush()
            other_id = other.id

        result = invoke_cli(
            ["executions", "values", str(other_id), "--execution-id", str(execution_id)],
            expected_exit_code=1,
        )

        assert "does not belong" in result.stderr

    def test_missing_group(self, db_seeded, invoke_cli):
        result = invoke_cli(["executions", "values", "99999"], expected_exit_code=1)

        assert "Execution group not found" in result.stderr
