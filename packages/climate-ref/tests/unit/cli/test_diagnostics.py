import json

import pytest
from climate_ref_esmvaltool import provider as esmvaltool_provider
from climate_ref_pmp import provider as pmp_provider

from climate_ref.models.diagnostic import Diagnostic
from climate_ref.models.execution import ExecutionGroup
from climate_ref.provider_registry import _register_provider


@pytest.fixture
def db_with_diagnostics(db_seeded):
    """A seeded DB with two providers and execution groups spread unevenly across diagnostics."""
    with db_seeded.session.begin():
        _register_provider(db_seeded, pmp_provider)
        _register_provider(db_seeded, esmvaltool_provider)

        diag_1 = db_seeded.session.query(Diagnostic).filter_by(slug="enso_tel").first()
        diag_2 = (
            db_seeded.session.query(Diagnostic)
            .filter_by(slug="extratropical-modes-of-variability-nao")
            .first()
        )

        db_seeded.session.add(ExecutionGroup(key="key1", diagnostic_id=diag_1.id, selectors={}))
        db_seeded.session.add(ExecutionGroup(key="key2", diagnostic_id=diag_1.id, selectors={}))
        db_seeded.session.add(ExecutionGroup(key="key3", diagnostic_id=diag_2.id, selectors={}))

    db_seeded.session.commit()
    return db_seeded


def test_diagnostics_help(invoke_cli):
    result = invoke_cli(["diagnostics", "--help"])

    assert "View diagnostic metadata" in result.stdout


class TestDiagnosticsList:
    def test_list(self, db_with_diagnostics, invoke_cli):
        result = invoke_cli(["diagnostics", "list"])

        assert "enso_tel" in result.stdout
        assert "extratropical-modes-of-variability-nao" in result.stdout
        assert "pmp" in result.stdout
        assert "execution_group_count" in result.stdout

    def test_list_shows_counts(self, db_with_diagnostics, invoke_cli):
        result = invoke_cli(["diagnostics", "list", "--format", "json"])

        payload = json.loads(result.stdout)
        by_slug = {row["diagnostic"]: row for row in payload}
        assert by_slug["enso_tel"]["execution_group_count"] == 2
        assert by_slug["extratropical-modes-of-variability-nao"]["execution_group_count"] == 1
        # No executions seeded, so promoted-version groups are counted in `total` only.
        assert by_slug["enso_tel"]["total"] == 2
        assert by_slug["enso_tel"]["successful"] == 0
        assert by_slug["enso_tel"]["inflight"] == 0

    def test_list_help_documents_columns(self, invoke_cli):
        result = invoke_cli(["diagnostics", "list", "--help"])

        for column in ("execution_group_count", "successful", "inflight", "total"):
            assert column in result.stdout

    def test_filter_by_provider(self, db_with_diagnostics, invoke_cli):
        result = invoke_cli(["diagnostics", "list", "--provider", "pmp"])

        assert "pmp" in result.stdout
        assert "esmvaltool" not in result.stdout

    def test_filter_by_diagnostic(self, db_with_diagnostics, invoke_cli):
        result = invoke_cli(["diagnostics", "list", "--diagnostic", "enso"])

        assert "enso_tel" in result.stdout
        assert "extratropical-modes-of-variability-nao" not in result.stdout

    def test_list_limit(self, db_with_diagnostics, invoke_cli):
        result = invoke_cli(["diagnostics", "list", "--limit", "1", "--format", "json"])

        payload = json.loads(result.stdout)
        assert len(payload) == 1

    def test_list_columns(self, db_with_diagnostics, invoke_cli):
        result = invoke_cli(["diagnostics", "list", "--column", "diagnostic", "--column", "provider"])

        assert "enso_tel" in result.stdout
        assert "provider" in result.stdout
        assert "promoted_version" not in result.stdout

    def test_list_columns_missing(self, db_with_diagnostics, invoke_cli):
        invoke_cli(
            ["diagnostics", "list", "--column", "diagnostic", "--column", "missing"],
            expected_exit_code=1,
        )

    def test_list_json_empty(self, db_seeded, invoke_cli):
        result = invoke_cli(["diagnostics", "list", "--diagnostic", "nonexistent", "--format", "json"])

        assert json.loads(result.stdout) == []

    def test_list_columns_missing_on_empty_results(self, db_seeded, invoke_cli):
        """An unknown ``--column`` must still be rejected when the filtered result set is empty --
        ``to_pandas()`` always emits explicit columns, so there is no reason to skip validation."""
        invoke_cli(
            ["diagnostics", "list", "--diagnostic", "nonexistent", "--column", "missing"],
            expected_exit_code=1,
        )
