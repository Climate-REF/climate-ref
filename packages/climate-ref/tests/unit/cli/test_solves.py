import datetime
import json

import pytest
from sqlalchemy import update

from climate_ref.models import Dataset, Execution, ExecutionGroup, Solve, SolveStage, SolveStatus
from climate_ref.models.diagnostic import Diagnostic
from climate_ref.models.execution import ExecutionFailureReason


@pytest.fixture
def db_with_solve(db_seeded):
    """A failed solve that created two executions, after every seeded dataset was created."""
    with db_seeded.session.begin():
        solve = Solve(
            status=SolveStatus.Failed,
            stage=SolveStage.Queuing,
            started_at=datetime.datetime(2026, 2, 1),
            finished_at=datetime.datetime(2026, 2, 1, 1),
            error="broker down",
            options={"limit": 3},
            skipped_providers=["pmp"],
            ref_version="1.2.3",
            hostname="solve-host",
        )
        db_seeded.session.add(solve)
        db_seeded.session.flush()
        db_seeded.session.execute(update(Dataset).values(created_at=datetime.datetime(2026, 1, 1)))

        diagnostic = db_seeded.session.query(Diagnostic).first()
        group = ExecutionGroup(
            key="key", diagnostic_id=diagnostic.id, selectors={}, created_by_solve_id=solve.id
        )
        db_seeded.session.add_all(
            [
                group,
                Execution(
                    execution_group=group,
                    dataset_hash="a",
                    output_fragment="a",
                    successful=True,
                    solve_id=solve.id,
                ),
                Execution(
                    execution_group=group,
                    dataset_hash="b",
                    output_fragment="b",
                    successful=False,
                    failure_reason=ExecutionFailureReason.NotStarted.value,
                    solve_id=solve.id,
                ),
            ]
        )
    return db_seeded, solve.id


def test_solves_help(invoke_cli):
    result = invoke_cli(["solves", "--help"])

    assert "View past solves" in result.stdout


def test_list(db_with_solve, invoke_cli):
    _, solve_id = db_with_solve

    result = invoke_cli(["solves", "list", "--format", "json"])

    payload = json.loads(result.stdout)
    assert [row["id"] for row in payload] == [solve_id]
    assert payload[0]["status"] == "failed"
    assert payload[0]["stage"] == "queuing"
    assert payload[0]["skipped_providers"] == ["pmp"]


def test_list_table(db_with_solve, invoke_cli):
    result = invoke_cli(["solves", "list"])

    assert "failed" in result.stdout
    assert "queuing" in result.stdout


def test_inspect(db_with_solve, invoke_cli):
    db, solve_id = db_with_solve
    dataset_count = db.session.query(Dataset).count()

    result = invoke_cli(["solves", "inspect", str(solve_id), "--limit", "2"])

    stdout = result.stdout
    assert "broker down" in stdout
    assert "queuing" in stdout
    assert "solve-host" in stdout
    assert "1.2.3" in stdout
    assert "pmp" in stdout
    assert '"limit": 3' in stdout
    assert "Succeeded: 1" in stdout
    assert "Failed: 1 (not_started: 1)" in stdout
    assert "Pending: 0" in stdout
    assert "Executions (2)" in stdout
    assert "succeeded" in stdout
    assert "failed (not_started)" in stdout
    assert f"New datasets ({dataset_count})" in stdout
    assert f"Showing 2 of {dataset_count}" in stdout
    assert "Retracted datasets" in stdout


def test_inspect_limits_executions(db_with_solve, invoke_cli):
    _, solve_id = db_with_solve

    result = invoke_cli(["solves", "inspect", str(solve_id), "--limit", "1"])

    assert "Showing 1 of 2" in result.stdout


def test_inspect_missing(db_seeded, invoke_cli):
    invoke_cli(["solves", "inspect", "9999"], expected_exit_code=1)
