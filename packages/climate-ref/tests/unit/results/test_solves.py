"""Unit tests for the solve read layer."""

import datetime

import pytest
from sqlalchemy import update

from climate_ref.models import Dataset, Execution, ExecutionGroup, Solve, SolveStage, SolveStatus
from climate_ref.models.diagnostic import Diagnostic
from climate_ref.models.execution import ExecutionFailureReason
from climate_ref.results import Reader

T0 = datetime.datetime(2026, 1, 1)
T1 = datetime.datetime(2026, 2, 1)
T2 = datetime.datetime(2026, 3, 1)


def _solve(started_at: datetime.datetime, **kwargs) -> Solve:
    defaults = dict(
        status=SolveStatus.Succeeded,
        stage=SolveStage.Waiting,
        started_at=started_at,
        options={"limit": None},
        skipped_providers=[],
        ref_version="1.2.3",
        hostname="host",
    )
    return Solve(**{**defaults, **kwargs})


@pytest.fixture
def db_with_solves(db_seeded):
    """Two solves, with one dataset created and another retracted between them."""
    with db_seeded.session.begin():
        first = _solve(T1)
        second = _solve(
            T2,
            status=SolveStatus.Degraded,
            skipped_providers=["pmp"],
        )
        db_seeded.session.add_all([first, second])
        db_seeded.session.flush()

        dataset_ids = [d.id for d in db_seeded.session.query(Dataset).order_by(Dataset.id).all()]
        db_seeded.session.execute(update(Dataset).values(created_at=T0))
        db_seeded.session.execute(
            update(Dataset).where(Dataset.id == dataset_ids[0]).values(created_at=T1 + datetime.timedelta(1))
        )
        db_seeded.session.execute(update(Dataset).where(Dataset.id == dataset_ids[1]).values(retracted_at=T1))

        diagnostic = db_seeded.session.query(Diagnostic).first()
        group = ExecutionGroup(
            key="key", diagnostic_id=diagnostic.id, selectors={}, created_by_solve_id=second.id
        )
        db_seeded.session.add(group)
        outcomes = [
            (True, None),
            (True, None),
            (False, ExecutionFailureReason.DiagnosticError.value),
            (False, None),
            (None, None),
        ]
        for i, (successful, reason) in enumerate(outcomes):
            db_seeded.session.add(
                Execution(
                    execution_group=group,
                    dataset_hash=str(i),
                    output_fragment=str(i),
                    successful=successful,
                    failure_reason=reason,
                    solve_id=second.id,
                )
            )
        db_seeded.session.add(
            Execution(
                execution_group=group,
                dataset_hash="reaped",
                output_fragment="reaped",
                successful=False,
                failure_reason=ExecutionFailureReason.Abandoned.value,
                reaped_by_solve_id=second.id,
            )
        )
    return db_seeded, first.id, second.id, dataset_ids


def test_list_most_recent_first(db_with_solves):
    db, first_id, second_id, _ = db_with_solves

    collection = Reader(db).solves.list()

    assert [s.id for s in collection] == [second_id, first_id]
    assert collection.total_count == 2
    df = collection.to_pandas()
    assert df["status"].tolist() == ["degraded", "succeeded"]


def test_list_limit(db_with_solves):
    db, _, second_id, _ = db_with_solves

    collection = Reader(db).solves.list(limit=1)

    assert [s.id for s in collection] == [second_id]
    assert collection.total_count == 2


def test_list_empty_has_columns(db_seeded):
    df = Reader(db_seeded).solves.list().to_pandas()

    assert df.empty
    assert "status" in df.columns


def test_get(db_with_solves):
    db, _, second_id, _ = db_with_solves
    reader = Reader(db).solves

    solve = reader.get(second_id)

    assert solve.status == SolveStatus.Degraded
    assert solve.skipped_providers == ("pmp",)
    assert reader.get(9999) is None


def test_outcomes(db_with_solves):
    db, first_id, second_id, _ = db_with_solves
    reader = Reader(db).solves

    outcomes = reader.outcomes(second_id)

    assert (outcomes.succeeded, outcomes.failed, outcomes.pending) == (2, 2, 1)
    assert outcomes.total == 5
    assert outcomes.failure_reasons == {"diagnostic_error": 1, "unknown": 1}
    assert reader.outcomes(first_id).total == 0


def test_executions_groups_and_reaped(db_with_solves):
    db, first_id, second_id, _ = db_with_solves
    reader = Reader(db).solves

    assert len(reader.executions(second_id)) == 5
    assert reader.executions(first_id) == ()
    groups = reader.groups(second_id)
    assert [g.key for g in groups] == ["key"]
    assert groups[0].latest is not None
    reaped = reader.reaped_executions(second_id)
    assert [e.dataset_hash for e in reaped] == ["reaped"]


def test_window(db_with_solves):
    db, first_id, second_id, _ = db_with_solves
    reader = Reader(db).solves

    assert reader.window(first_id).start is None
    assert reader.window(first_id).end == T1
    assert reader.window(second_id).start == T1
    assert reader.window(second_id).end == T2
    assert reader.window(9999) is None


def test_first_solve_sees_every_earlier_dataset(db_with_solves):
    db, first_id, _, dataset_ids = db_with_solves
    reader = Reader(db).solves

    new = reader.new_datasets(first_id)

    assert new.total_count == len(dataset_ids) - 1
    assert dataset_ids[0] not in {d.id for d in new}
    assert reader.retracted_datasets(first_id).total_count == 0


def test_later_solve_sees_changes_since_previous_solve(db_with_solves):
    db, _, second_id, dataset_ids = db_with_solves
    reader = Reader(db).solves

    assert [d.id for d in reader.new_datasets(second_id)] == [dataset_ids[0]]
    assert [d.id for d in reader.retracted_datasets(second_id)] == [dataset_ids[1]]


def test_window_datasets_limit(db_with_solves):
    db, first_id, _, dataset_ids = db_with_solves

    new = Reader(db).solves.new_datasets(first_id, limit=2)

    assert len(new) == 2
    assert new.total_count == len(dataset_ids) - 1


def test_window_datasets_missing_solve(db_seeded):
    assert Reader(db_seeded).solves.new_datasets(9999).total_count == 0
