"""Latest-execution-per-group ranking semantics.

Covers the reusable ranking primitive behind ``get_execution_group_and_latest``:

1. Deterministic tie-break -- two executions sharing ``created_at`` yield exactly
   one group row (not a duplicate, as the old ``max(created_at)`` equijoin did).
2. ``successful`` (post-rank) vs ``latest_successful`` (pre-rank) answer different
   questions and diverge for a group whose newest run failed after an earlier success.
"""

import datetime

from climate_ref.database import Database
from climate_ref.models.diagnostic import Diagnostic
from climate_ref.models.execution import (
    Execution,
    ExecutionGroup,
    get_execution_group_and_latest,
    get_execution_group_and_latest_filtered,
)

_T0 = datetime.datetime(2024, 1, 1, 0, 0, 0)


def _add_execution(
    db: Database, group_id: int, *, successful: bool | None, offset: int, tag: str
) -> Execution:
    ex = Execution(
        execution_group_id=group_id,
        successful=successful,
        output_fragment=f"out-{tag}",
        dataset_hash="h",
        created_at=_T0 + datetime.timedelta(minutes=offset),
    )
    db.session.add(ex)
    db.session.flush()
    return ex


def _make_group(db: Database) -> ExecutionGroup:
    diag = db.session.query(Diagnostic).first()
    eg = ExecutionGroup(key="ranking-key", diagnostic_id=diag.id, selectors={}, dirty=False)
    db.session.add(eg)
    db.session.flush()
    return eg


def test_tie_break_is_deterministic_single_winner(db_seeded: Database) -> None:
    """Two executions with an identical created_at must not duplicate the group row."""
    with db_seeded.session.begin():
        eg = _make_group(db_seeded)
        # Same timestamp on purpose: the old max(created_at) join would emit two rows.
        _add_execution(db_seeded, eg.id, successful=True, offset=0, tag="a")
        _add_execution(db_seeded, eg.id, successful=False, offset=0, tag="b")

    rows = [(g, e) for g, e in get_execution_group_and_latest(db_seeded.session).all() if g.id == eg.id]
    assert len(rows) == 1, "tie on created_at must still yield exactly one group row"
    # Highest id wins the tie -> execution 'b'.
    assert rows[0][1].output_fragment == "out-b"


def test_post_rank_vs_pre_rank_successful_diverge(db_seeded: Database) -> None:
    """A group whose newest run failed after an earlier success:

    - successful=True (post-rank): excluded, because the *latest* run failed.
    - latest_successful=True (pre-rank): included, returning the earlier success.
    """
    with db_seeded.session.begin():
        eg = _make_group(db_seeded)
        _add_execution(db_seeded, eg.id, successful=True, offset=0, tag="ok")
        _add_execution(db_seeded, eg.id, successful=False, offset=10, tag="fail")

    post = [
        (g, e)
        for g, e in get_execution_group_and_latest_filtered(db_seeded.session, successful=True)
        if g.id == eg.id
    ]
    assert post == [], "post-rank successful=True must drop a group whose latest run failed"

    pre = [
        (g, e)
        for g, e in get_execution_group_and_latest_filtered(db_seeded.session, latest_successful=True)
        if g.id == eg.id
    ]
    assert len(pre) == 1, "pre-rank latest_successful=True must surface the earlier successful run"
    assert pre[0][1].output_fragment == "out-ok"


def test_pre_rank_default_matches_latest_regardless_of_success(db_seeded: Database) -> None:
    """With no pre-filter, the winner is simply the newest run (here, the failure)."""
    with db_seeded.session.begin():
        eg = _make_group(db_seeded)
        _add_execution(db_seeded, eg.id, successful=True, offset=0, tag="ok")
        _add_execution(db_seeded, eg.id, successful=False, offset=10, tag="fail")

    rows = [(g, e) for g, e in get_execution_group_and_latest_filtered(db_seeded.session) if g.id == eg.id]
    assert len(rows) == 1
    assert rows[0][1].output_fragment == "out-fail"
