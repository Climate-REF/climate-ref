"""add solve table

Adds a ``solve`` table recording each solve,
and nullable links from ``execution`` and ``execution_group`` to the solve that created or reaped them.

Revision ID: cd446510084f
Revises: 42598155c23b
Create Date: 2026-10-16 12:01:11.626006

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "cd446510084f"
down_revision: str | None = "42598155c23b"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create ``solve`` and link executions and execution groups to it."""
    op.create_table(
        "solve",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column(
            "status",
            sa.Enum(
                "running", "succeeded", "degraded", "failed", name="solvestatus", native_enum=False, length=16
            ),
            nullable=False,
        ),
        sa.Column(
            "stage",
            sa.Enum(
                "preflight", "solving", "queuing", "waiting", name="solvestage", native_enum=False, length=16
            ),
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(), nullable=False),
        sa.Column("finished_at", sa.DateTime(), nullable=True),
        sa.Column("error", sa.String(), nullable=True),
        sa.Column("options", sa.JSON(), nullable=False),
        sa.Column("skipped_providers", sa.JSON(), nullable=False),
        sa.Column("ref_version", sa.String(), nullable=False),
        sa.Column("hostname", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.Column("updated_at", sa.DateTime(), server_default=sa.text("(CURRENT_TIMESTAMP)"), nullable=False),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_solve")),
    )
    with op.batch_alter_table("solve", schema=None) as batch_op:
        batch_op.create_index(batch_op.f("ix_solve_started_at"), ["started_at"], unique=False)
        batch_op.create_index(batch_op.f("ix_solve_status"), ["status"], unique=False)
        batch_op.create_index(batch_op.f("ix_solve_updated_at"), ["updated_at"], unique=False)

    with op.batch_alter_table("execution", schema=None) as batch_op:
        batch_op.add_column(sa.Column("solve_id", sa.Integer(), nullable=True))
        batch_op.add_column(sa.Column("reaped_by_solve_id", sa.Integer(), nullable=True))
        batch_op.create_index(
            batch_op.f("ix_execution_reaped_by_solve_id"), ["reaped_by_solve_id"], unique=False
        )
        batch_op.create_index(batch_op.f("ix_execution_solve_id"), ["solve_id"], unique=False)
        batch_op.create_foreign_key(batch_op.f("fk_execution_solve_id_solve"), "solve", ["solve_id"], ["id"])
        batch_op.create_foreign_key(
            batch_op.f("fk_execution_reaped_by_solve_id_solve"), "solve", ["reaped_by_solve_id"], ["id"]
        )

    with op.batch_alter_table("execution_group", schema=None) as batch_op:
        batch_op.add_column(sa.Column("created_by_solve_id", sa.Integer(), nullable=True))
        batch_op.create_index(
            batch_op.f("ix_execution_group_created_by_solve_id"), ["created_by_solve_id"], unique=False
        )
        batch_op.create_foreign_key(
            batch_op.f("fk_execution_group_created_by_solve_id_solve"),
            "solve",
            ["created_by_solve_id"],
            ["id"],
        )


def downgrade() -> None:
    """Drop the links to ``solve`` and the table itself."""
    with op.batch_alter_table("execution_group", schema=None) as batch_op:
        batch_op.drop_constraint(
            batch_op.f("fk_execution_group_created_by_solve_id_solve"), type_="foreignkey"
        )
        batch_op.drop_index(batch_op.f("ix_execution_group_created_by_solve_id"))
        batch_op.drop_column("created_by_solve_id")

    with op.batch_alter_table("execution", schema=None) as batch_op:
        batch_op.drop_constraint(batch_op.f("fk_execution_reaped_by_solve_id_solve"), type_="foreignkey")
        batch_op.drop_constraint(batch_op.f("fk_execution_solve_id_solve"), type_="foreignkey")
        batch_op.drop_index(batch_op.f("ix_execution_solve_id"))
        batch_op.drop_index(batch_op.f("ix_execution_reaped_by_solve_id"))
        batch_op.drop_column("reaped_by_solve_id")
        batch_op.drop_column("solve_id")

    with op.batch_alter_table("solve", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_solve_updated_at"))
        batch_op.drop_index(batch_op.f("ix_solve_status"))
        batch_op.drop_index(batch_op.f("ix_solve_started_at"))

    op.drop_table("solve")
