"""add obs4REF and ESMValTool reference dataset subtypes

Creates the two new polymorphic subtype tables, ``obs4ref_dataset`` and
``esmvaltool_reference_dataset``, and registers their ``SourceDatasetType`` labels.

``obs4ref_dataset`` carries the same column set as ``obs4mips_dataset`` and
``pmp_climatology_dataset``, via the shared ``ReferenceDatasetMixin``.
``esmvaltool_reference_dataset`` has its own smaller set, because ESMValTool reference data is
not CMOR/obs4MIPs compliant and only exposes what its DRS path and filename encode.

That table stores ``frequency`` and no MIP table. ``native6`` data carries a frequency and never
a table, while ``OBS``/``OBS6`` carry a table that reduces to one (``Amon`` -> ``mon``), so
frequency is the only axis both layouts share. It is non-nullable because it forms part of the
dataset identity: without it, monthly and daily ERA5 ``tas`` would share an ``instance_id``.

Dialect differences:

* PostgreSQL backs ``dataset_type`` with a native ``sourcedatasettype`` enum, so each new label
  needs an explicit ``ALTER TYPE ... ADD VALUE``. ``ADD VALUE`` is permitted inside a transaction
  on PostgreSQL >= 12 so long as the new label is not *used* before commit, and this migration
  never inserts a row of either type.
* SQLite stores the enum as a ``VARCHAR`` sized to the longest member name.
  ``ESMValToolReference`` (19 characters) is longer than the previous maximum
  ``PMPClimatology`` (14), so the column is widened to match.


Revision ID: e1f2a3b4c5d6
Revises: b7c8d9e0a1f2
Create Date: 2026-07-10
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "e1f2a3b4c5d6"
down_revision: str | None = "b7c8d9e0a1f2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_PREVIOUS_MEMBERS = ("CMIP6", "CMIP7", "obs4MIPs", "PMPClimatology")
_NEW_MEMBERS = ("obs4REF", "ESMValToolReference")


def upgrade() -> None:
    if op.get_bind().dialect.name == "postgresql":
        for member in _NEW_MEMBERS:
            op.execute(f"ALTER TYPE sourcedatasettype ADD VALUE IF NOT EXISTS '{member}'")
    else:
        # Widen the VARCHAR to fit ``ESMValToolReference`` (19 > 14).
        with op.batch_alter_table("dataset", schema=None) as batch_op:
            batch_op.alter_column(
                "dataset_type",
                existing_type=sa.String(length=14),
                type_=sa.Enum(*_PREVIOUS_MEMBERS, *_NEW_MEMBERS, name="sourcedatasettype"),
                existing_nullable=False,
            )

    op.create_table(
        "obs4ref_dataset",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("activity_id", sa.String(), nullable=False),
        sa.Column("frequency", sa.String(), nullable=False),
        sa.Column("grid", sa.String(), nullable=False),
        sa.Column("grid_label", sa.String(), nullable=False),
        sa.Column("institution_id", sa.String(), nullable=False),
        sa.Column("long_name", sa.String(), nullable=False),
        sa.Column("nominal_resolution", sa.String(), nullable=False),
        sa.Column("realm", sa.String(), nullable=False),
        sa.Column("product", sa.String(), nullable=False),
        sa.Column("source_id", sa.String(), nullable=False),
        sa.Column("source_type", sa.String(), nullable=False),
        sa.Column("units", sa.String(), nullable=False),
        sa.Column("variable_id", sa.String(), nullable=False),
        sa.Column("variant_label", sa.String(), nullable=False),
        sa.Column("version", sa.String(), nullable=False),
        sa.Column("vertical_levels", sa.Integer(), nullable=False),
        sa.Column("source_version_number", sa.String(), nullable=False),
        sa.Column("instance_id", sa.String(), nullable=False),
        sa.ForeignKeyConstraint(["id"], ["dataset.id"], name=op.f("fk_obs4ref_dataset_id_dataset")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_obs4ref_dataset")),
    )

    op.create_table(
        "esmvaltool_reference_dataset",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("project", sa.String(), nullable=False),
        sa.Column("source_id", sa.String(), nullable=False),
        sa.Column("variable_id", sa.String(), nullable=False),
        sa.Column("frequency", sa.String(), nullable=False),
        sa.Column("version", sa.String(), nullable=False),
        sa.Column("data_type", sa.String(), nullable=True),
        sa.Column("tier", sa.Integer(), nullable=True),
        sa.Column("long_name", sa.String(), nullable=True),
        sa.Column("units", sa.String(), nullable=True),
        sa.Column("instance_id", sa.String(), nullable=False),
        sa.ForeignKeyConstraint(
            ["id"], ["dataset.id"], name=op.f("fk_esmvaltool_reference_dataset_id_dataset")
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_esmvaltool_reference_dataset")),
    )
    with op.batch_alter_table("esmvaltool_reference_dataset", schema=None) as batch_op:
        batch_op.create_index(
            batch_op.f("ix_esmvaltool_reference_dataset_frequency"), ["frequency"], unique=False
        )
        batch_op.create_index(
            batch_op.f("ix_esmvaltool_reference_dataset_instance_id"), ["instance_id"], unique=False
        )
        batch_op.create_index(
            batch_op.f("ix_esmvaltool_reference_dataset_project"), ["project"], unique=False
        )
        batch_op.create_index(
            batch_op.f("ix_esmvaltool_reference_dataset_source_id"), ["source_id"], unique=False
        )
        batch_op.create_index(
            batch_op.f("ix_esmvaltool_reference_dataset_variable_id"), ["variable_id"], unique=False
        )


def downgrade() -> None:
    with op.batch_alter_table("esmvaltool_reference_dataset", schema=None) as batch_op:
        batch_op.drop_index(batch_op.f("ix_esmvaltool_reference_dataset_variable_id"))
        batch_op.drop_index(batch_op.f("ix_esmvaltool_reference_dataset_source_id"))
        batch_op.drop_index(batch_op.f("ix_esmvaltool_reference_dataset_project"))
        batch_op.drop_index(batch_op.f("ix_esmvaltool_reference_dataset_instance_id"))
        batch_op.drop_index(batch_op.f("ix_esmvaltool_reference_dataset_frequency"))
    op.drop_table("esmvaltool_reference_dataset")

    op.drop_table("obs4ref_dataset")

    if op.get_bind().dialect.name != "postgresql":
        # On PostgreSQL the two new labels are intentionally left as the enum type cannot drop a value.
        with op.batch_alter_table("dataset", schema=None) as batch_op:
            batch_op.alter_column(
                "dataset_type",
                existing_type=sa.Enum(*_PREVIOUS_MEMBERS, *_NEW_MEMBERS, name="sourcedatasettype"),
                type_=sa.Enum(*_PREVIOUS_MEMBERS, name="sourcedatasettype"),
                existing_nullable=False,
            )
