"""migration for obs4mips

Revision ID: 88c914704207
Revises: c1818a18d87f
Create Date: 2025-02-27 10:34:55.504822

"""

from collections.abc import Sequence
from typing import Union

# revision identifiers, used by Alembic.
revision: str = "88c914704207"
down_revision: Union[str, None] = "c1818a18d87f"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    pass


def downgrade() -> None:
    pass
