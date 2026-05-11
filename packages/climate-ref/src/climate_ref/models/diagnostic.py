from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, UniqueConstraint, func, select
from sqlalchemy.orm import Mapped, Session, mapped_column, relationship

from climate_ref.models.base import Base
from climate_ref.models.mixins import CreatedUpdatedMixin

if TYPE_CHECKING:
    from climate_ref.models.execution import ExecutionGroup
    from climate_ref.models.provider import Provider


class Diagnostic(CreatedUpdatedMixin, Base):
    """
    Represents a diagnostic that can be calculated
    """

    __tablename__ = "diagnostic"
    __table_args__ = (UniqueConstraint("provider_id", "slug", name="diagnostic_ident"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    slug: Mapped[str] = mapped_column()
    """
    Unique identifier for the diagnostic

    This will be used to reference the diagnostic in the benchmarking process
    """

    name: Mapped[str] = mapped_column()
    """
    Long name of the diagnostic
    """

    provider_id: Mapped[int] = mapped_column(ForeignKey("provider.id"))
    """
    The provider that provides the diagnostic
    """

    enabled: Mapped[bool] = mapped_column(default=True)
    """
    Whether the diagnostic is enabled or not

    If a diagnostic is not enabled, it will not be used for any calculations.
    """

    promoted_version: Mapped[int] = mapped_column(default=1, server_default="1")
    """
    Currently promoted diagnostic version for default queries.

    Default query helpers filter ``ExecutionGroup.diagnostic_version == Diagnostic.promoted_version``
    so consumers see exactly one version's worth of results.
    Recomputed as ``max(ExecutionGroup.diagnostic_version)`` after a new group is inserted
    (see ``recompute_promoted_version``).
    """

    provider: Mapped["Provider"] = relationship(back_populates="diagnostics")
    execution_groups: Mapped[list["ExecutionGroup"]] = relationship(back_populates="diagnostic")

    def __repr__(self) -> str:
        return f"<Metric slug={self.slug}>"

    def full_slug(self) -> str:
        """
        Get the full slug of the diagnostic, including the provider slug

        Returns
        -------
        str
            Full slug of the diagnostic
        """
        return f"{self.provider.slug}/{self.slug}"


def recompute_promoted_version(diagnostic_id: int, session: Session) -> int:
    """
    Recompute ``Diagnostic.promoted_version`` as ``max(ExecutionGroup.diagnostic_version)``.

    The largest ``diagnostic_version`` seen across this diagnostic's groups
    becomes the version that default queries return.
    A v2 group with zero successful executions still becomes the promoted version.

    This helper is called explicitly from the solver after a new ``ExecutionGroup`` is inserted.
    It is not a SQLAlchemy event listener because event-driven recomputation is
    harder to reason about during flush and harder to test in isolation.

    If the diagnostic has no execution groups yet, ``promoted_version`` is left at its default of 1.

    Parameters
    ----------
    diagnostic_id
        Primary key of the diagnostic to recompute.
    session
        SQLAlchemy session.  The caller is responsible for committing.

    Returns
    -------
    :
        The newly-stored ``promoted_version`` value.
    """
    # Avoid a circular import: ExecutionGroup's module imports Diagnostic at module load.
    from climate_ref.models.execution import ExecutionGroup  # noqa: PLC0415

    max_version = session.execute(
        select(func.max(ExecutionGroup.diagnostic_version)).where(
            ExecutionGroup.diagnostic_id == diagnostic_id
        )
    ).scalar_one_or_none()

    diagnostic = session.get(Diagnostic, diagnostic_id)
    if diagnostic is None:
        raise ValueError(f"Diagnostic with id={diagnostic_id} not found")

    new_promoted = max_version if max_version is not None else 1
    diagnostic.promoted_version = new_promoted
    return new_promoted
