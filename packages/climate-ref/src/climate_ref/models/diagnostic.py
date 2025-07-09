from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from climate_ref.models.base import Base, CreatedUpdatedMixin

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
