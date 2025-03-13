from typing import TYPE_CHECKING

from sqlalchemy import ForeignKey, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from cmip_ref.models.base import Base, CreatedUpdatedMixin

if TYPE_CHECKING:
    from cmip_ref.models.metric_execution import MetricExecutionGroup
    from cmip_ref.models.provider import Provider


class Metric(CreatedUpdatedMixin, Base):
    """
    Represents a provider that can provide metric calculations
    """

    __tablename__ = "metric"
    __table_args__ = (UniqueConstraint("provider_id", "slug", name="metric_ident"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    slug: Mapped[str] = mapped_column(unique=True)
    """
    Unique identifier for the metric

    This will be used to reference the metric in the benchmarking process
    """

    name: Mapped[str] = mapped_column()
    """
    Long name of the metric
    """

    provider_id: Mapped[int] = mapped_column(ForeignKey("provider.id"))
    """
    The provider that provides the metric
    """

    enabled: Mapped[bool] = mapped_column(default=True)
    """
    Whether the metric is enabled or not

    If a metric is not enabled, it will not be used for any calculations.
    """

    provider: Mapped["Provider"] = relationship(back_populates="metrics")
    execution_groups: Mapped[list["MetricExecutionGroup"]] = relationship(back_populates="metric")

    def __repr__(self) -> str:
        return f"<Metric slug={self.slug}>"

    def full_slug(self) -> str:
        """
        Get the full slug of the metric, including the provider slug

        Returns
        -------
        str
            Full slug of the metric
        """
        return f"{self.provider.slug}/{self.slug}"
