"""
Query builder for the polymorphic ``Dataset`` hierarchy.

[select_datasets][climate_ref.models.dataset_query.select_datasets]
is the cononical definition for selecting datasets consistently.
It backs both ``climate_ref.datasets`` (``DatasetAdapter.load_catalog``) and
the ``climate_ref.results`` read layer (``reader.datasets``), so the two cannot drift apart.
"""

from collections.abc import Mapping, Sequence
from typing import Any, cast

import attrs
import sqlalchemy as sa
from sqlalchemy import Select, select

from climate_ref.models.dataset import Dataset
from climate_ref.models.diagnostic import Diagnostic
from climate_ref.models.execution import Execution, ExecutionGroup, execution_datasets
from climate_ref_core.source_types import SourceDatasetType


def _as_facets(
    value: Mapping[str, Sequence[str]] | None,
) -> Mapping[str, tuple[str, ...]] | None:
    """Copy a facets mapping into an immutable ``dict`` of immutable ``tuple`` values."""
    if value is None:
        return None
    return {k: tuple(v) for k, v in value.items()}


@attrs.frozen(kw_only=True)
class DatasetFilter:
    """
    Declarative filter over datasets.

    Every field is optional.
    ``None`` means "do not constrain on this axis".

    ``source_type`` selects which concrete ``Dataset`` subclass
    (and therefore which facet columns) the query targets.
    When ``source_type=None``, the query stays on the base ``Dataset`` database table,
    so only base columns are filterable via ``facets``
    (``slug``, ``finalised``, ``dataset_type``, ``created_at``, ``updated_at``),
    and ``latest_only`` is a no-op as there is no ``dataset_id_metadata`` to group by.
    """

    source_type: SourceDatasetType | None = None
    facets: Mapping[str, tuple[str, ...]] | None = attrs.field(default=None, converter=_as_facets)
    finalised: bool | None = None
    execution_id: int | None = None
    diagnostic_slug: str | None = None
    latest_only: bool = True


def _entity_for(source_type: SourceDatasetType | None) -> type[Dataset]:
    """Resolve a source type to its concrete ``Dataset`` subclass via the polymorphic map."""
    if source_type is None:
        return Dataset
    return cast(type[Dataset], Dataset.__mapper__.polymorphic_map[source_type].class_)


def select_datasets(
    filter: DatasetFilter,  # noqa: A002
    *,
    latest_group_by: Sequence[str] | None = None,
) -> Select[Any]:
    """
    Build the ``Select`` over the (optionally concrete) ``Dataset`` entity for the given filter.

    Any limit is deliberately not applied here; callers apply it so a numeric limit is not spent
    on superseded versions.

    ``latest_group_by`` is the adapter's ``dataset_id_metadata`` -- the partition columns for the
    latest-version window. It is optional because ``select_datasets`` lives in the models layer and
    must not import the adapter registry, so it cannot look this up itself; callers pass it through.

    ``filter.latest_only`` is INERT unless ``latest_group_by`` is also given (non-empty): passing
    ``latest_only=True`` alone does NOT dedup. Both must be set together for SQL-side deduplication
    to apply. When both are set, rows are deduplicated with a ``RANK() OVER (PARTITION BY
    <latest_group_by> ORDER BY version_key DESC)`` window (applied after all other filters/joins),
    keeping every row tied at the maximum ``version_key`` -- so ties are not silently dropped.

    Raises
    ------
    ValueError
        If a key in ``filter.facets`` is not a mapped column on the target entity.
    """
    entity = _entity_for(filter.source_type)

    stmt = select(entity)
    if filter.source_type is not None:
        stmt = stmt.where(entity.dataset_type == filter.source_type)

    for facet, values in (filter.facets or {}).items():
        column = getattr(entity, facet, None)
        if column is None or facet not in entity.__mapper__.columns:
            raise ValueError(f"Unknown facet {facet!r} for {entity.__name__}")
        stmt = stmt.where(column.in_(values))

    if filter.finalised is not None:
        stmt = stmt.where(entity.finalised.is_(filter.finalised))

    # Both relationship axes reach through ``execution_datasets``
    # join it at most once so that setting ``execution_id`` and ``diagnostic_slug`` together
    # does not emit a duplicate, unaliased self-join (invalid SQL).
    needs_execution_join = filter.execution_id is not None or filter.diagnostic_slug is not None
    if needs_execution_join:
        stmt = stmt.join(execution_datasets, entity.id == execution_datasets.c.dataset_id)

    if filter.execution_id is not None:
        stmt = stmt.where(execution_datasets.c.execution_id == filter.execution_id)

    if filter.diagnostic_slug is not None:
        stmt = (
            stmt.join(Execution, Execution.id == execution_datasets.c.execution_id)
            .join(ExecutionGroup, ExecutionGroup.id == Execution.execution_group_id)
            .join(Diagnostic, Diagnostic.id == ExecutionGroup.diagnostic_id)
            .where(Diagnostic.slug == filter.diagnostic_slug)
        )

    if needs_execution_join:
        stmt = stmt.distinct()

    if filter.latest_only and latest_group_by:
        # Rank by version_key within each partition, then keep only ids at rank 1. Applied after all
        # the where-filters/joins above so "latest" is chosen among the already-filtered set.
        #
        # This uses the ``entity.id.in_(<rank==1 id subquery>)`` shape rather than
        # ``aliased(entity, subquery)``: making the outer entity an alias would break the class-bound
        # ``selectinload(entity.files)`` loader option that callers attach to the returned ``Select``
        # (``ArgumentError`` at execution), since a bare ``Select`` gives callers no handle on the
        # alias to rewrite their options against. Keeping the outer entity a plain class means all
        # existing ``.options(...)`` calls keep working unchanged.
        #
        # RANK (not ROW_NUMBER) keeps every row tied at the max version_key in a partition.
        rank = sa.func.rank().over(
            partition_by=[getattr(entity, c) for c in latest_group_by],
            order_by=entity.version_key.desc(),
        )
        inner = stmt.add_columns(rank.label("_rank")).subquery()
        latest_ids = select(inner.c.id).where(inner.c._rank == 1)
        # Plain filtered select of the entity (not the ``stmt`` built above, which now carries the
        # window's subquery machinery) -- the id membership already encodes every filter/join from
        # above, so the outer select only needs the entity and the id predicate.
        stmt = select(entity).where(entity.id.in_(latest_ids))

    return stmt.order_by(entity.updated_at.desc())
