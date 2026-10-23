"""
Builder over the polymorphic ``Dataset`` hierarchy.

[select_datasets][climate_ref.query.datasets.select_datasets]
is the one definition of which datasets a filter selects.
It backs both the solver's catalogue load (``DatasetAdapter.load_catalog``)
and ``reader.datasets``, so the two cannot drift apart.
"""

from collections.abc import Sequence
from typing import Any

import sqlalchemy as sa
from sqlalchemy import Select, select
from sqlalchemy.orm import configure_mappers

from climate_ref.models.execution import Execution, ExecutionGroup, execution_datasets
from climate_ref.query.filters import DatasetFilter, _contains_any, _entity_for


def select_datasets(
    filter: DatasetFilter,  # noqa: A002
    *,
    latest_group_by: Sequence[str] | None = None,
) -> Select[Any]:
    """
    Build the ``Select`` over the ``Dataset`` subclass for the given filter.

    Any limit is deliberately not applied here.
    Callers should apply limits after filtering out superseded versions.

    Retracted datasets (``retracted_at is not None``) are excluded unless ``filter.include_retracted`` is set,
    applied before the ``latest_only`` window so a retracted row never wins a latest-version tie either.

    ``latest_group_by`` is the adapter's ``dataset_id_metadata``,
    which is used as the partition columns for the latest-version window.
    It is passed in because ``climate_ref.query`` must not import the adapter registry.

    ``latest_group_by`` is required whenever ``filter.latest_only`` is True (the default):
    passing ``latest_only=True`` without it raises ``ValueError`` rather than silently returning an
    un-deduplicated result.
    When both are set, rows are deduplicated with a
    ``RANK() OVER (PARTITION BY <latest_group_by> ORDER BY version_key DESC)`` window
    (applied after the other filters and joins, but before ``finalised`` and ``slug_contains``),
    keeping every row tied at the maximum ``version_key``, so ties are not silently dropped.
    Set ``latest_only=False`` to list every version. ``latest_group_by`` is then ignored.

    Raises
    ------
    ValueError
        If ``filter.latest_only`` is True but ``latest_group_by`` is not provided,
        or if a key in ``filter.facets`` is not a mapped column on the target entity.
    """
    entity = _entity_for(filter.source_type)
    # Callers eager-load the ``files`` backref, which only exists once the mappers are configured.
    configure_mappers()

    if filter.latest_only and not latest_group_by:
        raise ValueError("`latest_group_by` must be provided when `latest_only` is True")

    stmt = select(entity).where(entity.dataset_type == filter.source_type)

    if not filter.include_retracted:
        stmt = stmt.where(entity.retracted_at.is_(None))

    for facet, values in (filter.facets or {}).items():
        column = getattr(entity, facet, None)
        if column is None or facet not in entity.__mapper__.columns:
            raise ValueError(f"Unknown facet {facet!r} for {entity.__name__}")
        stmt = stmt.where(column.in_(values))

    # Both relationship axes reach through ``execution_datasets``, so it is joined at most once.
    needs_execution_join = filter.execution_ids is not None or filter.matches_any
    if needs_execution_join:
        stmt = stmt.join(execution_datasets, entity.id == execution_datasets.c.dataset_id)

    if filter.execution_ids is not None:
        stmt = stmt.where(execution_datasets.c.execution_id.in_(filter.execution_ids))

    if filter.matches_any:
        stmt = (
            stmt.join(Execution, Execution.id == execution_datasets.c.execution_id)
            .join(ExecutionGroup, ExecutionGroup.id == Execution.execution_group_id)
            .where(ExecutionGroup.diagnostic_id.in_(filter.diagnostic_ids()))
        )

    if needs_execution_join:
        stmt = stmt.distinct()

    if filter.latest_only and latest_group_by:
        # Rank by version_key within each partition, then keep only ids at rank 1.
        # Applied after all the where-filters/joins above so "latest" is chosen among the filtered set.
        #
        # RANK (not ROW_NUMBER) keeps every row tied at the max version_key in a partition.
        rank = sa.func.rank().over(
            partition_by=[getattr(entity, c) for c in latest_group_by],
            order_by=entity.version_key.desc(),
        )
        inner = stmt.with_only_columns(entity.id, rank.label("_rank")).subquery()
        latest_ids = select(inner.c.id).where(inner.c._rank == 1)
        # The id membership already encodes every filter/join from above,
        # so the outer select only needs the entity and the id predicate.
        stmt = select(entity).where(entity.id.in_(latest_ids))

    # After the window, so an older finalised version never stands in for a newer unfinalised one.
    if filter.finalised is not None:
        stmt = stmt.where(entity.finalised.is_(filter.finalised))

    if filter.slug_contains is not None:
        stmt = stmt.where(_contains_any(entity.slug, filter.slug_contains))

    return stmt.order_by(entity.updated_at.desc())
