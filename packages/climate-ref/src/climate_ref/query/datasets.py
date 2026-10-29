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

from climate_ref.models.dataset import Dataset
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

    The diagnostic fields only match executions that are not retracted
    (unless ``include_retracted_executions``) and whose group is at the promoted diagnostic version
    (unless ``include_superseded``).
    Executions named by ``filter.execution_ids`` are exempt from both.

    Retracted datasets (``retracted_at is not None``) are excluded unless ``filter.include_retracted`` is set,
    applied before the ``latest_only`` window so a retracted row never wins a latest-version tie either.

    ``latest_group_by`` is the adapter's ``dataset_id_metadata``,
    which is used as the partition columns for the latest-version window.
    When it is omitted, it is read from the adapter for ``filter.source_type``.

    With ``filter.latest_only`` (the default), rows are deduplicated with a
    ``ROW_NUMBER() OVER (PARTITION BY <latest_group_by> ORDER BY version_key DESC, updated_at DESC, id DESC)``
    window (applied after the other filters and joins, but before ``finalised`` and ``slug_contains``).
    Exactly one row survives per partition, even when versions do not parse and share a ``version_key``.
    Set ``latest_only=False`` to list every version. ``latest_group_by`` is then ignored.

    Results are ordered by ``updated_at`` then ``id``, both descending, so paging is stable.

    Raises
    ------
    ValueError
        If a key in ``filter.facets`` is not a facet column of the target entity.
        Base columns such as ``finalised`` are not facets, use the matching ``DatasetFilter`` field.
    """
    entity = _entity_for(filter.source_type)
    # Callers eager-load the ``files`` backref, which only exists once the mappers are configured.
    configure_mappers()

    if filter.latest_only and not latest_group_by:
        # Local import: the adapter registry imports this module.
        from climate_ref.datasets import get_dataset_adapter  # noqa: PLC0415

        latest_group_by = get_dataset_adapter(filter.source_type.value).dataset_id_metadata

    stmt = select(entity).where(entity.dataset_type == filter.source_type)

    if not filter.include_retracted:
        stmt = stmt.where(entity.retracted_at.is_(None))

    facet_columns = set(entity.__mapper__.columns.keys()) - set(Dataset.__mapper__.columns.keys())
    for facet, values in (filter.facets or {}).items():
        if facet not in facet_columns:
            hint = f", use `DatasetFilter.{facet}`" if hasattr(filter, facet) else ""
            raise ValueError(f"Unknown facet {facet!r} for {entity.__name__}{hint}")
        stmt = stmt.where(getattr(entity, facet).in_(values))

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
        retracted = sa.false() if filter.include_retracted_executions else Execution.retracted.is_(True)
        # Explicit ``execution_ids`` stay exact, so only the diagnostic match is scoped.
        exact = execution_datasets.c.execution_id.in_(filter.execution_ids or ())
        stmt = stmt.where(sa.or_(exact, sa.and_(~retracted, *filter.version_clauses())))

    if needs_execution_join:
        stmt = stmt.distinct()

    if filter.latest_only and latest_group_by:
        # Rank by version_key within each partition, then keep only ids at rank 1.
        # Applied after all the where-filters/joins above so "latest" is chosen among the filtered set.
        # Unparseable versions share a version_key, so updated_at and id break ties to keep one row.
        rank = sa.func.row_number().over(
            partition_by=[getattr(entity, c) for c in latest_group_by],
            order_by=[entity.version_key.desc(), entity.updated_at.desc(), entity.id.desc()],
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

    return stmt.order_by(entity.updated_at.desc(), entity.id.desc())
