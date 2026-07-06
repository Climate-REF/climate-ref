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
    """
    Copy a facets mapping into an immutable ``dict`` of immutable ``tuple`` values.

    A bare ``str`` value is itself a ``Sequence[str]``, so without this guard a caller passing
    ``facets={"source_id": "TEST-MODEL"}`` would have it iterated character-by-character
    (``"TEST-MODEL"`` -> ``T``, ``E``, ``S``, ...) before it ever reaches the ``IN`` clause.
    """
    if value is None:
        return None
    return {k: (v,) if isinstance(v, str) else tuple(v) for k, v in value.items()}


@attrs.frozen(kw_only=True)
class DatasetFilter:
    """
    Declarative filter over datasets.

    ``source_type`` is required.
    It selects which which facet columns the query can target.
    This limits our filtering to a single source type at a time to ensure that the files can be
    collapsed into a dataframe.

    Every other field is optional with ``None`` meaning "do not constrain on this axis".
    """

    source_type: SourceDatasetType
    facets: Mapping[str, tuple[str, ...]] | None = attrs.field(default=None, converter=_as_facets)
    finalised: bool | None = None
    execution_id: int | None = None
    diagnostic_slug: str | None = None
    latest_only: bool = True


def _entity_for(source_type: SourceDatasetType) -> type[Dataset]:
    """Resolve a source type to its concrete ``Dataset`` subclass via the polymorphic map."""
    return cast(type[Dataset], Dataset.__mapper__.polymorphic_map[source_type].class_)


def select_datasets(
    filter: DatasetFilter,  # noqa: A002
    *,
    latest_group_by: Sequence[str] | None = None,
) -> Select[Any]:
    """
    Build the ``Select`` over the ``Dataset`` subclass for the given filter.

    Any limit is deliberately not applied here.
    Callers should apply limits after filtering out superseded versions.

    ``latest_group_by`` is the adapter's ``dataset_id_metadata``,
    which is used as the partition columns for the latest-version window.
    It is optional because ``select_datasets`` lives in the models layer
    and must not import the adapter registry, so it cannot look this up itself; callers pass it through.

    ``filter.latest_only`` does not take effect unless ``latest_group_by`` is also given (non-empty).
    When both are set, rows are deduplicated with a
    ``RANK() OVER (PARTITION BY <latest_group_by> ORDER BY version_key DESC)`` window
    (applied after all other filters/joins),
    keeping every row tied at the maximum ``version_key`` -- so ties are not silently dropped.

    Raises
    ------
    ValueError
        If a key in ``filter.facets`` is not a mapped column on the target entity.
    """
    entity = _entity_for(filter.source_type)

    if filter.latest_only and not latest_group_by:
        raise ValueError("`latest_group_by` must be provided when `latest_only` is True")

    stmt = select(entity).where(entity.dataset_type == filter.source_type)

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
        # Rank by version_key within each partition, then keep only ids at rank 1.
        # Applied after all the where-filters/joins above so "latest" is chosen among the filtered set.
        #
        # RANK (not ROW_NUMBER) keeps every row tied at the max version_key in a partition.
        rank = sa.func.rank().over(
            partition_by=[getattr(entity, c) for c in latest_group_by],
            order_by=entity.version_key.desc(),
        )
        inner = stmt.add_columns(rank.label("_rank")).subquery()
        latest_ids = select(inner.c.id).where(inner.c._rank == 1)
        # The id membership already encodes every filter/join from above,
        # so the outer select only needs the entity and the id predicate.
        stmt = select(entity).where(entity.id.in_(latest_ids))

    return stmt.order_by(entity.updated_at.desc())
