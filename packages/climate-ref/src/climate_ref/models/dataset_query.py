"""
Query builder for the polymorphic ``Dataset`` hierarchy.

[select_datasets][climate_ref.models.dataset_query.select_datasets] is the single definition of
"the datasets query": it backs both ``climate_ref.datasets`` (``DatasetAdapter.load_catalog``) and
the ``climate_ref.results`` read layer (``reader.datasets``), so the two cannot drift apart.

It lives in the models layer -- below both of those -- and depends only on the model classes.
``source_type`` is resolved to its concrete subclass through SQLAlchemy's polymorphic map rather
than the adapter registry, so this module needs no import from ``climate_ref.datasets`` and the
import graph stays strictly top-down (no cycle).
"""

from collections.abc import Mapping, Sequence
from typing import Any, cast

import attrs
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


def select_datasets(filter: DatasetFilter) -> Select[Any]:  # noqa: A002
    """
    Build the ``Select`` over the (optionally concrete) ``Dataset`` entity for the given filter.

    The latest-version filter and any limit are deliberately not applied here; callers apply them
    so a numeric limit is not spent on superseded versions.

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

    return stmt.order_by(entity.updated_at.desc())
