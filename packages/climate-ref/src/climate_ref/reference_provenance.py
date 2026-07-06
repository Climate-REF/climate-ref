"""
Record reference-dataset provenance for diagnostic executions.

Diagnostics compare their inputs against reference (observational/reanalysis) datasets that are
not part of the solver's input selection (see
:meth:`climate_ref_core.diagnostics.Diagnostic.reference_dataset_selectors`). This module resolves
those selectors to ingested dataset rows and links them to an execution, so the reference data an
execution used can be tracked and surfaced without affecting the execution's dataset hash or the
data the diagnostic actually receives.
"""

from __future__ import annotations

from collections.abc import Sequence

from loguru import logger

from climate_ref.database import Database
from climate_ref.datasets import get_dataset_adapter
from climate_ref.models.execution import Execution, execution_datasets
from climate_ref_core.diagnostics import Diagnostic, ReferenceDatasetSelector


def resolve_reference_dataset_ids(db: Database, selectors: Sequence[ReferenceDatasetSelector]) -> list[int]:
    """
    Resolve reference dataset selectors to ingested dataset primary keys.

    Parameters
    ----------
    db
        Database instance.
    selectors
        The reference dataset selectors to resolve.

    Returns
    -------
    :
        Deduplicated dataset ids matching the selectors, preserving first-seen order.
        A warning is logged for any selector that matches nothing (e.g. reference data not ingested).
    """
    ids: list[int] = []
    seen: set[int] = set()
    for selector in selectors:
        model = get_dataset_adapter(selector.source_type.value).dataset_cls
        query = db.session.query(model.id)
        for column, value in selector.facets.items():
            query = query.filter(getattr(model, column) == value)

        rows = query.all()
        if not rows:
            logger.warning(
                f"No ingested {selector.source_type.value} datasets match reference selector "
                f"{dict(selector.facets)}; run `ref providers setup` to ingest reference data."
            )
        for (dataset_id,) in rows:
            if dataset_id not in seen:
                seen.add(dataset_id)
                ids.append(dataset_id)
    return ids


def link_reference_datasets(db: Database, execution: Execution, diagnostic: Diagnostic) -> int:
    """
    Link a diagnostic's reference datasets to an execution for provenance.

    Idempotent: dataset ids already linked to the execution (e.g. from a previous call or as model
    inputs) are skipped.

    Parameters
    ----------
    db
        Database instance.
    execution
        The execution to record provenance against. Must already be persisted (``execution.id`` set).
    diagnostic
        The diagnostic whose reference datasets should be linked.

    Returns
    -------
    :
        The number of new reference-dataset links created.
    """
    selectors = diagnostic.reference_dataset_selectors()
    if not selectors:
        return 0

    dataset_ids = resolve_reference_dataset_ids(db, selectors)
    if not dataset_ids:
        return 0

    existing = {
        row[0]
        for row in db.session.query(execution_datasets.c.dataset_id).filter(
            execution_datasets.c.execution_id == execution.id
        )
    }
    new_ids = [dataset_id for dataset_id in dataset_ids if dataset_id not in existing]
    if new_ids:
        db.session.execute(
            execution_datasets.insert(),
            [{"execution_id": execution.id, "dataset_id": dataset_id} for dataset_id in new_ids],
        )
    return len(new_ids)
