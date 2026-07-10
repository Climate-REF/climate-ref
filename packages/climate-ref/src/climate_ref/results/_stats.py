"""Shared execution-statistics mapping for the results read layer."""

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from climate_ref.results.executions import ExecutionStats


def rows_to_execution_stats(rows: Iterable[Any]) -> "tuple[ExecutionStats, ...]":
    """
    Map raw ``select_execution_statistics`` rows to detached ``ExecutionStats`` DTOs.

    Both [ExecutionsReader.statistics][climate_ref.results.executions.ExecutionsReader.statistics]
    and [DiagnosticsReader.stats][climate_ref.results.diagnostics.DiagnosticsReader.stats] delegate
    here so the row-to-DTO mapping is defined exactly once.
    """
    from climate_ref.results.executions import ExecutionStats  # noqa: PLC0415

    return tuple(
        ExecutionStats(
            provider=row.provider,
            diagnostic=row.diagnostic,
            running=row.running,
            failed=row.failed,
            successful=row.successful,
            not_started=row.not_started,
            dirty=row.dirty,
            total=row.total,
        )
        for row in rows
    )
