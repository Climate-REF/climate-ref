"""
Capture of regression baselines from a diagnostic execution.

Capture reuses :func:`climate_ref_core.output_files.copy_execution_outputs`,
so the captured native set is identical to what REF actually persists when handling an execution.

Note that this is not the raw output from an execution (the "scratch" directory),
but the curated subset of files copied into the "results" directory for persistence and comparison.
This avoids the need to maintain a separate ignore list for regression captures.

It produces two things:

- the small **committed bundle**
  (``series.json`` / ``diagnostic.json`` / ``output.json``)
  written into the test case ``regression/`` directory,
  sanitised text-only for portability and tracked in git
- a **native snapshot**: a ``{relpath: NativeEntry}`` map recording the sha256 digest
  and size of every persisted native file, for the manifest and the object store.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from climate_ref_core.output_files import copy_execution_outputs, to_placeholders
from climate_ref_core.regression.manifest import (
    COMMITTED_BUNDLE_FILES,
    NativeEntry,
    compute_committed_digests,
    sha256_file,
)

if TYPE_CHECKING:
    from climate_ref_core.diagnostics import ExecutionResult
    from climate_ref_core.regression.store import NativeStore


def write_committed_bundle(
    source_dir: Path,
    regression_dir: Path,
    *,
    output_dir: Path,
    test_data_dir: Path,
) -> dict[str, str]:
    """
    Write the sanitised committed CMEC bundle into ``regression_dir``.

    Copies each committed artefact present in ``source_dir`` into ``regression_dir``,
    then rewrites absolute paths to portable placeholders in place
    (:func:`~climate_ref_core.output_files.to_placeholders`).
    When a committed artefact is absent from ``source_dir``,
    any stale copy left in ``regression_dir`` from a previous capture is removed so it is not re-digested.

    Parameters
    ----------
    source_dir
        Directory holding the freshly persisted CMEC artefacts (the per-execution
        results directory).
    regression_dir
        The destination ``regression/`` directory (created if needed).
    output_dir
        The absolute execution output directory, for path substitution.
    test_data_dir
        The absolute provider test-data directory, for path substitution.

    Returns
    -------
    :
        The committed digests ``{filename: sha256}`` of the bytes just written,
        suitable for :attr:`Manifest.committed`.
    """
    regression_dir.mkdir(parents=True, exist_ok=True)

    for filename in COMMITTED_BUNDLE_FILES:
        source = source_dir / filename
        dest = regression_dir / filename
        if source.exists():
            shutil.copy(source, dest)
        else:
            # Drop a stale copy from a previous capture so it is not re-digested.
            dest.unlink(missing_ok=True)

    to_placeholders(regression_dir, output_dir=output_dir, test_data_dir=test_data_dir)
    return compute_committed_digests(regression_dir)


def build_native_snapshot(base_dir: Path, relpaths: list[Path]) -> dict[str, NativeEntry]:
    """
    Record a sha256 + size snapshot of each persisted native file.

    Parameters
    ----------
    base_dir
        The per-execution results directory the relpaths are resolved against.
    relpaths
        The persisted files (relative to ``base_dir``), e.g. the return value of
        :func:`~climate_ref_core.output_files.copy_execution_outputs`.

    Returns
    -------
    :
        Mapping of POSIX relpath -> :class:`NativeEntry` for every persisted file.
    """
    entries: dict[str, NativeEntry] = {}
    for relpath in relpaths:
        path = base_dir / relpath
        entries[relpath.as_posix()] = NativeEntry(sha256=sha256_file(path), size=path.stat().st_size)
    return entries


def capture_execution(  # noqa: PLR0913
    scratch_directory: Path,
    results_directory: Path,
    fragment: Path | str,
    result: ExecutionResult,
    *,
    regression_dir: Path,
    output_dir: Path,
    test_data_dir: Path,
    # TODO: Unify the log handling
    include_log: bool = False,
) -> tuple[dict[str, str], dict[str, NativeEntry]]:
    """
    Persist a successful execution and capture its committed bundle + native snapshot.

    Copies the curated output set from scratch to results via
    :func:`~climate_ref_core.output_files.copy_execution_outputs`
    (the production persistence path),
    then writes the committed bundle and snapshots every persisted native file.

    Parameters
    ----------
    scratch_directory
        Base scratch directory the diagnostic wrote into.
    results_directory
        Base results directory to persist the curated subset into.
    fragment
        The per-execution fragment under both base directories.
    result
        The successful execution result (must carry a metric bundle filename).
    regression_dir
        The test case ``regression/`` directory for the committed bundle.
    output_dir
        The absolute execution output directory, for path substitution.
    test_data_dir
        The absolute provider test-data directory, for path substitution.
    include_log
        If True, the execution log is included in the persisted/native set.

        Defaults to False, matching the behaviour of
        :func:`~climate_ref_core.output_files.copy_execution_outputs`.

    Returns
    -------
    :
        A ``(committed_digests, native_snapshot)`` tuple.
    """
    relpaths = copy_execution_outputs(
        scratch_directory,
        results_directory,
        fragment,
        result,
        include_log=include_log,
    )
    base_dir = results_directory / fragment
    committed = write_committed_bundle(
        base_dir,
        regression_dir,
        output_dir=output_dir,
        test_data_dir=test_data_dir,
    )
    native = build_native_snapshot(base_dir, relpaths)
    return committed, native


def materialise_native(native: dict[str, NativeEntry], store: NativeStore, dest: Path) -> None:
    """
    Materialise a native snapshot from a store into a destination directory.

    For each ``(relpath, entry)`` the blob is fetched from ``store`` (keyed by its
    sha256 digest) to ``dest / relpath``, creating parent directories as needed.

    Parameters
    ----------
    native
        Mapping of relpath -> :class:`NativeEntry` (from a manifest).
    store
        A content-addressed :class:`~climate_ref_core.regression.store.NativeStore`.
    dest
        The destination directory the snapshot is materialised into.
    """
    for relpath, entry in native.items():
        target = dest / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        store.fetch(entry.sha256, target)
