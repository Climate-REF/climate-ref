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

import json
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from climate_ref_core.output_files import copy_execution_outputs, to_placeholders
from climate_ref_core.paths import safe_path

from ._quantise import round_floats
from .manifest import (
    COMMITTED_BUNDLE_FILES,
    NativeEntry,
    compute_committed_digests,
    sha256_file,
)

if TYPE_CHECKING:
    from climate_ref_core.diagnostics import ExecutionResult
    from climate_ref_core.regression.store import NativeStore


# Per-file serialisation parameters for the committed bundle, mirroring exactly
# how each artefact is written natively, so re-dumping after rounding changes only
# float precision and never the surrounding JSON format.
#
# - series.json     -> climate_ref_core.metric_values.typing.SeriesMetricValue.dump_to_json
# - diagnostic.json -> climate_ref_core.pycmec.metric.CMECMetric.dump_to_json
#
# output.json is produced by pydantic ``model_dump_json(indent=2)`` and carries no
# floats (its schema is filenames/paths/strings/dicts). It is therefore omitted here:
# rounding would be a no-op and re-dumping via the stdlib ``json`` module instead of
# pydantic could perturb its bytes, defeating the goal of a byte-stable bundle.
_COMMITTED_FLOAT_JSON_KWARGS: dict[str, dict[str, object]] = {
    "series.json": {"indent": 2, "allow_nan": False, "sort_keys": True},
    "diagnostic.json": {"indent": 2, "allow_nan": False, "sort_keys": True},
}


def _round_committed_floats(regression_dir: Path) -> None:
    """
    Round floats in the committed JSON bundle to seven significant figures in place.

    Full-precision floats in ``series.json`` / ``diagnostic.json`` churn byte-for-byte
    between CI and local runs even when numerically identical, producing noisy diffs in
    the committed (git-tracked) bundle. Rounding stabilises those bytes;
    seven figures stays an order of magnitude under the regression compare tolerance
    (``rtol=1e-6``), so a gate verdict is never flipped
    (see :mod:`climate_ref_core.regression._quantise`).

    Each file is re-serialised with the same JSON parameters used to write it natively,
    so the only byte difference versus the copied file is reduced float precision.
    A file is rewritten only when rounding actually changes its parsed content,
    keeping float-free artefacts byte-identical to the copy.
    The native blobs and their content-addressed digests are never touched;
    this operates solely on the copied committed JSON.

    Parameters
    ----------
    regression_dir
        The test case ``regression/`` directory holding the committed bundle.
    """
    for filename, dump_kwargs in _COMMITTED_FLOAT_JSON_KWARGS.items():
        path = regression_dir / filename
        if not path.exists():
            continue
        original = json.loads(path.read_text(encoding="utf-8"))
        rounded = round_floats(original)
        if rounded == original:
            continue
        path.write_text(json.dumps(rounded, **dump_kwargs), encoding="utf-8")  # type: ignore[arg-type]


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
    # Round floats in place before digesting, so the committed bytes (and their
    # recorded digests) are the stable, rounded ones. Placeholder substitution only
    # rewrites path strings, so order relative to it does not matter for floats.
    _round_committed_floats(regression_dir)
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
        # Defend against path traversal: a hand-edited or hostile manifest could
        # carry an absolute path or one with '..' components that escapes dest.
        target = safe_path(relpath, dest, label="native path")
        target.parent.mkdir(parents=True, exist_ok=True)
        store.fetch(entry.sha256, target)
