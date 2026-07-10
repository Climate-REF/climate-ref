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
import re
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from climate_ref_core.output_files import PlaceholderMap, copy_execution_outputs
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


# Canonical JSON serialisation applied to every committed-bundle file.
_COMMITTED_JSON_DUMP_KWARGS: dict[str, object] = {
    "indent": 2,
    "sort_keys": True,
    "allow_nan": False,
    "ensure_ascii": False,
}

# CMEC provenance lives in the metric bundle's uppercase ``PROVENANCE`` block and the
# output bundle's lowercase ``provenance`` block; series.json never carries one.
_PROVENANCE_BLOCK_KEYS: frozenset[str] = frozenset({"PROVENANCE", "provenance"})

# Redacted to stable placeholders so committed bundles are machine-independent.
_REDACTED_PROVENANCE_FIELDS: dict[str, str] = {
    "userId": "<USER>",
    "date": "<DATE>",
}

# ``platform`` sub-block fields, redacted so a baseline is portable across hosts and OSes
# (a ``Darwin`` mint must match a ``Linux`` CI execute).
_REDACTED_PROVENANCE_PLATFORM_FIELDS: dict[str, str] = {
    "Name": "<HOSTNAME>",
    "Version": "<HOST_VERSION>",
    "OS": "<OS>",
}

_SOURCE_DIR_PLACEHOLDER = "<SOURCE_DIR>"

# Matches the machine-specific prefix before ``/packages/climate-ref-<pkg>/`` in in-repo paths.
# Checkout-agnostic, so Linux and macOS checkouts redact to identical bytes.
_SOURCE_PATH_RE = re.compile(r'(?:/[^\s"]*?)/packages/(climate-ref-[A-Za-z0-9_.-]+/)')


def _redact_source_paths(text: str) -> str:
    """Redact the checkout-root prefix of in-repo package paths (e.g. a provenance ``commandLine``)."""
    return _SOURCE_PATH_RE.sub(rf"{_SOURCE_DIR_PLACEHOLDER}/packages/\1", text)


def _redact_fields(block: dict[str, object], fields: dict[str, str]) -> bool:
    """Overwrite each present ``fields`` key in ``block`` with its placeholder; return if changed."""
    changed = False
    for field, placeholder in fields.items():
        if field in block and block[field] != placeholder:
            block[field] = placeholder
            changed = True
    return changed


def _redact_provenance_fields(obj: object) -> bool:
    """
    Redact the declared provenance fields in every CMEC provenance block of ``obj`` in place.

    Walks the parsed bundle and, for each ``PROVENANCE`` / ``provenance`` block,
    overwrites the fields in :data:`_REDACTED_PROVENANCE_FIELDS` with their placeholders,
    and the host fields in :data:`_REDACTED_PROVENANCE_PLATFORM_FIELDS` inside its nested ``platform``.

    Returns
    -------
    :
        ``True`` if any field was changed.
    """
    changed = False
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key in _PROVENANCE_BLOCK_KEYS and isinstance(value, dict):
                changed |= _redact_fields(value, _REDACTED_PROVENANCE_FIELDS)
                platform = value.get("platform")
                if isinstance(platform, dict):
                    changed |= _redact_fields(platform, _REDACTED_PROVENANCE_PLATFORM_FIELDS)
            elif _redact_provenance_fields(value):
                changed = True
    elif isinstance(obj, list):
        for item in obj:
            if _redact_provenance_fields(item):
                changed = True
    return changed


def _canonicalise_committed_bundle(regression_dir: Path) -> None:
    """
    Rewrite every committed JSON file into its portable, reproducible canonical form, in place.

    For each file in :data:`COMMITTED_BUNDLE_FILES` that is present:
    round floats to a stable precision (:func:`~climate_ref_core.regression._quantise.round_floats`),
    redact host/user CMEC provenance (:func:`_redact_provenance_fields`),
    re-serialise with :data:`_COMMITTED_JSON_DUMP_KWARGS`,
    then redact the checkout-root prefix of in-repo command-line paths (:func:`_redact_source_paths`).

    The same transform runs on every file so the committed files are deterministic
    regardless of how the diagnostic originally serialised them,
    and a mint and a replay on different machines produce identical bytes.

    Parameters
    ----------
    regression_dir
        The test case ``regression/`` directory holding the committed bundle.
    """
    for filename in COMMITTED_BUNDLE_FILES:
        path = regression_dir / filename
        if not path.exists():
            continue
        data = round_floats(json.loads(path.read_text(encoding="utf-8")))
        _redact_provenance_fields(data)
        # Terminate with a newline so the committed bytes match the manifest serialisation
        # (Manifest.save) and satisfy POSIX/end-of-file-fixer conventions.
        serialised = json.dumps(data, **_COMMITTED_JSON_DUMP_KWARGS) + "\n"  # type: ignore[arg-type]
        # Text pass: the paths live inside string values, not their own JSON fields.
        path.write_text(_redact_source_paths(serialised), encoding="utf-8")


def write_committed_bundle(
    source_dir: Path,
    regression_dir: Path,
    *,
    placeholders: PlaceholderMap,
) -> dict[str, str]:
    """
    Write the sanitised committed CMEC bundle into ``regression_dir``.

    Copies each committed artefact present in ``source_dir`` into ``regression_dir``,
    rewrites absolute paths to portable placeholders in place
    (:meth:`~climate_ref_core.output_files.PlaceholderMap.sanitise`),
    then canonicalises every committed JSON file -- rounding floats and redacting host/user CMEC
    provenance into a deterministic on-disk form (:func:`_canonicalise_committed_bundle`).
    When a committed artefact is absent from ``source_dir``,
    any stale copy left in ``regression_dir`` from a previous capture is removed so it is not re-digested.

    Parameters
    ----------
    source_dir
        Directory holding the freshly persisted CMEC artefacts (the per-execution
        results directory).
    regression_dir
        The destination ``regression/`` directory (created if needed).
    placeholders
        The placeholder map for this execution, already bound to the output directory via
        :meth:`~climate_ref_core.output_files.PlaceholderMap.with_output`. Its absolute paths are
        rewritten to portable ``<TOKEN>`` placeholders in the copied bundle.

    Returns
    -------
    :
        The committed digests ``{filename: sha256}`` of the bytes just written,
        suitable for :attr:`Manifest.committed`.

    Raises
    ------
    ValueError
        If ``placeholders`` is not bound to an output directory.
        An unbound map would leave execution-specific output paths in the committed bundle
        and digest those machine-specific bytes.
    """
    if not placeholders.is_output_bound:
        raise ValueError(
            "placeholders must be bound to an output directory via with_output() "
            "before writing a committed bundle"
        )

    regression_dir.mkdir(parents=True, exist_ok=True)

    for filename in COMMITTED_BUNDLE_FILES:
        source = source_dir / filename
        dest = regression_dir / filename
        if source.exists():
            shutil.copy(source, dest)
        else:
            # Drop a stale copy from a previous capture so it is not re-digested.
            dest.unlink(missing_ok=True)

    placeholders.sanitise(regression_dir)
    # Canonicalise every committed file (round floats, redact provenance, re-dump deterministically)
    # before digesting, so the recorded digests are over the stable, portable bytes. Placeholder
    # substitution only rewrites path strings, so its order relative to canonicalisation is immaterial.
    _canonicalise_committed_bundle(regression_dir)
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
    placeholders: PlaceholderMap,
    # TODO: Unify the log handling
    include_log: bool = False,
    extra_globs: tuple[str, ...] = (),
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
    placeholders
        The placeholder map for this execution, already bound to the output directory via
        :meth:`~climate_ref_core.output_files.PlaceholderMap.with_output`.
    include_log
        If True, the execution log is included in the persisted/native set.

        Defaults to False, matching the behaviour of
        :func:`~climate_ref_core.output_files.copy_execution_outputs`.
    extra_globs
        Extra output globs to persist beyond the bundle-referenced files
        (a diagnostic's :attr:`~climate_ref_core.diagnostics.Diagnostic.reconstruction_inputs`).

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
        extra_globs=extra_globs,
    )
    base_dir = results_directory / fragment
    committed = write_committed_bundle(base_dir, regression_dir, placeholders=placeholders)
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
