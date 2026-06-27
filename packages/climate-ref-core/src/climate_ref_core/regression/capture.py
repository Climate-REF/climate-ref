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

from loguru import logger

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
    from collections.abc import Callable

    from climate_ref_core.diagnostics import ExecutionResult
    from climate_ref_core.regression.store import NativeStore


# Serialisation parameters for committed-bundle files that contain floats.
_COMMITTED_FLOAT_JSON_KWARGS: dict[str, dict[str, object]] = {
    "series.json": {"indent": 2, "allow_nan": False, "sort_keys": True},
    "diagnostic.json": {"indent": 2, "allow_nan": False, "sort_keys": True},
}

# Committed files deliberately NOT float-rounded
# output.json is the CMEC *output* bundle, which by construction carries no float values.
_UNROUNDED_COMMITTED_FILES: frozenset[str] = frozenset({"output.json"})


def _contains_float(obj: object) -> bool:
    """Return True if ``obj`` (a parsed-JSON structure) has any float leaf (bool excluded)."""
    if isinstance(obj, bool):
        return False
    if isinstance(obj, float):
        return True
    if isinstance(obj, dict):
        return any(_contains_float(v) for v in obj.values())
    if isinstance(obj, list):
        return any(_contains_float(v) for v in obj)
    return False


def _rewrite_committed_json(
    path: Path,
    dump_kwargs: dict[str, object],
    transform: Callable[[object], tuple[object, bool]],
) -> None:
    """
    Load a committed JSON file, apply ``transform``, and rewrite it only if the content changed.

    ``transform`` receives the parsed structure and returns ``(obj_to_write, changed)``. The file is
    re-serialised with ``dump_kwargs`` (its canonical on-disk parameters) only when ``changed`` is
    True, so a no-op transform leaves the bytes byte-for-byte untouched. A missing file is skipped.

    This single load/transform/conditional-redump path is shared by :func:`_round_committed_floats`
    and :func:`_redact_committed_provenance` so the canonical-reserialisation contract is defined once.

    Parameters
    ----------
    path
        The committed JSON file to rewrite in place.
    dump_kwargs
        The :func:`json.dumps` parameters matching the file's canonical on-disk form.
    transform
        Callable mapping the parsed structure to ``(obj_to_write, changed)``.
    """
    if not path.exists():
        return
    original = json.loads(path.read_text(encoding="utf-8"))
    obj, changed = transform(original)
    if changed:
        path.write_text(json.dumps(obj, **dump_kwargs), encoding="utf-8")  # type: ignore[arg-type]


def _round_transform(obj: object) -> tuple[object, bool]:
    """Round floats in ``obj`` (a parsed bundle), returning ``(rounded, changed)``."""
    rounded = round_floats(obj)
    return rounded, rounded != obj


def _round_committed_floats(regression_dir: Path) -> None:
    """
    Round floats in the committed JSON bundle to seven significant figures in place.

    Only ``series.json`` / ``diagnostic.json`` are rounded.
    Full-precision floats in those files churn byte-for-byte
    between CI and local runs even when numerically identical,
    producing noisy diffs in the committed (git-tracked) bundle.
    Rounding stabilises those bytes;
    seven figures stays an order of magnitude under the regression compare tolerance (``rtol=1e-6``),
    so a gate verdict is never flipped (see :mod:`climate_ref_core.regression._quantise`).

    Each rounded file is re-serialised with the same JSON parameters used to write it natively,
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
        _rewrite_committed_json(regression_dir / filename, dump_kwargs, _round_transform)

    # Check that the unrounded files don't contain floats
    for filename in _UNROUNDED_COMMITTED_FILES:
        path = regression_dir / filename
        if path.exists() and _contains_float(json.loads(path.read_text(encoding="utf-8"))):
            logger.warning(f"{filename} contains float values but is not float-rounded.")


# CMEC provenance lives in the metric bundle's uppercase ``PROVENANCE`` block and the
# output bundle's lowercase ``provenance`` block; series.json never carries one.
_PROVENANCE_BEARING_FILES: tuple[str, ...] = ("diagnostic.json", "output.json")
_PROVENANCE_BLOCK_KEYS: frozenset[str] = frozenset({"PROVENANCE", "provenance"})

# Provenance fields redacted to stable placeholders so the committed bundle stays portable
# (machine-independent) and reproducible: ``userId`` is the minting user and ``date`` is a
# non-reproducible wall-clock timestamp. Absolute paths in ``commandLine`` are made portable
# by path placeholdering (``<OUTPUT_DIR>`` / ``<SOFTWARE_ROOT_DIR>``), not by redaction.
_REDACTED_PROVENANCE_FIELDS: dict[str, str] = {
    "userId": "<USER>",
    "date": "<DATE>",
}

# Host fields in the provenance ``platform`` sub-block: ``Name`` (hostname) and ``Version`` (kernel)
# leak host identity and churn the committed digest across machines, so they are redacted; the coarse
# ``OS`` carries no host identity and is kept.
_REDACTED_PROVENANCE_PLATFORM_FIELDS: dict[str, str] = {
    "Name": "<HOSTNAME>",
    "Version": "<HOST_VERSION>",
}

# JSON dump parameters matching each committed file's canonical on-disk form, so a structured edit
# re-serialises byte-for-byte apart from the changed fields. diagnostic.json reuses
# _COMMITTED_FLOAT_JSON_KWARGS so the rounding and redaction re-dumps can't drift. output.json is
# pydantic model_dump_json output (raw UTF-8), so ensure_ascii=False avoids \u-escaping non-ASCII
# provenance and rewriting the whole file.
_COMMITTED_DUMP_KWARGS: dict[str, dict[str, object]] = {
    "diagnostic.json": _COMMITTED_FLOAT_JSON_KWARGS["diagnostic.json"],
    "output.json": {"indent": 2, "ensure_ascii": False},
}


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

    Walks the parsed bundle and, for each ``PROVENANCE`` / ``provenance`` block, overwrites the
    fields in :data:`_REDACTED_PROVENANCE_FIELDS` with their placeholders, and the host fields in
    :data:`_REDACTED_PROVENANCE_PLATFORM_FIELDS` inside its nested ``platform`` sub-block. Scoping to
    the provenance block means a same-named key elsewhere in the bundle is never touched.

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


def _redact_transform(obj: object) -> tuple[object, bool]:
    """Redact provenance fields in ``obj`` in place, returning ``(obj, changed)``."""
    return obj, _redact_provenance_fields(obj)


def _redact_committed_provenance(regression_dir: Path) -> None:
    """
    Redact host/user-specific provenance fields from the committed CMEC bundle in place.

    CMEC providers (e.g. PMP) stamp each bundle with a provenance block recording the minting
    ``userId`` and a wall-clock ``date``.
    Those leak personal metadata into git-tracked fixtures and never reproduce,
    so they are replaced with stable placeholders.
    The edit is structured -- the declared fields are set on the parsed object -- and the file is
    re-serialised with its canonical parameters,
    so the only byte difference is the redacted field values.

    Runs after :func:`_round_committed_floats` so it operates on the final, NaN-free bytes,
    and inside :func:`write_committed_bundle` so it is applied identically at mint and replay --
    keeping the committed digests reproducible across machines.

    Parameters
    ----------
    regression_dir
        The test case ``regression/`` directory holding the committed bundle.
    """
    for filename in _PROVENANCE_BEARING_FILES:
        _rewrite_committed_json(
            regression_dir / filename, _COMMITTED_DUMP_KWARGS[filename], _redact_transform
        )


def write_committed_bundle(
    source_dir: Path,
    regression_dir: Path,
    *,
    placeholders: PlaceholderMap,
) -> dict[str, str]:
    """
    Write the sanitised committed CMEC bundle into ``regression_dir``.

    Copies each committed artefact present in ``source_dir`` into ``regression_dir``,
    then rewrites absolute paths to portable placeholders in place
    (:meth:`~climate_ref_core.output_files.PlaceholderMap.sanitise`),
    rounds floats (:func:`_round_committed_floats`),
    and redacts host/user-specific CMEC provenance fields (:func:`_redact_committed_provenance`).
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
    # Round floats in place before digesting,
    # so the committed bytes (and their recorded digests) are the stable, rounded ones.
    # Placeholder substitution only rewrites path strings, so order relative to it does not matter for floats.
    _round_committed_floats(regression_dir)
    # Redact host/user-specific provenance fields (userId, date) last, so it operates on the
    # final NaN-free bytes and the only change is the redacted field values.
    _redact_committed_provenance(regression_dir)
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
        Extra output globs to persist beyond the bundle-referenced files (a diagnostic's
        :attr:`~climate_ref_core.diagnostics.Diagnostic.reconstruction_inputs`).

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
