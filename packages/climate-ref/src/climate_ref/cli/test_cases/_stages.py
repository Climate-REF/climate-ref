"""
Composable stages behind the ``ref test-cases`` verbs.

The verbs ``run`` / ``mint`` / ``replay`` / ``build`` are thin compositions over a small
set of stages:

- **execute** -- run the diagnostic and copy its curated native set into a slot
- **materialise** -- fetch a committed manifest's native blobs from the store into a slot
- **build** -- assemble the committed bundle from the native in a slot
- **upload** -- push changed-digest native blobs to the store
- **compare** -- diff a slot's rebuilt bundle against the tracked committed baseline

Native produced by a source stage (execute or materialise) lands in a gitignored *output slot*
(``<case>/output/<label>/``), flat at manifest-relative paths,
with a ``regression/`` subdirectory holding the rebuilt committed bundle and a ``.source.json`` stamp.
``latest`` (the default label) is overwritten on every run.
A custom named slot persists so two runs can be diffed (``--label before`` vs ``--label after``).
See ``docs/background/regression-baselines.md``.
"""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

from loguru import logger

from climate_ref_core.diagnostics import ExecutionDefinition
from climate_ref_core.output_files import PlaceholderMap, copy_execution_outputs
from climate_ref_core.regression import (
    COMMITTED_BUNDLE_FILES,
    Tolerance,
    assert_bundle_regression,
    materialise_native,
)
from climate_ref_core.regression.capture import build_native_snapshot, write_committed_bundle

if TYPE_CHECKING:
    from collections.abc import Iterable

    from climate_ref.config import Config
    from climate_ref_core.datasets import ExecutionDatasetCollection
    from climate_ref_core.diagnostics import Diagnostic, ExecutionResult
    from climate_ref_core.regression.manifest import Manifest, NativeEntry
    from climate_ref_core.regression.store import NativeStore
    from climate_ref_core.testing import TestCase, TestCasePaths

SLOT_REGRESSION_DIRNAME = "regression"
SLOT_SOURCE_STAMP = ".source.json"


class StageError(Exception):
    """A test-case stage failed in a way the caller should report and skip."""


class SourceOutputs(NamedTuple):
    """
    The native a source stage placed in a slot, plus what ``build`` needs to consume it.

    ``bundle_output_dir`` is the absolute output directory the rebuilt bundle's text
    contents reference -- the real execution directory for ``execute`` (the copied bundle
    still points there), or the slot itself for ``materialise`` (``build_execution_result``
    rewrote the bundle into the slot).
    """

    result: ExecutionResult
    bundle_output_dir: Path


def baseline_placeholders(paths: TestCasePaths, config: Config) -> PlaceholderMap:
    """
    Build the run-level baseline placeholder map shared by every ``ref test-cases`` verb.

    Declares the configuration-stable token set once (``<TEST_DATA_DIR>`` from the test case and
    ``<SOFTWARE_ROOT_DIR>`` from the configured software root) so ``run`` / ``mint`` / ``replay`` /
    ``build`` cannot sanitise against drifting token sets.

    The per-execution ``<OUTPUT_DIR>`` is bound later by the caller.

    Parameters
    ----------
    paths
        Resolved paths for the test case (provides the test-data root).
    config
        The active configuration (provides the software root).
    """
    return PlaceholderMap.for_baseline(
        test_data_dir=paths.test_data_dir, software_root_dir=config.paths.software
    )


def prepare_slot(paths: TestCasePaths, label: str) -> Path:
    """
    Wipe and recreate ``output/<label>/`` and return the slot base directory.

    Used by the source stages (execute / materialise), which repopulate the native set.
    ``build`` does not call this -- it reuses the native already in the slot.
    """
    slot = paths.output_slot(label)
    if slot.exists():
        shutil.rmtree(slot)
    slot.mkdir(parents=True, exist_ok=True)
    return slot


def slot_native_relpaths(slot: Path) -> list[Path]:
    """
    Return the native files in a slot -- everything except the rebuilt bundle and the stamp.

    A slot is populated only by a source stage with the curated output set, so this is
    exactly the curated native set (it excludes the ``regression/`` subdirectory written
    by ``build`` and the ``.source.json`` stamp).
    """
    relpaths: list[Path] = []
    for path in sorted(slot.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(slot)
        if rel.parts[0] == SLOT_REGRESSION_DIRNAME or rel.name == SLOT_SOURCE_STAMP:
            continue
        relpaths.append(rel)
    return relpaths


def stage_execute(  # noqa: PLR0913
    *,
    config: Config,
    diag: Diagnostic,
    tc: TestCase,
    datasets: ExecutionDatasetCollection,
    slot: Path,
    execution_dir: Path | None,
    clean: bool,
) -> SourceOutputs:
    """Run the diagnostic and copy its curated native set (flat) into ``slot``."""
    from climate_ref.testing import TestCaseRunner

    runner = TestCaseRunner(config=config, datasets=datasets)
    result = runner.run(diag, tc.name, execution_dir, clean=clean)
    if not result.successful:
        raise StageError("execution was not successful")

    real_output_dir = result.definition.output_directory
    # Copy the curated subset from the real execution dir into the slot, flat (fragment=".").
    # The copied bundle text still references real_output_dir, so build substitutes that.
    # The diagnostic's reconstruction_inputs (raw artefacts build_execution_result re-scans) are
    # captured alongside, so a replay can rebuild the bundle from the persisted set alone.
    copy_execution_outputs(real_output_dir, slot, ".", result, extra_globs=diag.reconstruction_inputs)
    return SourceOutputs(result=result, bundle_output_dir=real_output_dir)


def stage_materialise(  # noqa: PLR0913
    *,
    diag: Diagnostic,
    tc: TestCase,
    paths: TestCasePaths,
    manifest: Manifest,
    store: NativeStore,
    slot: Path,
    placeholders: PlaceholderMap,
) -> SourceOutputs:
    """Fetch the manifest's native blobs into ``slot`` and rebuild the result from them."""
    materialise_native(manifest.native, store, slot)
    return stage_rebuild_from_slot(diag=diag, tc=tc, paths=paths, slot=slot, placeholders=placeholders)


def stage_rebuild_from_slot(
    *,
    diag: Diagnostic,
    tc: TestCase,
    paths: TestCasePaths,
    slot: Path,
    placeholders: PlaceholderMap,
) -> SourceOutputs:
    """
    Rebuild the execution result from native already present in ``slot``.

    Hydrates portable placeholders to concrete paths, then re-runs ``build_execution_result``
    so the rebuilt bundle is written into the slot (referencing the slot). No execution and
    no store access -- this is the shared core of ``replay`` (after a fetch) and ``build``.

    The slot is its own output directory, so the placeholder map is bound to it
    (``placeholders.with_output(slot)``) before hydrating.
    """
    from climate_ref_core.testing import load_datasets_from_yaml

    placeholders.with_output(slot).hydrate(slot)
    datasets = load_datasets_from_yaml(paths.catalog)
    definition = ExecutionDefinition(
        diagnostic=diag,
        key=f"test-{tc.name}",
        datasets=datasets,
        output_directory=slot,
        root_directory=slot.parent,
    )
    result = diag.build_execution_result(definition)
    return SourceOutputs(result=result, bundle_output_dir=slot)


def stage_build(
    *,
    slot: Path,
    source: SourceOutputs,
    placeholders: PlaceholderMap,
) -> dict[str, str]:
    """
    Assemble the committed bundle into the slot's ``regression/`` directory.

    Returns the committed digests ``{filename: sha256}`` of the sanitised, float-quantised
    bytes just written -- suitable for ``Manifest.committed`` and identical to what would be
    promoted to the tracked baseline.

    The rebuilt bundle's text still references ``source.bundle_output_dir``, so the placeholder
    map is bound to it before sanitising.
    """
    return write_committed_bundle(
        slot,
        slot / SLOT_REGRESSION_DIRNAME,
        placeholders=placeholders.with_output(source.bundle_output_dir),
    )


def snapshot_native(
    slot: Path,
    *,
    source: SourceOutputs,
    placeholders: PlaceholderMap,
) -> dict[str, NativeEntry]:
    """
    Sanitise the slot's native set to portable placeholders, then snapshot it (manifest / upload).

    The curated native (and any captured reconstruction inputs) still embed absolute paths to the
    output directory of the execution (``source.bundle_output_dir``),
    plus the shared ``<TEST_DATA_DIR>`` / ``<SOFTWARE_ROOT_DIR>`` roots.
    Rewriting those to ``<TOKEN>`` placeholders *before* digesting makes the stored blobs,
    and therefore their recorded digests, machine independent,
    and ``replay`` can hydrate the blobs into any slot.
    Binary artefacts (``.nc`` / ``.png``) are never rewritten.

    Like :func:`stage_build`,
    the placeholder map is bound to ``source.bundle_output_dir`` here rather than by the caller,
    so the two stages cannot drift on which output directory they sanitise.

    Parameters
    ----------
    slot
        The output slot whose native set is sanitised and snapshotted.
    source
        The source stage's outputs, carrying the absolute output directory the native text references.
    placeholders
        The (unbound) placeholder map for this run.
    """
    placeholders.with_output(source.bundle_output_dir).sanitise(slot)

    return build_native_snapshot(slot, slot_native_relpaths(slot))


def promote_to_baseline(slot: Path, paths: TestCasePaths) -> None:
    """
    Copy a slot's rebuilt committed bundle into the tracked ``regression/`` baseline.

    The slot bundle is already sanitised and float-quantised, so the promoted bytes (and
    therefore their digests) match what ``stage_build`` returned.
    """
    slot_regression = slot / SLOT_REGRESSION_DIRNAME
    if paths.regression.exists():
        shutil.rmtree(paths.regression)
    paths.regression.mkdir(parents=True, exist_ok=True)
    for filename in COMMITTED_BUNDLE_FILES:
        source = slot_regression / filename
        if source.exists():
            shutil.copy(source, paths.regression / filename)


def stage_upload(
    *,
    slot: Path,
    native: dict[str, NativeEntry],
    store: NativeStore,
    previous: dict[str, NativeEntry],
) -> list[str]:
    """
    Upload native blobs to the store, skipping any whose digest is unchanged and present.

    A blob is uploaded only when its digest differs from the previous manifest entry or the
    store does not already serve it, so a re-mint after a bundle-only change (e.g.
    ``mint --from-replay``) uploads nothing. Returns digest-mismatch error messages
    (empty on success).
    """
    errors: list[str] = []
    uploaded = 0
    skipped = 0
    for relpath, entry in native.items():
        prev = previous.get(relpath)
        try:
            if prev is not None and prev.sha256 == entry.sha256 and store.has(entry.sha256):
                skipped += 1
                continue
            digest = store.put(slot / relpath)
        except Exception as exc:
            errors.append(f"upload failed for {relpath}: {exc}")
            continue
        if digest != entry.sha256:
            errors.append(f"digest mismatch for {relpath} (store={digest}, captured={entry.sha256})")
        else:
            uploaded += 1
    logger.info(f"Uploaded {uploaded} native blob(s), skipped {skipped} unchanged")
    return errors


def stage_compare(
    *, slot: Path, paths: TestCasePaths, slug: str, expected: Iterable[str]
) -> tuple[list[str], list[str]]:
    """
    Compare a slot's rebuilt bundle against the tracked committed baseline.

    ``expected`` is the committed bundle's source of truth -- the filenames the manifest
    records under ``committed``. Every expected file must be present both in the tracked
    ``regression/`` baseline and in the slot's regenerated bundle, and must match within
    tolerance; a file missing on either side, or an empty expected set, is a hard failure.
    Driving the comparison from the manifest (rather than from whatever happens to exist on
    disk) stops a replay reporting success when the committed baseline is absent or incomplete.

    Both sides are already placeholder-sanitised, so no replacements are needed.
    Returns ``(failures, compared)`` -- the drift/missing messages and the filenames compared.
    """
    slot_regression = slot / SLOT_REGRESSION_DIRNAME
    expected_files = list(expected)
    if not expected_files:
        return ["manifest records no committed bundle to compare against"], []

    failures: list[str] = []
    compared: list[str] = []
    for filename in expected_files:
        baseline = paths.regression / filename
        regenerated = slot_regression / filename
        if not baseline.exists():
            failures.append(f"committed baseline file missing from regression/: {filename}")
            continue
        if not regenerated.exists():
            failures.append(f"rebuilt bundle did not regenerate committed file: {filename}")
            continue
        try:
            assert_bundle_regression(
                baseline,
                regenerated,
                slug=slug,
                tol=Tolerance(),
                replacements={},
            )
        except AssertionError as exc:
            failures.append(str(exc))
        except json.JSONDecodeError as exc:
            # A committed baseline (or regenerated file) that no longer parses is drift, not a
            # crash -- report it so replay fails cleanly instead of letting the error escape.
            failures.append(f"{filename}: committed bundle is not valid JSON ({exc})")
        else:
            compared.append(filename)
    return failures, compared


def native_is_stale(fresh: dict[str, NativeEntry], previous: dict[str, NativeEntry]) -> bool:
    """
    Return True when a previous (mint-owned) native block exists and differs from the fresh snapshot.

    Used to warn -- not block -- after a committed-bundle regeneration whose underlying native
    has drifted, so the author knows to re-mint.
    """
    if not previous:
        return False
    return {k: v.sha256 for k, v in fresh.items()} != {k: v.sha256 for k, v in previous.items()}


def write_source_stamp(
    slot: Path,
    *,
    label: str,
    verb: str,
    source: str,
    test_case_version: int,
) -> None:
    """Write the slot's ``.source.json`` so it is clear what currently populates the slot."""
    stamp = {
        "label": label,
        "verb": verb,
        "source": source,
        "test_case_version": test_case_version,
        "created": datetime.now(UTC).isoformat(),
    }
    (slot / SLOT_SOURCE_STAMP).write_text(json.dumps(stamp, indent=2) + "\n", encoding="utf-8")
