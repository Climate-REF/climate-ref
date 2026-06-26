"""
Raw-file operations on diagnostic execution outputs.

This module groups every operation that manipulates the files produced by a diagnostic execution.
Execution output is written to the ``scratch`` directory.
When the execution is ingested,
a curated subset of outputs are copied from the scratch directory to the results directory:

- logs
- the metric bundle
- the output bundle
- the files the output bundle references (plots/data/html)
- the series bundle

Only files in the results directory are accessed by the API/public.

For some tests we must sanitise paths to files as well as the contents of text files
(:class:`PlaceholderMap`).
This ensures that the regression data is machine independent.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from attrs import frozen

from climate_ref_core.diagnostics import ensure_relative_path
from climate_ref_core.logging import EXECUTION_LOG_FILENAME
from climate_ref_core.pycmec.output import CMECOutput

if TYPE_CHECKING:
    from climate_ref_core.diagnostics import ExecutionResult

PLACEHOLDER_OUTPUT_DIR = "<OUTPUT_DIR>"
"""Placeholder substituted for the absolute execution output directory."""

PLACEHOLDER_TEST_DATA_DIR = "<TEST_DATA_DIR>"
"""Placeholder substituted for the absolute provider test-data directory."""

PLACEHOLDER_SOFTWARE_ROOT_DIR = "<SOFTWARE_ROOT_DIR>"
"""Placeholder substituted for the absolute shared-software root directory.

Provider command lines stamped into CMEC provenance reference the installed
software environment (e.g. a conda prefix) under this root.
"""

SANITISED_FILE_GLOBS: tuple[str, ...] = (
    "*.json",
    "*.txt",
    "*.yaml",
    "*.yml",
    "*.html",
    "*.xml",
)
"""Text artefacts whose absolute paths are rewritten for portability.

Binary outputs (``.nc``, ``.png``, ...) are never rewritten.
"""


def ordered_replacements(replacements: dict[str, str]) -> list[tuple[str, str]]:
    """Order ``replacements`` longest-key-first.

    This is the canonical replacement ordering for all sanitisation:
    an overlapping shorter key cannot partially shadow a longer one.

    Parameters
    ----------
    replacements
        Mapping of substring to replacement.

    Returns
    -------
    :
        The ``(old, new)`` pairs ordered longest-``old``-first.
    """
    return sorted(replacements.items(), key=lambda kv: len(kv[0]), reverse=True)


def rewrite_tree(
    directory: Path,
    replacements: dict[str, str],
    globs: tuple[str, ...] = SANITISED_FILE_GLOBS,
) -> None:
    """Apply ``replacements`` to the text content of every matching file under ``directory``.

    Keys are applied longest-first so that an overlapping shorter path cannot
    partially shadow a longer one.
    Only files matching ``globs`` are rewritten while binary artefacts are never touched.

    Parameters
    ----------
    directory
        The tree of files to rewrite in place.
    replacements
        Mapping of substring to replacement, applied to every matching file.
    globs
        File globs whose contents are rewritten.
    """
    ordered = ordered_replacements(replacements)
    for glob in globs:
        for file in sorted(directory.rglob(glob)):
            text = file.read_text(encoding="utf-8")
            rewritten = text
            for old, new in ordered:
                rewritten = rewritten.replace(old, new)
            if rewritten != text:
                file.write_text(rewritten, encoding="utf-8")


@frozen
class PlaceholderMap:
    """
    Bidirectional map between absolute runtime directories and portable ``<TOKEN>`` placeholders.

    A committed regression bundle is made machine-independent
    by replacing host-specific absolute paths with stable tokens --
    ``<OUTPUT_DIR>``, ``<TEST_DATA_DIR>`` and (optionally) ``<SOFTWARE_ROOT_DIR>``.
    The *same* map drives every direction:

    - :meth:`sanitise` rewrites absolute paths to tokens (capture / mint).
    - :meth:`hydrate` rewrites tokens back to absolute paths (replay / rebuild).
    - :meth:`as_replacements` yields the ``{absolute: token}`` mapping
      the bundle and series comparators apply to a freshly regenerated artefact before diffing.

    The token set is declared in one place (:meth:`for_baseline` then :meth:`with_output`),
    so the capture side and the verification side cannot declare different sets and drift apart.
    Adding a placeholder is a one-line change here,
    not a new parameter threaded through every caller.

    The two configuration-stable tokens (``<TEST_DATA_DIR>`` / ``<SOFTWARE_ROOT_DIR>``)
    are fixed for a whole run;
    the per-execution ``<OUTPUT_DIR>`` is late-bound with :meth:`with_output`.
    """

    pairs: tuple[tuple[str, Path], ...]
    """Ordered ``(token, absolute_directory)`` pairs.

    Application order is not significant:
    all rewriting goes through :func:`rewrite_tree`,
    which re-sorts longest-match-first so an overlapping shorter path cannot shadow a longer one.
    """

    @classmethod
    def for_baseline(cls, *, test_data_dir: Path, software_root_dir: Path | None = None) -> PlaceholderMap:
        """
        Build the configuration-stable placeholder set for a committed baseline.

        Holds every token except the per-execution output directory,
        which is added with :meth:`with_output`.
        ``software_root_dir`` is optional:
        when ``None`` no ``<SOFTWARE_ROOT_DIR>`` substitution is applied --
        a verification context that does not know the shared-software root relies on this,
        and the omission is explicit and declared once.
        """
        pairs: list[tuple[str, Path]] = [(PLACEHOLDER_TEST_DATA_DIR, test_data_dir)]
        if software_root_dir is not None:
            pairs.append((PLACEHOLDER_SOFTWARE_ROOT_DIR, software_root_dir))
        return cls(pairs=tuple(pairs))

    def with_output(self, output_dir: Path) -> PlaceholderMap:
        """Return a new map that also binds ``<OUTPUT_DIR>`` to the per-execution ``output_dir``.

        Rebinding replaces any existing ``<OUTPUT_DIR>`` entry,
        so a second call hydrates to the latest directory rather than the first.
        """
        rest = tuple((token, path) for token, path in self.pairs if token != PLACEHOLDER_OUTPUT_DIR)
        return PlaceholderMap(pairs=((PLACEHOLDER_OUTPUT_DIR, output_dir), *rest))

    @property
    def is_output_bound(self) -> bool:
        """Whether ``<OUTPUT_DIR>`` has been bound via :meth:`with_output`.

        The committed-bundle writer requires this:
        an unbound map would leave execution-specific output paths in the committed bundle.
        """
        return any(token == PLACEHOLDER_OUTPUT_DIR for token, _ in self.pairs)

    def as_replacements(self) -> dict[str, str]:
        """Return the ``{absolute_directory: token}`` mapping (real path -> placeholder).

        This is what the bundle and series comparators apply to a regenerated artefact
        before diffing it against the committed (already-placeholdered) baseline.
        """
        return {str(abs_dir): token for token, abs_dir in self.pairs}

    def sanitise(self, directory: Path, globs: tuple[str, ...] = SANITISED_FILE_GLOBS) -> None:
        """Rewrite absolute paths to ``<TOKEN>`` placeholders in every text artefact under ``directory``.

        Binary files are never touched.
        In-place.
        """
        rewrite_tree(directory, self.as_replacements(), globs)

    def hydrate(self, directory: Path, globs: tuple[str, ...] = SANITISED_FILE_GLOBS) -> None:
        """Inverse of :meth:`sanitise`: rewrite ``<TOKEN>`` placeholders back to absolute paths.

        Binary files are never touched.
        In-place.
        """
        rewrite_tree(directory, {token: str(abs_dir) for token, abs_dir in self.pairs}, globs)


def copy_output_file(
    scratch_directory: Path,
    results_directory: Path,
    fragment: Path | str,
    filename: Path | str,
) -> Path:
    """
    Copy a single output file from the scratch directory to the results directory.

    Parameters
    ----------
    scratch_directory
        The base directory where the file is currently located.
    results_directory
        The base directory where the file should be copied to.
    fragment
        The per-execution fragment under both base directories.
    filename
        The file to copy, relative to ``scratch_directory / fragment`` (an absolute
        path under that directory is also accepted).

    Returns
    -------
    :
        The copied file's path, relative to ``fragment``.
    """
    if results_directory == scratch_directory:
        raise ValueError("results_directory and scratch_directory must differ")

    input_directory = scratch_directory / fragment
    output_directory = results_directory / fragment

    relative_filename = ensure_relative_path(filename, input_directory)

    if not (input_directory / relative_filename).exists():
        raise FileNotFoundError(f"Could not find {relative_filename} in {input_directory}")

    output_filename = output_directory / relative_filename
    output_filename.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy(input_directory / relative_filename, output_filename)
    return relative_filename


def _copy_output_bundle_files(
    scratch_directory: Path,
    results_directory: Path,
    fragment: Path | str,
    cmec_output_bundle_filename: Path,
) -> list[Path]:
    """Copy the plots/data/html files referenced by a CMEC output bundle."""
    cmec_output_bundle = CMECOutput.load_from_json(cmec_output_bundle_filename)
    scratch_base = scratch_directory / fragment

    copied: list[Path] = []
    for attr in ("plots", "data", "html"):
        for output_info in (getattr(cmec_output_bundle, attr) or {}).values():
            filename = ensure_relative_path(output_info.filename, scratch_base)
            copied.append(copy_output_file(scratch_directory, results_directory, fragment, filename))
    return copied


def _copy_extra_globs(
    scratch_directory: Path,
    results_directory: Path,
    fragment: Path | str,
    extra_globs: tuple[str, ...],
) -> list[Path]:
    """
    Copy every file matching ``extra_globs`` under the execution fragment into results.

    These are *extra* raw artefacts a diagnostic declares
    (:attr:`~climate_ref_core.diagnostics.Diagnostic.reconstruction_inputs`)
    beyond the files its CMEC output bundle references,
    so that :meth:`~climate_ref_core.diagnostics.Diagnostic.build_execution_result`
    can re-derive its bundle on replay from the persisted set alone.
    Each glob is resolved against ``scratch_directory / fragment``
    (so ``**`` and ``/`` in the pattern behave as for :meth:`pathlib.Path.glob`);
    directories are skipped and a file matched by several globs is copied once.
    """
    input_directory = scratch_directory / fragment

    copied: list[Path] = []
    seen: set[Path] = set()
    for glob in extra_globs:
        for match in sorted(input_directory.glob(glob)):
            if not match.is_file():
                continue
            relative = match.relative_to(input_directory)
            if relative in seen:
                continue
            seen.add(relative)
            copied.append(copy_output_file(scratch_directory, results_directory, fragment, relative))
    return copied


def copy_execution_outputs(  # noqa: PLR0913
    scratch_directory: Path,
    results_directory: Path,
    fragment: Path | str,
    result: ExecutionResult,
    *,
    include_log: bool = False,
    extra_globs: tuple[str, ...] = (),
) -> list[Path]:
    """
    Copy the curated set of persisted outputs from scratch to results.

    This is the canonical definition of *what REF persists* for a successful execution,

    - the metric bundle
    - the output bundle
    - every file it references (plots/data/html)
    - the series bundle
    - the execution log (if ``include_log=True``)
    - any files matching ``extra_globs`` (a diagnostic's declared reconstruction inputs)

    Parameters
    ----------
    scratch_directory
        Base scratch directory the diagnostic wrote into.
    results_directory
        Base results directory to copy the curated subset into.
    fragment
        The per-execution fragment under both base directories.
    result
        The successful execution result (must carry a metric bundle filename).
    include_log
        If True, copy the execution log.
    extra_globs
        Additional output globs (relative to the execution directory) to persist beyond the
        bundle-referenced files, e.g. a diagnostic's
        :attr:`~climate_ref_core.diagnostics.Diagnostic.reconstruction_inputs`. A file already
        copied as part of the curated set is not duplicated in the returned key set.

    Returns
    -------
    :
        The copied files, each relative to ``fragment`` (the manifest key set).
    """
    if result.metric_bundle_filename is None:
        raise ValueError("Cannot copy outputs for a result without a metric bundle")

    copied: list[Path] = []

    if include_log:
        copied.append(
            copy_output_file(scratch_directory, results_directory, fragment, EXECUTION_LOG_FILENAME)
        )

    copied.append(
        copy_output_file(scratch_directory, results_directory, fragment, result.metric_bundle_filename)
    )

    if result.output_bundle_filename:
        output_bundle_relpath = copy_output_file(
            scratch_directory, results_directory, fragment, result.output_bundle_filename
        )
        copied.append(output_bundle_relpath)
        bundle_path = scratch_directory / fragment / output_bundle_relpath
        copied.extend(_copy_output_bundle_files(scratch_directory, results_directory, fragment, bundle_path))

    if result.series_filename:
        copied.append(
            copy_output_file(scratch_directory, results_directory, fragment, result.series_filename)
        )

    if extra_globs:
        already = set(copied)
        for relpath in _copy_extra_globs(scratch_directory, results_directory, fragment, extra_globs):
            if relpath not in already:
                copied.append(relpath)
                already.add(relpath)

    return copied
