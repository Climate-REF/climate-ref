"""
Raw-file operations on diagnostic execution outputs.

This module groups every operation that manipulates the files produced by a diagnostic execution.
Execution output is written to the ``scratch`` directory.
When the execution is handled, successful or not,
everything it wrote is copied from the scratch directory to the results directory,
apart from symlinks and the files its provider excludes
(:attr:`~climate_ref_core.providers.DiagnosticProvider.results_exclude`).

Of these, REF ingests the files named by a successful execution's bundles
(:func:`referenced_output_files`):

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
from fnmatch import fnmatchcase
from pathlib import Path
from typing import TYPE_CHECKING

from attrs import frozen

from climate_ref_core.diagnostics import ensure_relative_path
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
    Map between absolute runtime directories and portable ``<TOKEN>`` placeholders.

    A committed regression bundle is made machine-independent
    by replacing host-specific absolute paths with stable tokens.
    The *same* map drives every direction:

    - :meth:`sanitise` rewrites absolute paths to tokens (capture / mint).
    - :meth:`hydrate` rewrites tokens back to absolute paths (replay / rebuild).
    - :meth:`as_replacements` yields the ``{absolute: token}`` mapping
      the bundle and series comparators apply to a freshly regenerated artefact before diffing.

    The two configuration-stable tokens (``<TEST_DATA_DIR>`` / ``<SOFTWARE_ROOT_DIR>``)
    are fixed for a whole run,
    while the per-execution ``<OUTPUT_DIR>`` is late-bound with :meth:`with_output`.
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


def referenced_output_files(output_directory: Path, result: ExecutionResult) -> list[Path]:
    """
    List the files a successful execution's bundles name.

    These are the files REF ingests into the database:

    - the metric bundle
    - the output bundle
    - every file it references (plots/data/html)
    - the series bundle

    Parameters
    ----------
    output_directory
        The execution's output directory, which holds the output bundle.
        Absolute filenames in the bundles must be under it.
    result
        The successful execution result (must carry a metric bundle filename).

    Raises
    ------
    FileNotFoundError
        If the output bundle does not exist.

    Returns
    -------
    :
        The files, relative to ``output_directory``, without duplicates.
    """
    if result.metric_bundle_filename is None:
        raise ValueError("Cannot list the outputs of a result without a metric bundle")

    filenames: list[Path | str] = [result.metric_bundle_filename]
    if result.output_bundle_filename:
        filenames.append(result.output_bundle_filename)
        output_bundle = output_directory / ensure_relative_path(
            result.output_bundle_filename, output_directory
        )
        if not output_bundle.is_file():
            raise FileNotFoundError(f"Could not find the output bundle {output_bundle}")
        cmec_output_bundle = CMECOutput.load_from_json(output_bundle)
        for attr in ("plots", "data", "html"):
            filenames.extend(
                output_info.filename for output_info in (getattr(cmec_output_bundle, attr) or {}).values()
            )
    if result.series_filename:
        filenames.append(result.series_filename)

    relative = (ensure_relative_path(filename, output_directory) for filename in filenames)
    return list(dict.fromkeys(relative))


def copy_execution_outputs(
    scratch_directory: Path,
    results_directory: Path,
    fragment: Path | str,
    *,
    exclude: tuple[str, ...] = (),
) -> list[Path]:
    """
    Copy the files an execution wrote from scratch to results.

    This is the canonical definition of *what REF persists* for an execution,
    whether it succeeded or failed:
    every file under the execution's output directory, including its log, except

    - symlinks, which point at input data rather than at something the execution wrote
    - files matching ``exclude``

    Only the files :func:`referenced_output_files` lists are ingested into the database.

    Parameters
    ----------
    scratch_directory
        Base scratch directory the diagnostic wrote into.
    results_directory
        Base results directory to copy into.
    fragment
        The per-execution fragment under both base directories.
    exclude
        Patterns (see :func:`fnmatch.fnmatchcase`) of files not to copy,
        matched against each file's POSIX path relative to the execution directory.
        ``*`` also matches ``/``, so ``preproc/*`` excludes everything under ``preproc``.
        Providers declare these in
        :attr:`~climate_ref_core.providers.DiagnosticProvider.results_exclude`.

    Returns
    -------
    :
        The copied files, each relative to ``fragment``.
    """
    input_directory = scratch_directory / fragment
    if not input_directory.is_dir():
        return []

    copied: list[Path] = []
    for directory, dirnames, filenames in input_directory.walk():
        relative_directory = directory.relative_to(input_directory)
        # Symlinked directories are pruned, so the walk never leaves the execution directory
        dirnames[:] = sorted(name for name in dirnames if not (directory / name).is_symlink())
        for name in sorted(filenames):
            relative = relative_directory / name
            if (directory / name).is_symlink() or any(
                fnmatchcase(relative.as_posix(), pattern) for pattern in exclude
            ):
                continue
            copied.append(copy_output_file(scratch_directory, results_directory, fragment, relative))
    return copied
