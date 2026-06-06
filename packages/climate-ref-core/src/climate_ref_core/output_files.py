"""
Raw-file operations on diagnostic execution outputs.

This module groups every operation that manipulates the files produced by a diagnostic execution.
Execution output is written to the ``scratch`` directory.
When the execution is ingested, a curated subset of outputs
(log + metric bundle + output bundle + the files the output bundle references + series)
are copied from the scratch directory to the results directory.
Only files in the results directory are accessed by the API/public.

For some tests we must sanitise paths to files as well as the contents of text files
(:func:`to_placeholders` / :func:`from_placeholders`).
This ensures that the regression data is machine independent.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

from climate_ref_core.diagnostics import ensure_relative_path
from climate_ref_core.logging import EXECUTION_LOG_FILENAME
from climate_ref_core.pycmec.output import CMECOutput

if TYPE_CHECKING:
    from climate_ref_core.diagnostics import ExecutionResult

PLACEHOLDER_OUTPUT_DIR = "<OUTPUT_DIR>"
"""Placeholder substituted for the absolute execution output directory."""

PLACEHOLDER_TEST_DATA_DIR = "<TEST_DATA_DIR>"
"""Placeholder substituted for the absolute provider test-data directory."""

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


def _rewrite_tree(directory: Path, replacements: dict[str, str], globs: tuple[str, ...]) -> None:
    """Apply ``replacements`` to the text content of every matching file under ``directory``.

    Keys are applied longest-first so that an overlapping shorter path cannot
    partially shadow a longer one.
    """
    ordered = sorted(replacements.items(), key=lambda kv: len(kv[0]), reverse=True)
    for glob in globs:
        for file in sorted(directory.rglob(glob)):
            text = file.read_text(encoding="utf-8")
            rewritten = text
            for old, new in ordered:
                rewritten = rewritten.replace(old, new)
            if rewritten != text:
                file.write_text(rewritten, encoding="utf-8")


def to_placeholders(
    directory: Path,
    *,
    output_dir: Path,
    test_data_dir: Path,
    globs: tuple[str, ...] = SANITISED_FILE_GLOBS,
) -> None:
    """
    Rewrite absolute paths in committed artefacts to portable placeholders ("to").

    Replaces the absolute ``output_dir`` with ``<OUTPUT_DIR>`` and the absolute
    ``test_data_dir`` with ``<TEST_DATA_DIR>`` in every text artefact under
    ``directory``. Binary files are never touched.

    Parameters
    ----------
    directory
        The tree of committed artefacts to sanitise in place.
    output_dir
        The absolute execution output directory.
    test_data_dir
        The absolute provider test-data directory.
    globs
        File globs whose contents are rewritten.
    """
    _rewrite_tree(
        directory,
        {str(output_dir): PLACEHOLDER_OUTPUT_DIR, str(test_data_dir): PLACEHOLDER_TEST_DATA_DIR},
        globs,
    )


def from_placeholders(
    directory: Path,
    *,
    output_dir: Path,
    test_data_dir: Path,
    globs: tuple[str, ...] = SANITISED_FILE_GLOBS,
) -> None:
    """
    Rewrite portable placeholders back to absolute paths ("from").

    Inverse of :func:`to_placeholders`: replaces ``<OUTPUT_DIR>`` with the absolute
    ``output_dir`` and ``<TEST_DATA_DIR>`` with the absolute ``test_data_dir`` in
    every text artefact under ``directory``.

    Parameters
    ----------
    directory
        The tree of artefacts to hydrate in place.
    output_dir
        The absolute execution output directory to substitute in.
    test_data_dir
        The absolute provider test-data directory to substitute in.
    globs
        File globs whose contents are rewritten.
    """
    _rewrite_tree(
        directory,
        {PLACEHOLDER_OUTPUT_DIR: str(output_dir), PLACEHOLDER_TEST_DATA_DIR: str(test_data_dir)},
        globs,
    )


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
    assert results_directory != scratch_directory
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


def copy_execution_outputs(
    scratch_directory: Path,
    results_directory: Path,
    fragment: Path | str,
    result: ExecutionResult,
    *,
    include_log: bool = False,
) -> list[Path]:
    """
    Copy the curated set of persisted outputs from scratch to results.

    This is the canonical definition of *what REF persists* for a successful execution,


    - the metric bundle
    - the output bundle
    - every file it references (plots/data/html)
    - the series bundle
    - the execution log (if ``include_log=True``)

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
        copied.append(
            copy_output_file(scratch_directory, results_directory, fragment, result.output_bundle_filename)
        )
        scratch_base = scratch_directory / fragment
        bundle_path = scratch_base / ensure_relative_path(result.output_bundle_filename, scratch_base)
        copied.extend(_copy_output_bundle_files(scratch_directory, results_directory, fragment, bundle_path))

    if result.series_filename:
        copied.append(
            copy_output_file(scratch_directory, results_directory, fragment, result.series_filename)
        )

    return copied
