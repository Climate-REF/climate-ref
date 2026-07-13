"""
Path resolution for execution output artifacts.

[ArtifactsReader][climate_ref.results.artifacts.ArtifactsReader] is reached via
[Reader.artifacts][climate_ref.results.values.Reader.artifacts].
It resolves the primitive fragments an execution already stores
(``output_fragment``, ``Execution.path``, ``ExecutionOutput.filename``) into filesystem
[Path][pathlib.Path]s under a results root directory.

This module is deliberately narrow: no DB, no [Config][climate_ref.config.Config], and no
coupling to the ``executions`` DTOs.
It depends on a single ``results`` root path -- not the whole ``Config`` -- passed straight in;
a value object would only earn its keep once a second root (e.g. an archive root) is needed.
Resolution is purely lexical over the results root; opening/streaming the resolved paths
stays with the consumer.
"""

import os
from pathlib import Path

from climate_ref_core.logging import EXECUTION_LOG_FILENAME


class ArtifactsReader:
    """
    Resolves execution output fragments into filesystem paths.

    Constructed from a single ``results`` root directory.
    Every method is containment-guarded: a resolved path that would escape the results root
    raises ``ValueError`` rather than returning a path outside it.
    """

    def __init__(self, results: Path) -> None:
        self._results = results

    def _within(self, base: str, *parts: str) -> Path:
        """
        Join ``parts`` onto ``base`` and guard against escaping ``base``.

        Lexically normalises the joined path (no filesystem access, no symlink resolution) and
        checks containment with ``os.path.commonpath`` -- a real containment primitive, not a
        string-prefix check (which would wrongly accept a sibling directory like ``results2``).
        """
        root = os.path.normpath(base)
        candidate = os.path.normpath(os.path.join(root, *parts))
        if os.path.commonpath([root, candidate]) != root:
            raise ValueError(f"Path {candidate!r} escapes {root!r}")
        return Path(candidate)

    def _within_results(self, *parts: str) -> Path:
        """Join ``parts`` onto the results root and guard against escaping it."""
        return self._within(str(self._results), *parts)

    def output_directory(self, output_fragment: str) -> Path:
        """Resolve an execution's output directory from its ``output_fragment``."""
        return self._within_results(output_fragment)

    def log_file(self, output_fragment: str) -> Path:
        """Resolve the execution log file within an execution's output directory."""
        output_dir = self.output_directory(output_fragment)
        return self._within(str(output_dir), EXECUTION_LOG_FILENAME)

    def bundle(self, output_fragment: str, bundle_path: str | None) -> Path | None:
        """
        Resolve an execution's output bundle (``Execution.path``), if one is set.

        Returns ``None`` when ``bundle_path`` is ``None`` (no bundle recorded).
        """
        if bundle_path is None:
            return None
        output_dir = self.output_directory(output_fragment)
        return self._within(str(output_dir), bundle_path)

    def output_file(self, output_fragment: str, filename: str) -> Path:
        """Resolve a registered ``ExecutionOutput.filename`` within an execution's output directory."""
        output_dir = self.output_directory(output_fragment)
        return self._within(str(output_dir), filename)
