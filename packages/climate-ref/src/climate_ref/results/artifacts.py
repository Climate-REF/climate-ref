"""
Where is the plot, log or data file that a run made?

REF writes each execution's outputs into a folder under the results root, ``config.paths.results``.
The database stores only the part of the path below that root.
[ArtifactsReader][climate_ref.results.artifacts.ArtifactsReader] turns it into a full
[Path][pathlib.Path] that you can open.
Reach it as [Reader.artifacts][climate_ref.results.reader.Reader.artifacts].

It only builds paths.
It does not read the database, and it does not check that the file exists.
A path that would leave the results root, even through a symlink, raises ``ValueError``.

Examples
--------
Print the path of every file that execution 42 registered.
Pass ``results`` to the [Reader][climate_ref.results.reader.Reader], or ``reader.artifacts`` is not available:

```python
with Database.from_config(config, read_only=True) as db:
    reader = Reader(db, results=config.paths.results)
    execution = reader.executions.execution(execution_id=42)
    for output in reader.executions.outputs(execution_id=42):
        print(reader.artifacts.output_file(execution.output_fragment, output.filename))
```

Without ``results``, ``reader.artifacts`` raises ``ValueError``:

```python
reader = Reader(db)
reader.artifacts  # ValueError: reader.artifacts requires a results root
```
"""

import os
from pathlib import Path

from climate_ref_core.logging import EXECUTION_LOG_FILENAME


def _resolve(path: Path) -> Path:
    """
    Resolve symlinks in ``path``, allowing it to be missing.

    A symlink loop or an unreadable component raises ``ValueError``,
    because Python 3.12 raises ``RuntimeError`` for a loop and later versions return the path unresolved.
    """
    try:
        return path.resolve(strict=True)
    except FileNotFoundError:
        return path.resolve()
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"Cannot resolve {str(path)!r}: {exc}") from exc


class ArtifactsReader:
    """
    Find the files a run wrote, given the folder where REF keeps its outputs.

    Create it with the results root, usually ``config.paths.results``.
    Each method takes the execution's ``output_fragment``, the path of its folder below that root,
    and returns the full path.
    Every method raises ``ValueError`` instead of returning a path outside the root,
    whether it escapes through ``..`` or through a symlink.
    It does not need a database, so it also works on its own:

    >>> from pathlib import Path
    >>> from climate_ref.results.artifacts import ArtifactsReader
    >>> artifacts = ArtifactsReader(Path("/data/results"))
    >>> artifacts.output_directory("pmp/enso_tel/ACCESS-CM2_g12_v1_3f9a1c2e/42")
    PosixPath('/data/results/pmp/enso_tel/ACCESS-CM2_g12_v1_3f9a1c2e/42')
    >>> artifacts.output_file("pmp/enso_tel/ACCESS-CM2_g12_v1_3f9a1c2e/42", "plot.png")
    PosixPath('/data/results/pmp/enso_tel/ACCESS-CM2_g12_v1_3f9a1c2e/42/plot.png')
    >>> artifacts.output_directory("../elsewhere")
    Traceback (most recent call last):
    ...
    ValueError: Path '/data/elsewhere' escapes '/data/results'
    """

    def __init__(self, results: Path) -> None:
        self._results = results

    def _within(self, base: str, *parts: str) -> Path:
        """
        Join ``parts`` onto ``base`` and guard against escaping ``base``.

        The joined path is normalised and checked against ``base`` with ``os.path.commonpath``.
        After resolving symlinks it only has to stay under the results root,
        so a link into another execution's outputs still resolves.
        The lexical path is returned, so callers see the layout they asked for.
        """
        root = os.path.normpath(base)
        candidate = os.path.normpath(os.path.join(root, *parts))
        if os.path.commonpath([root, candidate]) != root:
            raise ValueError(f"Path {candidate!r} escapes {root!r}")
        if not _resolve(Path(candidate)).is_relative_to(_resolve(self._results)):
            raise ValueError(f"Path {candidate!r} escapes {str(self._results)!r} through a symlink")
        return Path(candidate)

    def _within_results(self, *parts: str) -> Path:
        """Join ``parts`` onto the results root and guard against escaping it."""
        return self._within(str(self._results), *parts)

    def output_directory(self, output_fragment: str) -> Path:
        """
        Which folder holds everything this run wrote?

        Parameters
        ----------
        output_fragment
            The execution's ``output_fragment``, the folder path below the results root.

        Returns
        -------
        Path
            The full path of the folder.

        Raises
        ------
        ValueError
            If the folder would be outside the results root.
        """
        return self._within_results(output_fragment)

    def log_file(self, output_fragment: str) -> Path:
        """
        Where is the log of this run?

        The log is ``out.log`` in the run's output folder.
        Read it to see why a run failed.

        Returns
        -------
        Path
            The full path of the log file.

        Raises
        ------
        ValueError
            If the path would be outside the results root.

        Examples
        --------
        >>> from pathlib import Path
        >>> from climate_ref.results.artifacts import ArtifactsReader
        >>> ArtifactsReader(Path("/data/results")).log_file("pmp/enso_tel/ACCESS-CM2_g12_v1_3f9a1c2e/42")
        PosixPath('/data/results/pmp/enso_tel/ACCESS-CM2_g12_v1_3f9a1c2e/42/out.log')
        """
        output_dir = self.output_directory(output_fragment)
        return self._within(str(output_dir), EXECUTION_LOG_FILENAME)

    def bundle(self, output_fragment: str, bundle_path: str | None) -> Path | None:
        """
        Where is the metadata file that describes this run's outputs?

        This is the bundle recorded for the execution, such as ``diagnostic.json``.

        Parameters
        ----------
        output_fragment
            The execution's ``output_fragment``, the folder path below the results root.
        bundle_path
            The execution's ``path``, relative to its output folder.
            ``None`` means the run recorded no bundle.

        Returns
        -------
        Path or None
            The full path of the bundle, or ``None`` when ``bundle_path`` is ``None``.

        Raises
        ------
        ValueError
            If the path would be outside the results root.

        Examples
        --------
        >>> from pathlib import Path
        >>> from climate_ref.results.artifacts import ArtifactsReader
        >>> artifacts = ArtifactsReader(Path("/data/results"))
        >>> artifacts.bundle("pmp/enso_tel/ACCESS-CM2_g12_v1_3f9a1c2e/42", "diagnostic.json")
        PosixPath('/data/results/pmp/enso_tel/ACCESS-CM2_g12_v1_3f9a1c2e/42/diagnostic.json')
        >>> artifacts.bundle("pmp/enso_tel/ACCESS-CM2_g12_v1_3f9a1c2e/42", None) is None
        True
        """
        if bundle_path is None:
            return None
        output_dir = self.output_directory(output_fragment)
        return self._within(str(output_dir), bundle_path)

    def output_file(self, output_fragment: str, filename: str) -> Path:
        """
        Where is the plot or data file that this run made?

        Parameters
        ----------
        output_fragment
            The execution's ``output_fragment``, the folder path below the results root.
        filename
            The file's name, as listed by ``reader.executions.outputs(execution_id)``.

        Returns
        -------
        Path
            The full path of the file.

        Raises
        ------
        ValueError
            If the path would be outside the results root.

        Examples
        --------
        >>> from pathlib import Path
        >>> from climate_ref.results.artifacts import ArtifactsReader
        >>> artifacts = ArtifactsReader(Path("/data/results"))
        >>> artifacts.output_file("pmp/enso_tel/ACCESS-CM2_g12_v1_3f9a1c2e/42", "plot.png")
        PosixPath('/data/results/pmp/enso_tel/ACCESS-CM2_g12_v1_3f9a1c2e/42/plot.png')
        """
        output_dir = self.output_directory(output_fragment)
        return self._within(str(output_dir), filename)
