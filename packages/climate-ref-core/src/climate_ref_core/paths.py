"""
Path-safety primitive for containing untrusted relative paths.

:func:`safe_path` guards every place that joins an externally
supplied string onto a trusted base directory before reading or writing files.
It is a sanity check that the executions are not colliding or escaping the intended directories.
"""

from pathlib import Path, PurePosixPath


def safe_path(relpath: str | Path, base: Path | None = None, *, label: str = "path") -> Path:
    """
    Validate ``relpath`` is a contained relative path and return it.

    The check has two layers:

    - a lexical layer (always applied) that rejects empty strings, absolute paths,
        ``..`` components and NUL bytes, so the path cannot escape upwards or smuggle a path terminator
    - a containment layer (applied only when ``base`` is given) that joins ``relpath`` onto ``base``,
        resolves symlinks and ``..`` segments, and confirms the result still lives under ``base``.
        This requires filesystem access but catches escapes the lexical check cannot see (e.g. symlinks).

    Parameters
    ----------
    relpath
        The candidate relative path (e.g. a manifest native key or an output fragment).
    base
        The trusted base directory ``relpath`` is joined onto.
        When ``None`` only the lexical layer runs and the validated relative path is returned unchanged.
    label
        A human-readable description of ``relpath`` used in error messages.

    Returns
    -------
    :
        ``base / relpath`` when ``base`` is given, otherwise ``Path(relpath)``.

    Raises
    ------
    ValueError
        If ``relpath`` is empty, absolute, contains ``..`` or a NUL byte

        When ``base`` is given, if the resolved path escapes ``base``.
    """
    text = str(relpath)
    pure = PurePosixPath(text)
    if not text or "\x00" in text or pure.is_absolute() or ".." in pure.parts:
        raise ValueError(
            f"Unsafe {label} {text!r}: must be a contained relative path "
            "(no absolute paths, no '..', no NUL bytes)."
        )
    if base is None:
        return Path(relpath)

    target = base / relpath
    if not target.resolve().is_relative_to(base.resolve()):
        raise ValueError(f"Unsafe {label} {text!r}: escapes {base}.")

    return target
