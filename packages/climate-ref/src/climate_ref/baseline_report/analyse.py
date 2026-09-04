"""
Turn a collected report into everything the templates need.

Text blobs are fetched from the native store and diffed here.
"""

from __future__ import annotations

import difflib
import json
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from attrs import frozen

from climate_ref.baseline_report.collect import CaseChange, FileChange, FileKind, Report

if TYPE_CHECKING:
    from climate_ref_core.regression.manifest import NativeEntry
    from climate_ref_core.regression.store import NativeStore

# A blob larger than this is summarised rather than diffed. Keeps a runaway report
# from stalling the job on the download alone.
MAX_FETCH_BYTES = 2_000_000

# Unified-diff lines kept per file before the rest is elided.
MAX_DIFF_LINES = 5000


@frozen
class DiffLine:
    """One line of a unified diff, tagged so a template can style it."""

    kind: str
    """One of ``add``, ``remove``, ``context``, ``hunk`` or ``header``."""

    text: str
    """The line itself, without a trailing newline."""


@frozen
class TextDiff:
    """The unified diff of one text output, or the reason there is not one."""

    lines: tuple[DiffLine, ...]
    """The diff lines, empty when :attr:`note` is set."""

    note: str | None
    """Why no diff could be produced, or ``None`` when :attr:`lines` carries one."""

    elided: int
    """Lines dropped past :data:`MAX_DIFF_LINES`."""


@frozen
class AnalysedFile:
    """One native file, with its blob URLs and its diff where it has one."""

    change: FileChange
    """The underlying change."""

    old_url: str | None
    """Store URL of the base blob, or ``None`` when the file is new."""

    new_url: str | None
    """Store URL of the head blob, or ``None`` when the file was removed."""

    text: TextDiff | None
    """The diff, set only for :attr:`~climate_ref.baseline_report.collect.FileKind.TEXT` files."""

    size_delta: int | None
    """Signed byte change, or ``None`` when the file exists on only one side."""


@frozen
class AnalysedCase:
    """One test case, with its files analysed and tallied."""

    change: CaseChange
    """The underlying change."""

    files: tuple[AnalysedFile, ...]
    """Every analysed file, in the order collection produced them."""

    counts: dict[str, dict[str, int]]
    """``kind -> {added, changed, removed}``, with every kind present."""

    images: tuple[AnalysedFile, ...]
    """The image files, which render as a two-up comparison."""

    texts: tuple[AnalysedFile, ...]
    """The text files, which render as a diff."""

    binaries: tuple[AnalysedFile, ...]
    """The NetCDF and other files, which render as a table row."""

    back_link: str
    """Relative link from this case's page back to the index, one ``..`` per slug segment."""


@frozen
class AnalysedReport:
    """A whole report, ready to render."""

    report: Report
    """The underlying report."""

    store_url: str
    """Base URL of the native store, without a trailing slash."""

    cases: tuple[AnalysedCase, ...]
    """The analysed cases, in the order collection produced them."""


def blob_url(store_url: str, digest: str) -> str:
    """
    Build the URL a blob is served from.

    Parameters
    ----------
    store_url
        Base URL of the native store.
    digest
        The blob's sha256 hex digest.

    Returns
    -------
    :
        The URL.
    """
    return f"{store_url.rstrip('/')}/{digest}"


def _as_lines(path: Path | None, name: str) -> list[str]:
    """
    Decode a blob into diffable lines.

    JSON is re-serialised with indentation first, because a minified bundle would otherwise
    diff as a single unreadable line.

    Parameters
    ----------
    path
        The fetched blob, or ``None`` when the file is absent on that side.
    name
        The file's name, used to decide whether it is JSON.

    Returns
    -------
    :
        The lines to diff.
    """
    if path is None:
        return []
    text = path.read_bytes().decode("utf-8", errors="replace")
    if Path(name).suffix.lower() == ".json":
        try:
            text = json.dumps(json.loads(text), indent=2, sort_keys=True)
        except json.JSONDecodeError:
            pass
    return text.splitlines()


def _classify_line(line: str) -> str:
    """
    Tag one unified-diff line with the CSS class a template should use.

    Parameters
    ----------
    line
        The raw diff line.

    Returns
    -------
    :
        One of ``header``, ``hunk``, ``add``, ``remove`` or ``context``.
    """
    if line.startswith(("---", "+++")):
        return "header"
    if line.startswith("@@"):
        return "hunk"
    if line.startswith("+"):
        return "add"
    if line.startswith("-"):
        return "remove"
    return "context"


def text_diff(old: Path | None, new: Path | None, name: str) -> TextDiff:
    """
    Build the unified diff between two fetched blobs.

    Parameters
    ----------
    old
        The base blob, or ``None`` when the file is new.
    new
        The head blob, or ``None`` when the file was removed.
    name
        The file's name, used in the diff header and to detect JSON.

    Returns
    -------
    :
        The diff, or a note explaining why there is not one.
    """
    raw = list(
        difflib.unified_diff(
            _as_lines(old, name),
            _as_lines(new, name),
            fromfile="old" if old is not None else "(absent)",
            tofile="new" if new is not None else "(absent)",
            lineterm="",
            n=3,
        )
    )
    if not raw:
        return TextDiff(lines=(), note="identical after decoding", elided=0)
    elided = max(len(raw) - MAX_DIFF_LINES, 0)
    kept = raw[:MAX_DIFF_LINES]
    return TextDiff(
        lines=tuple(DiffLine(kind=_classify_line(line), text=line) for line in kept),
        note=None,
        elided=elided,
    )


def _fetch_side(
    store: NativeStore, entry: NativeEntry | None, workdir: Path
) -> tuple[Path | None, str | None]:
    """
    Fetch one side of a text file.

    Parameters
    ----------
    store
        The store to read from.
    entry
        The manifest entry, or ``None`` when the file is absent on that side.
    workdir
        Directory the blob is written into.

    Returns
    -------
    :
        A ``(path, note)`` pair. ``path`` is ``None`` when the blob is absent or unfetchable,
        and ``note`` describes a failure.
    """
    if entry is None:
        return None, None
    if entry.size > MAX_FETCH_BYTES:
        return None, f"too large to diff ({entry.size:,} B)"
    digest = entry.sha256
    dest = workdir / digest
    if dest.exists():
        return dest, None
    try:
        store.fetch(digest, dest)
    except (OSError, ValueError) as exc:
        return None, f"could not fetch {digest[:12]} ({exc})"
    return dest, None


def _analyse_file(
    change: FileChange,
    store: NativeStore,
    store_url: str,
    *,
    fetch: bool,
    workdir: Path,
) -> AnalysedFile:
    """
    Build the URLs and, for text, the diff of one native file.

    Parameters
    ----------
    change
        The file that moved.
    store
        The store to read blobs from.
    store_url
        Base URL of the store, used to build links.
    fetch
        Whether blobs may be downloaded.
    workdir
        Directory fetched blobs are written into.

    Returns
    -------
    :
        The analysed file.
    """

    def build(text: TextDiff | None) -> AnalysedFile:
        """Build the file with the URLs and delta that do not depend on the diff."""
        return AnalysedFile(
            change=change,
            old_url=blob_url(store_url, change.old.sha256) if change.old else None,
            new_url=blob_url(store_url, change.new.sha256) if change.new else None,
            text=text,
            size_delta=change.new.size - change.old.size if change.old and change.new else None,
        )

    if change.kind is not FileKind.TEXT:
        return build(None)
    if not fetch:
        return build(TextDiff(lines=(), note="fetching disabled", elided=0))

    old_path, old_note = _fetch_side(store, change.old, workdir)
    new_path, new_note = _fetch_side(store, change.new, workdir)
    note = old_note or new_note
    if note is not None:
        return build(TextDiff(lines=(), note=note, elided=0))
    return build(text_diff(old_path, new_path, change.name))


def _of_kind(files: tuple[AnalysedFile, ...], *kinds: FileKind) -> tuple[AnalysedFile, ...]:
    """
    Select the files of the given kinds, keeping their order.

    Parameters
    ----------
    files
        The analysed files.
    kinds
        The kinds to keep.

    Returns
    -------
    :
        The matching files.
    """
    return tuple(file for file in files if file.change.kind in kinds)


def _counts(files: tuple[AnalysedFile, ...]) -> dict[str, dict[str, int]]:
    """
    Tally each file kind's added, changed and removed counts.

    Parameters
    ----------
    files
        The analysed files.

    Returns
    -------
    :
        ``kind -> {added, changed, removed}``, with every kind present so a template
        never has to test for a missing key.
    """
    tally = {kind.value: {"added": 0, "changed": 0, "removed": 0} for kind in FileKind}
    for analysed in files:
        tally[analysed.change.kind.value][analysed.change.status] += 1
    return tally


def analyse(report: Report, store: NativeStore, *, fetch: bool, workdir: Path) -> AnalysedReport:
    """
    Build everything the templates need from a collected report.

    A blob that cannot be fetched becomes a note on its file rather than an exception, because
    one unreachable object should not cost the whole report.

    Parameters
    ----------
    report
        The collected report.
    store
        The store to read blobs from.
    fetch
        Whether blobs may be downloaded. With ``False`` every text file carries a note instead
        of a diff and the store is never called.
    workdir
        Directory fetched blobs are written into.

    Returns
    -------
    :
        The analysed report.
    """
    store_url = store.url.rstrip("/")
    cases = []
    for case in report.cases:
        files = tuple(
            _analyse_file(change, store, store_url, fetch=fetch, workdir=workdir) for change in case.files
        )
        depth = len(PurePosixPath(case.slug).parts)
        cases.append(
            AnalysedCase(
                change=case,
                files=files,
                counts=_counts(files),
                images=_of_kind(files, FileKind.IMAGE),
                texts=_of_kind(files, FileKind.TEXT),
                binaries=_of_kind(files, FileKind.NETCDF, FileKind.OTHER),
                back_link="/".join([*[".."] * depth, "index.html"]),
            )
        )
    return AnalysedReport(report=report, store_url=store_url, cases=tuple(cases))
