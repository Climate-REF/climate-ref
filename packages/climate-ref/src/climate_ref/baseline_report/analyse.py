"""
Turn a collected report into everything the templates need.

Text blobs are fetched from the native store and diffed here.
"""

from __future__ import annotations

import difflib
import io
import json
import warnings
from collections import Counter
from contextlib import ExitStack
from itertools import islice
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from attrs import frozen

from climate_ref.baseline_report.collect import CaseChange, FileChange, FileKind, Report

if TYPE_CHECKING:
    from climate_ref_core.regression.manifest import NativeEntry
    from climate_ref_core.regression.store import NativeStore

# A blob larger than this is summarised rather than diffed. Keeps a runaway report
# from stalling the job on the download alone.
MAX_FETCH_BYTES = 2_000_000

# Digest prefix shown wherever a blob is named. Long enough to identify it, short enough to read.
SHORT_DIGEST = 12

# A NetCDF blob larger than this is left unopened. Well above the largest baseline file,
# because opening one is cheap next to downloading it.
NETCDF_FETCH_BYTES = 100_000_000

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
class StatRow:
    """Whole-array statistics for one data variable, on each side of the change."""

    name: str
    """The variable's name."""

    shape_old: str | None
    """Dimensions on the base ref, as ``180x360``, or ``None`` when the variable is absent."""

    shape_new: str | None
    """Dimensions on HEAD, as ``180x360``, or ``None`` when the variable is absent."""

    min_old: float | None
    """Minimum on the base ref, ignoring NaN. ``None`` when unavailable."""

    min_new: float | None
    """Minimum on HEAD, ignoring NaN. ``None`` when unavailable."""

    max_old: float | None
    """Maximum on the base ref, ignoring NaN. ``None`` when unavailable."""

    max_new: float | None
    """Maximum on HEAD, ignoring NaN. ``None`` when unavailable."""

    mean_old: float | None
    """Mean on the base ref, ignoring NaN. ``None`` when unavailable."""

    mean_new: float | None
    """Mean on HEAD, ignoring NaN. ``None`` when unavailable."""

    nan_old: int | None
    """NaN cells on the base ref, or ``None`` when the variable is absent or not numeric."""

    nan_new: int | None
    """NaN cells on HEAD, or ``None`` when the variable is absent or not numeric."""

    max_abs_diff: float | None
    """Largest absolute change, or ``None`` when the shapes differ or a side is absent."""

    max_rel_diff: float | None
    """:attr:`max_abs_diff` over the largest magnitude on the base ref."""

    cells_differ: int | None
    """Cells that changed, counting NaN as equal to NaN."""

    moved: bool
    """Whether anything about this variable changed, which is what shades its row."""


@frozen
class NetcdfDiff:
    """What changed inside one NetCDF file, or the reason nothing could be read."""

    header: tuple[DiffLine, ...]
    """Unified diff of the two ncdump-style headers, empty when they match."""

    rows: tuple[StatRow, ...]
    """One row per data variable, in name order."""

    note: str | None
    """Why the file could not be analysed, or ``None`` when it was."""


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

    netcdf: NetcdfDiff | None = None
    """The header and stats diff, set only for
    :attr:`~climate_ref.baseline_report.collect.FileKind.NETCDF` files."""


@frozen
class KindCounts:
    """How many files of one kind were added, changed and removed."""

    label: str
    """The kind's name, as the report column header."""

    added: int
    """Files present only on HEAD."""

    changed: int
    """Files present on both sides with a different digest."""

    removed: int
    """Files present only on the base ref."""


@frozen
class AnalysedCase:
    """One test case, with its files analysed and tallied."""

    change: CaseChange
    """The underlying change."""

    files: tuple[AnalysedFile, ...]
    """Every analysed file, in the order collection produced them."""

    counts: tuple[KindCounts, ...]
    """One entry per kind, in report column order."""

    images: tuple[AnalysedFile, ...]
    """The image files, which render as a two-up comparison."""

    texts: tuple[AnalysedFile, ...]
    """The text files, which render as a diff."""

    netcdfs: tuple[AnalysedFile, ...]
    """The NetCDF files, which render as a header diff and a table of variable statistics."""

    others: tuple[AnalysedFile, ...]
    """Every remaining file, which renders as a table row."""

    back_link: str
    """Relative link from this case's page back to the index, one ``..`` per label segment."""


@frozen
class AnalysedReport:
    """A whole report, ready to render."""

    report: Report
    """The underlying report."""

    store_url: str
    """Base location blobs are served from, matching what :func:`blob_url` builds."""

    cases: tuple[AnalysedCase, ...]
    """The analysed cases, in the order collection produced them."""

    kinds: tuple[str, ...]
    """The count column headers, matching the order of every case's ``counts``."""


def blob_url(store: NativeStore, digest: str) -> str:
    """
    Build the URL a blob is served from.

    A local store fans its blobs out by the first two digest characters, so a flat URL under
    its root would point at nothing.

    Parameters
    ----------
    store
        The store the blob lives in.
    digest
        The blob's sha256 hex digest.

    Returns
    -------
    :
        An absolute URL a browser can open.
    """
    if store.root is not None:
        return (store.root / digest[:2] / digest).absolute().as_uri()
    return f"{store.url.rstrip('/')}/{digest}"


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


def _diff_lines(
    old_lines: list[str],
    new_lines: list[str],
    *,
    fromfile: str,
    tofile: str,
) -> tuple[tuple[DiffLine, ...], int]:
    """
    Build the tagged unified diff of two line lists.

    Parameters
    ----------
    old_lines
        The base side, empty when it is absent.
    new_lines
        The head side, empty when it is absent.
    fromfile
        Label for the base side in the diff header.
    tofile
        Label for the head side in the diff header.

    Returns
    -------
    :
        The kept lines and the number dropped past :data:`MAX_DIFF_LINES`.
    """
    raw = difflib.unified_diff(old_lines, new_lines, fromfile=fromfile, tofile=tofile, lineterm="", n=3)
    kept = list(islice(raw, MAX_DIFF_LINES))
    return tuple(DiffLine(kind=_classify_line(line), text=line) for line in kept), sum(1 for _ in raw)


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
    lines, elided = _diff_lines(
        _as_lines(old, name),
        _as_lines(new, name),
        fromfile="old" if old is not None else "(absent)",
        tofile="new" if new is not None else "(absent)",
    )
    if not lines:
        return TextDiff(lines=(), note="identical after decoding", elided=0)
    return TextDiff(lines=lines, note=None, elided=elided)


def _header(dataset: xr.Dataset | None) -> list[str]:
    """
    Render a dataset's ncdump-style header as lines.

    Parameters
    ----------
    dataset
        The open dataset, or ``None`` when the file is absent on that side.

    Returns
    -------
    :
        The header lines, empty when there is no dataset.
    """
    if dataset is None:
        return []
    buf = io.StringIO()
    dataset.info(buf)
    return buf.getvalue().splitlines()


def _values(dataset: xr.Dataset | None, name: str) -> np.ndarray | None:
    """
    Read one data variable as a float array.

    Parameters
    ----------
    dataset
        The open dataset, or ``None`` when the file is absent on that side.
    name
        The variable's name.

    Returns
    -------
    :
        The values as float64, or ``None`` when the variable is absent or not numeric.
    """
    if dataset is None or name not in dataset.data_vars:
        return None
    values = dataset[name].values
    if not np.issubdtype(values.dtype, np.number):
        return None
    return np.asarray(values, dtype=float)


def _shape(dataset: xr.Dataset | None, name: str) -> str | None:
    """
    Render one variable's shape.

    Parameters
    ----------
    dataset
        The open dataset, or ``None`` when the file is absent on that side.
    name
        The variable's name.

    Returns
    -------
    :
        For example ``180x360``, ``scalar`` for a zero-dimensional variable, or ``None``
        when the variable is absent.
    """
    if dataset is None or name not in dataset.data_vars:
        return None
    shape = dataset[name].shape
    return "x".join(str(size) for size in shape) if shape else "scalar"


def _summarise(values: np.ndarray | None) -> tuple[float | None, float | None, float | None, int | None]:
    """
    Reduce one side's values to min, max, mean and NaN count.

    Parameters
    ----------
    values
        The float array, or ``None`` when the variable is absent or not numeric.

    Returns
    -------
    :
        The four statistics. The first three are ``None`` for an empty or all-NaN array,
        which is what numpy would otherwise report as NaN with a warning.
    """
    if values is None:
        return None, None, None, None
    nan_count = int(np.isnan(values).sum())
    if values.size in (0, nan_count):
        return None, None, None, nan_count
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return (
            float(np.nanmin(values)),
            float(np.nanmax(values)),
            float(np.nanmean(values)),
            nan_count,
        )


def _compare(old: np.ndarray | None, new: np.ndarray | None) -> tuple[float | None, float | None, int | None]:
    """
    Compare two sides of the same variable cell by cell.

    NaN counts as equal to NaN, because a masked cell staying masked is not a change.

    Parameters
    ----------
    old
        The base side, or ``None`` when absent or not numeric.
    new
        The head side, or ``None`` when absent or not numeric.

    Returns
    -------
    :
        The largest absolute difference, the same relative to the base side's largest
        magnitude, and the number of cells that differ. All ``None`` when the sides cannot
        be compared.
    """
    if old is None or new is None or old.shape != new.shape:
        return None, None, None
    same = (old == new) | (np.isnan(old) & np.isnan(new))
    cells_differ = int(np.sum(~same))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        diff = np.abs(new - old)
        max_abs = float(np.nanmax(diff)) if diff.size else 0.0
        scale = float(np.nanmax(np.abs(old))) if old.size else 0.0
    if np.isnan(max_abs):
        max_abs = 0.0
    if np.isnan(scale):
        scale = 0.0
    return max_abs, max_abs / max(scale, float(np.finfo(float).tiny)), cells_differ


def _stat_row(old: xr.Dataset | None, new: xr.Dataset | None, name: str) -> StatRow:
    """
    Build one variable's row.

    Parameters
    ----------
    old
        The base dataset, or ``None`` when the file is absent on that side.
    new
        The head dataset, or ``None`` when the file is absent on that side.
    name
        The variable's name.

    Returns
    -------
    :
        The row, with ``moved`` set when the shape or any cell changed.
    """
    old_values = _values(old, name)
    new_values = _values(new, name)
    shape_old = _shape(old, name)
    shape_new = _shape(new, name)
    min_old, max_old, mean_old, nan_old = _summarise(old_values)
    min_new, max_new, mean_new, nan_new = _summarise(new_values)
    max_abs_diff, max_rel_diff, cells_differ = _compare(old_values, new_values)
    return StatRow(
        name=name,
        shape_old=shape_old,
        shape_new=shape_new,
        min_old=min_old,
        min_new=min_new,
        max_old=max_old,
        max_new=max_new,
        mean_old=mean_old,
        mean_new=mean_new,
        nan_old=nan_old,
        nan_new=nan_new,
        max_abs_diff=max_abs_diff,
        max_rel_diff=max_rel_diff,
        cells_differ=cells_differ,
        moved=(cells_differ or 0) > 0 or shape_old != shape_new,
    )


def netcdf_diff(old: Path | None, new: Path | None) -> NetcdfDiff:
    """
    Diff the headers and whole-array statistics of two NetCDF blobs.

    Decoding is turned off so a non-standard calendar or unit cannot fail the report.

    Parameters
    ----------
    old
        The base blob, or ``None`` when the file is new.
    new
        The head blob, or ``None`` when the file was removed.

    Returns
    -------
    :
        The header diff and one row per data variable, or a note when a file could not be opened.
    """
    try:
        with ExitStack() as stack:
            old_ds = (
                stack.enter_context(xr.open_dataset(old, decode_times=False, decode_cf=False))
                if old is not None
                else None
            )
            new_ds = (
                stack.enter_context(xr.open_dataset(new, decode_times=False, decode_cf=False))
                if new is not None
                else None
            )
            header, _ = _diff_lines(
                _header(old_ds),
                _header(new_ds),
                fromfile="old" if old_ds is not None else "(absent)",
                tofile="new" if new_ds is not None else "(absent)",
            )
            names = sorted(
                {
                    str(name)
                    for dataset in (old_ds, new_ds)
                    if dataset is not None
                    for name in dataset.data_vars
                }
            )
            rows = tuple(_stat_row(old_ds, new_ds, name) for name in names)
    except (OSError, ValueError, KeyError) as exc:
        return NetcdfDiff(header=(), rows=(), note=f"could not open: {exc}")
    return NetcdfDiff(header=header, rows=rows, note=None)


def _fetch_side(
    store: NativeStore,
    entry: NativeEntry | None,
    workdir: Path,
    *,
    limit: int = MAX_FETCH_BYTES,
    oversize: str = "too large to diff",
) -> tuple[Path | None, str | None]:
    """
    Fetch one side of a file.

    Parameters
    ----------
    store
        The store to read from.
    entry
        The manifest entry, or ``None`` when the file is absent on that side.
    workdir
        Directory the blob is written into.
    limit
        Largest blob worth downloading.
    oversize
        Note to return when the blob is past ``limit``.

    Returns
    -------
    :
        A ``(path, note)`` pair. ``path`` is ``None`` when the blob is absent or unfetchable,
        and ``note`` describes a failure.
    """
    if entry is None:
        return None, None
    if entry.size > limit:
        return None, f"{oversize} ({entry.size:,} B)"
    digest = entry.sha256
    dest = workdir / digest
    if dest.exists():
        return dest, None
    try:
        store.fetch(digest, dest)
    except (OSError, ValueError) as exc:
        return None, f"could not fetch {digest[:SHORT_DIGEST]} ({exc})"
    return dest, None


def _diff_for(
    change: FileChange,
    store: NativeStore,
    *,
    fetch: bool,
    workdir: Path,
) -> TextDiff | None:
    """
    Build the diff for one file, or ``None`` when its kind is not diffed.

    Parameters
    ----------
    change
        The file that moved.
    store
        The store to read blobs from.
    fetch
        Whether blobs may be downloaded.
    workdir
        Directory fetched blobs are written into.

    Returns
    -------
    :
        The diff, a note explaining why there is not one, or ``None`` for a non-text file.
    """
    if change.kind is not FileKind.TEXT:
        return None
    if not fetch:
        return TextDiff(lines=(), note="fetching disabled", elided=0)

    old_path, old_note = _fetch_side(store, change.old, workdir)
    new_path, new_note = _fetch_side(store, change.new, workdir)
    note = old_note or new_note
    if note is not None:
        return TextDiff(lines=(), note=note, elided=0)
    return text_diff(old_path, new_path, change.name)


def _netcdf_for(
    change: FileChange,
    store: NativeStore,
    *,
    fetch: bool,
    workdir: Path,
) -> NetcdfDiff | None:
    """
    Build the NetCDF analysis for one file, or ``None`` when its kind is not NetCDF.

    Parameters
    ----------
    change
        The file that moved.
    store
        The store to read blobs from.
    fetch
        Whether blobs may be downloaded.
    workdir
        Directory fetched blobs are written into.

    Returns
    -------
    :
        The analysis, a note explaining why there is not one, or ``None`` for another kind.
    """
    if change.kind is not FileKind.NETCDF:
        return None
    if not fetch:
        return NetcdfDiff(header=(), rows=(), note="fetching disabled")

    old_path, old_note = _fetch_side(
        store, change.old, workdir, limit=NETCDF_FETCH_BYTES, oversize="too large to analyse"
    )
    new_path, new_note = _fetch_side(
        store, change.new, workdir, limit=NETCDF_FETCH_BYTES, oversize="too large to analyse"
    )
    note = old_note or new_note
    if note is not None:
        return NetcdfDiff(header=(), rows=(), note=note)
    return netcdf_diff(old_path, new_path)


def _analyse_file(
    change: FileChange,
    store: NativeStore,
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
    fetch
        Whether blobs may be downloaded.
    workdir
        Directory fetched blobs are written into.

    Returns
    -------
    :
        The analysed file.
    """
    return AnalysedFile(
        change=change,
        old_url=blob_url(store, change.old.sha256) if change.old else None,
        new_url=blob_url(store, change.new.sha256) if change.new else None,
        text=_diff_for(change, store, fetch=fetch, workdir=workdir),
        size_delta=change.new.size - change.old.size if change.old and change.new else None,
        netcdf=_netcdf_for(change, store, fetch=fetch, workdir=workdir),
    )


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


def _counts(files: tuple[AnalysedFile, ...]) -> tuple[KindCounts, ...]:
    """
    Tally each file kind's added, changed and removed counts.

    Parameters
    ----------
    files
        The analysed files.

    Returns
    -------
    :
        One entry per kind, in report column order, so a template never has to test
        for a missing kind or decide the column order itself.
    """
    tally: dict[FileKind, Counter[str]] = {kind: Counter() for kind in FileKind}
    for analysed in files:
        tally[analysed.change.kind][analysed.change.status] += 1
    return tuple(
        KindCounts(
            label=kind.value,
            added=tally[kind]["added"],
            changed=tally[kind]["changed"],
            removed=tally[kind]["removed"],
        )
        for kind in FileKind
    )


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
    store_url = store.root.absolute().as_uri() if store.root is not None else store.url.rstrip("/")
    cases = []
    for case in report.cases:
        files = tuple(_analyse_file(change, store, fetch=fetch, workdir=workdir) for change in case.files)
        depth = len(PurePosixPath(case.label).parts)
        cases.append(
            AnalysedCase(
                change=case,
                files=files,
                counts=_counts(files),
                images=_of_kind(files, FileKind.IMAGE),
                texts=_of_kind(files, FileKind.TEXT),
                netcdfs=_of_kind(files, FileKind.NETCDF),
                others=_of_kind(files, FileKind.OTHER),
                back_link="/".join([*[".."] * depth, "index.html"]),
            )
        )
    return AnalysedReport(
        report=report,
        store_url=store_url,
        cases=tuple(cases),
        kinds=tuple(kind.value for kind in FileKind),
    )
