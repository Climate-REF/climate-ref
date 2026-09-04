"""
Turn a collected report into everything the templates need.

Text blobs are fetched from the native store and diffed here.
"""

from __future__ import annotations

import difflib
import io
import json
from collections import Counter
from contextlib import ExitStack
from itertools import islice
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

import numpy as np
import xarray as xr
from attrs import frozen

from climate_ref.baseline_report.collect import (
    CaseChange,
    CommittedChange,
    FileChange,
    FileKind,
    Report,
)
from climate_ref_core.regression.manifest import COMMITTED_DIRNAME

if TYPE_CHECKING:
    from climate_ref_core.regression.manifest import NativeEntry
    from climate_ref_core.regression.store import NativeStore

# A blob larger than this is summarised rather than diffed. Keeps a runaway report
# from stalling the job on the download alone.
MAX_FETCH_BYTES = 2_000_000

# Digest prefix shown wherever a blob is named. Long enough to identify it, short enough to read.
SHORT_DIGEST = 12

# A NetCDF blob larger than this is left unopened.
NETCDF_FETCH_BYTES = 100_000_000

# Compression means a blob under the fetch cap can still decode to far more, and each variable
# is held on both sides plus a difference. This bounds what one side may expand to.
MAX_DECODED_BYTES = 500_000_000

# Unified-diff lines kept per file before the rest is elided.
MAX_DIFF_LINES = 5000

# A difference this small relative to the variable's own magnitude reads as numerical noise.
# Double precision carries about 16 significant digits, so this leaves several digits of headroom
# for a real change to sit above.
NOISE_REL_TOLERANCE = 1e-9


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
class Pair[T]:
    """One statistic on each side of the change."""

    old: T | None
    """The value on the base ref, or ``None`` when it could not be computed."""

    new: T | None
    """The value on HEAD, or ``None`` when it could not be computed."""

    tolerance: float = 0.0
    """How far a numeric pair may move before it reads as changed. Zero compares exactly."""

    @property
    def changed(self) -> bool:
        """
        Whether the two sides differ by more than :attr:`tolerance`.

        Returns
        -------
        :
            ``True`` when the value moved, which is what emphasises it in the table.
        """
        if isinstance(self.old, float) and isinstance(self.new, float):
            return abs(self.new - self.old) > self.tolerance
        return self.old != self.new


@frozen
class StatRow:
    """Whole-array statistics for one data variable, on each side of the change."""

    name: str
    """The variable's name."""

    shape: Pair[str]
    """Dimensions on each side, as ``180x360``, or ``scalar``."""

    minimum: Pair[float]
    """Minimum on each side, ignoring NaN."""

    maximum: Pair[float]
    """Maximum on each side, ignoring NaN."""

    mean: Pair[float]
    """Mean on each side, ignoring NaN."""

    nan: Pair[int]
    """NaN cells on each side, ``None`` when the variable is absent or not numeric."""

    max_abs_diff: float | None
    """Largest absolute change, or ``None`` when the shapes differ or a side is absent."""

    max_rel_diff: float | None
    """:attr:`max_abs_diff` over the largest magnitude on the base ref."""

    cells_differ: int | None
    """Cells that changed, counting NaN as equal to NaN."""

    moved: bool
    """Whether anything about this variable changed, which is what shades its row."""

    @property
    def severity(self) -> str:
        """
        How much weight the row deserves.

        A run reproduced on different hardware moves the last bits of nearly every cell, so a
        row that only moved that far is called out as noise rather than as a change.

        Returns
        -------
        :
            ``changed`` when the move is larger than :data:`NOISE_REL_TOLERANCE` or cannot be
            measured, ``noise`` when it is smaller, and ``same`` when nothing moved.
        """
        if not self.moved:
            return "same"
        if self.max_rel_diff is None or self.max_rel_diff > NOISE_REL_TOLERANCE:
            return "changed"
        return "noise"

    @property
    def differs(self) -> bool:
        """
        Whether the cell by cell comparison found a change worth reading.

        Returns
        -------
        :
            ``True`` when at least one cell moved further than the noise tolerance, which is
            what emphasises the diff columns.
        """
        return bool(self.cells_differ) and self.severity == "changed"


@frozen
class NetcdfDiff:
    """What changed inside one NetCDF file, or the reason nothing could be read."""

    header: tuple[DiffLine, ...]
    """
    The two ncdump-style headers merged into one tagged listing.

    Every line is kept rather than only the changed hunks, because the header is what tells a
    reader what the file holds.
    """

    header_old: tuple[str, ...]
    """The base ref's header, for reading one side on its own."""

    header_new: tuple[str, ...]
    """HEAD's header, for reading one side on its own."""

    rows: tuple[StatRow, ...]
    """One row per data variable, in name order."""

    note: str | None
    """Why the file could not be analysed, or ``None`` when it was."""

    @property
    def header_changed(self) -> bool:
        """
        Whether the headers differ.

        Returns
        -------
        :
            ``True`` when any header line was added or removed.
        """
        return any(line.kind != "context" for line in self.header)


@frozen
class AnalysedFile:
    """One native file, with its blob URLs and whatever its kind is analysed into."""

    change: FileChange
    """The underlying change."""

    old_url: str | None
    """Store URL of the base blob, or ``None`` when the file is new."""

    new_url: str | None
    """Store URL of the head blob, or ``None`` when the file was removed."""

    text: TextDiff | None
    """The diff, set only for :attr:`~climate_ref.baseline_report.collect.FileKind.TEXT` files."""

    netcdf: NetcdfDiff | None
    """The header and stats diff, set only for
    :attr:`~climate_ref.baseline_report.collect.FileKind.NETCDF` files."""

    size_delta: int | None
    """Signed byte change, or ``None`` when the file exists on only one side."""


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
class AnalysedCommitted:
    """One committed regression artefact, with its diff."""

    change: CommittedChange
    """The underlying change."""

    text: TextDiff
    """The diff between the two committed versions."""


@frozen
class TreeNode:
    """One row of the captured baseline's folder listing."""

    name: str
    """The directory or file name at this level."""

    depth: int
    """How deep the entry sits, so a template can indent it."""

    is_dir: bool
    """Whether the row is a directory rather than a file."""

    size: int | None
    """The file's size on HEAD, or on the base ref when it was removed.

    ``None`` for a directory and for a committed artefact, whose manifest entry has no size.
    """

    status: str | None
    """``added``, ``changed`` or ``removed``, or ``None`` when the file did not move."""


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

    committed: tuple[AnalysedCommitted, ...]
    """The committed artefacts that moved, in name order."""

    tree: tuple[TreeNode, ...]
    """The captured baseline's native files, as a flattened folder listing."""

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


def _decode(path: Path | None) -> str | None:
    """
    Decode a fetched blob, replacing any byte that is not valid UTF-8.

    Parameters
    ----------
    path
        The blob, or ``None`` when the file is absent on that side.

    Returns
    -------
    :
        The blob's text, or ``None`` when there is no blob.
    """
    if path is None:
        return None
    return path.read_bytes().decode("utf-8", errors="replace")


def _text_lines(text: str | None, name: str) -> list[str]:
    """
    Split text into diffable lines.

    JSON is re-serialised with indentation first, because a minified bundle would otherwise
    diff as a single unreadable line.

    Parameters
    ----------
    text
        The content, or ``None`` when the file is absent on that side.
    name
        The file's name, used to decide whether it is JSON.

    Returns
    -------
    :
        The lines to diff.
    """
    if text is None:
        return []
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


def _build_text_diff(old_lines: list[str], new_lines: list[str], *, has_old: bool, has_new: bool) -> TextDiff:
    """
    Build a :class:`TextDiff` from two already decoded sides.

    Parameters
    ----------
    old_lines
        The base side's lines, empty when it is absent.
    new_lines
        The head side's lines, empty when it is absent.
    has_old
        Whether the file exists on the base ref, which labels the diff header.
    has_new
        Whether the file exists on HEAD, which labels the diff header.

    Returns
    -------
    :
        The diff, or a note when the two sides decode identically.
    """
    lines, elided = _diff_lines(
        old_lines,
        new_lines,
        fromfile="old" if has_old else "(absent)",
        tofile="new" if has_new else "(absent)",
    )
    if not lines:
        return TextDiff(lines=(), note="identical after decoding", elided=0)
    return TextDiff(lines=lines, note=None, elided=elided)


def committed_diff(change: CommittedChange) -> TextDiff:
    """
    Build the unified diff of one committed regression artefact.

    Parameters
    ----------
    change
        The artefact that moved, carrying both sides' content.

    Returns
    -------
    :
        The diff, or a note explaining why there is not one.
    """
    if change.old is not None and change.old_text is None:
        return TextDiff(lines=(), note="could not read the base version from git", elided=0)
    if change.new is not None and change.new_text is None:
        return TextDiff(lines=(), note="could not read the working tree version", elided=0)
    return _build_text_diff(
        _text_lines(change.old_text, change.name),
        _text_lines(change.new_text, change.name),
        has_old=change.old is not None,
        has_new=change.new is not None,
    )


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
    return _build_text_diff(
        _text_lines(_decode(old), name),
        _text_lines(_decode(new), name),
        has_old=old is not None,
        has_new=new is not None,
    )


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


def _header_diff(old_lines: list[str], new_lines: list[str]) -> tuple[DiffLine, ...]:
    """
    Merge two headers into one listing, tagging each line.

    Unlike a unified diff this keeps every line, so the listing doubles as the file's
    description rather than only naming what moved.

    Parameters
    ----------
    old_lines
        The base side's header, empty when that side is absent.
    new_lines
        The head side's header, empty when that side is absent.

    Returns
    -------
    :
        Every line, in reading order, tagged ``context``, ``add`` or ``remove``. The two
        character marker is kept on the text so the tags survive without colour.
    """
    kinds = {" ": "context", "-": "remove", "+": "add"}
    return tuple(
        DiffLine(kind=kinds[line[0]], text=line)
        for line in difflib.Differ().compare(old_lines, new_lines)
        if line[0] in kinds
    )


def _variable(dataset: xr.Dataset | None, name: str) -> xr.DataArray | None:
    """
    Look one data variable up on one side.

    Parameters
    ----------
    dataset
        The open dataset, or ``None`` when the file is absent on that side.
    name
        The variable's name.

    Returns
    -------
    :
        The variable, or ``None`` when this side does not carry it.
    """
    if dataset is None or name not in dataset.data_vars:
        return None
    return dataset[name]


def _values(variable: xr.DataArray | None) -> np.ndarray | None:
    """
    Read one variable as a float array.

    Parameters
    ----------
    variable
        The variable, or ``None`` when it is absent on that side.

    Returns
    -------
    :
        The values in their stored dtype, or ``None`` when the variable is absent or not
        numeric. The dtype is kept because an ``int64`` past 2**53 does not survive a cast to
        float, so two adjacent values would compare equal.
    """
    if variable is None or not np.issubdtype(variable.dtype, np.number):
        return None
    return np.asarray(variable.values)


def _shape(variable: xr.DataArray | None) -> str | None:
    """
    Render one variable's shape.

    Parameters
    ----------
    variable
        The variable, or ``None`` when it is absent on that side.

    Returns
    -------
    :
        For example ``180x360``, ``scalar`` for a zero-dimensional variable, or ``None``
        when the variable is absent.
    """
    if variable is None:
        return None
    return "x".join(str(size) for size in variable.shape) if variable.shape else "scalar"


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
    return (
        float(np.nanmin(values)),
        float(np.nanmax(values)),
        float(np.nanmean(values, dtype=float)),
        nan_count,
    )


def _compare(
    old: np.ndarray | None, new: np.ndarray | None, scale: float
) -> tuple[float | None, float | None, int | None]:
    """
    Compare two sides of the same variable cell by cell.

    NaN counts as equal to NaN, because a masked cell staying masked is not a change.

    Parameters
    ----------
    old
        The base side, or ``None`` when absent or not numeric.
    new
        The head side, or ``None`` when absent or not numeric.
    scale
        The base side's largest magnitude, which the relative difference is measured against.

    Returns
    -------
    :
        The largest absolute difference, the same relative to ``scale``, and the number of
        cells that differ. The two differences are ``None`` when a cell moved between NaN and
        a number, because that gap has no magnitude and the finite maximum would read as zero.
        All three are ``None`` when the sides cannot be compared.
    """
    if old is None or new is None or old.shape != new.shape:
        return None, None, None
    old_nan = np.isnan(old)
    new_nan = np.isnan(new)
    cells_differ = int(np.sum(~((old == new) | (old_nan & new_nan))))
    if np.any(old_nan != new_nan):
        return None, None, cells_differ
    diff = np.abs(new.astype(float) - old.astype(float))
    max_abs = 0.0 if np.isnan(diff).all() else float(np.nanmax(diff))
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
    old_variable = _variable(old, name)
    new_variable = _variable(new, name)
    shape_old = _shape(old_variable)
    shape_new = _shape(new_variable)
    old_values = _values(old_variable)
    new_values = _values(new_variable)
    min_old, max_old, mean_old, nan_old = _summarise(old_values)
    min_new, max_new, mean_new, nan_new = _summarise(new_values)
    scale = max(abs(min_old), abs(max_old)) if min_old is not None and max_old is not None else 0.0
    max_abs_diff, max_rel_diff, cells_differ = _compare(old_values, new_values, scale)
    tolerance = scale * NOISE_REL_TOLERANCE
    shape = Pair(old=shape_old, new=shape_new)
    return StatRow(
        name=name,
        shape=shape,
        minimum=Pair(old=min_old, new=min_new, tolerance=tolerance),
        maximum=Pair(old=max_old, new=max_new, tolerance=tolerance),
        mean=Pair(old=mean_old, new=mean_new, tolerance=tolerance),
        nan=Pair(old=nan_old, new=nan_new),
        max_abs_diff=max_abs_diff,
        max_rel_diff=max_rel_diff,
        cells_differ=cells_differ,
        moved=(cells_differ or 0) > 0 or shape.changed,
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
            for dataset in (old_ds, new_ds):
                if dataset is not None and dataset.nbytes > MAX_DECODED_BYTES:
                    return NetcdfDiff(
                        header=(),
                        header_old=(),
                        header_new=(),
                        rows=(),
                        note=f"decodes to too much to analyse ({dataset.nbytes:,} B)",
                    )
            header_old = _header(old_ds)
            header_new = _header(new_ds)
            header = _header_diff(header_old, header_new)
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
        return NetcdfDiff(header=(), header_old=(), header_new=(), rows=(), note=f"could not open: {exc}")
    return NetcdfDiff(
        header=header,
        header_old=tuple(header_old),
        header_new=tuple(header_new),
        rows=rows,
        note=None,
    )


def _fetch_side(
    store: NativeStore,
    entry: NativeEntry | None,
    workdir: Path,
    *,
    limit: int,
    oversize: str,
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


def _fetch_pair(
    change: FileChange,
    store: NativeStore,
    workdir: Path,
    *,
    limit: int,
    oversize: str,
) -> tuple[Path | None, Path | None, str | None]:
    """
    Fetch both sides of a file.

    Parameters
    ----------
    change
        The file that moved.
    store
        The store to read blobs from.
    workdir
        Directory fetched blobs are written into.
    limit
        Largest blob worth downloading.
    oversize
        Note to return when a blob is past ``limit``.

    Returns
    -------
    :
        An ``(old, new, note)`` triple. ``note`` is the first failure of the two sides, and
        the paths should be ignored once it is set.
    """
    old_path, old_note = _fetch_side(store, change.old, workdir, limit=limit, oversize=oversize)
    new_path, new_note = _fetch_side(store, change.new, workdir, limit=limit, oversize=oversize)
    return old_path, new_path, old_note or new_note


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

    old_path, new_path, note = _fetch_pair(
        change, store, workdir, limit=MAX_FETCH_BYTES, oversize="too large to diff"
    )
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
        return NetcdfDiff(header=(), header_old=(), header_new=(), rows=(), note="fetching disabled")

    old_path, new_path, note = _fetch_pair(
        change, store, workdir, limit=NETCDF_FETCH_BYTES, oversize="too large to analyse"
    )
    if note is not None:
        return NetcdfDiff(header=(), header_old=(), header_new=(), rows=(), note=note)
    return netcdf_diff(old_path, new_path)


def _analyse_file(
    change: FileChange,
    store: NativeStore,
    *,
    fetch: bool,
    workdir: Path,
) -> AnalysedFile:
    """
    Build the URLs and, for text and NetCDF, the analysis of one native file.

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
        netcdf=_netcdf_for(change, store, fetch=fetch, workdir=workdir),
        size_delta=change.new.size - change.old.size if change.old and change.new else None,
    )


def _of_kind(files: tuple[AnalysedFile, ...], kind: FileKind) -> tuple[AnalysedFile, ...]:
    """
    Select the files of one kind, keeping their order.

    Parameters
    ----------
    files
        The analysed files.
    kind
        The kind to keep.

    Returns
    -------
    :
        The matching files.
    """
    return tuple(file for file in files if file.change.kind is kind)


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


def _baseline_entries(case: CaseChange) -> dict[str, tuple[int | None, str | None]]:
    """
    Gather every file the baseline captures, keyed by its path within the test case.

    The committed bundle sits under its own directory, which is where a run writes it, so the
    listing matches the layout on disk rather than splitting the capture by where it is stored.
    Statuses come from the collected case, which is what decided that a file moved at all.

    Parameters
    ----------
    case
        The collected case.

    Returns
    -------
    :
        ``{path: (size, status)}``, where the status is ``None`` for a file that did not move.
        The size is ``None`` for a committed artefact, whose manifest entry carries only a digest.
    """
    moved = {file.name: file.status for file in case.files}
    native = {**(case.base.native if case.base else {}), **(case.head.native if case.head else {})}
    entries: dict[str, tuple[int | None, str | None]] = {
        name: (entry.size, moved.get(name)) for name, entry in native.items()
    }

    moved_committed = {artefact.name: artefact.status for artefact in case.committed}
    committed = {**(case.base.committed if case.base else {}), **(case.head.committed if case.head else {})}
    for name in committed:
        entries[f"{COMMITTED_DIRNAME}/{name}"] = (None, moved_committed.get(name))
    return entries


def baseline_tree(case: CaseChange) -> tuple[TreeNode, ...]:
    """
    Flatten the captured baseline's files into an indented folder listing.

    Every captured file is listed, not only the ones that moved, because the listing is what
    tells a reviewer what the baseline actually holds.

    Parameters
    ----------
    case
        The collected case.

    Returns
    -------
    :
        One row per directory and file, in path order.
    """
    entries = _baseline_entries(case)
    rows: list[TreeNode] = []
    previous: tuple[str, ...] = ()
    for path in sorted(entries):
        *directories, filename = PurePosixPath(path).parts
        shared = 0
        while shared < min(len(previous), len(directories)) and previous[shared] == directories[shared]:
            shared += 1
        for depth in range(shared, len(directories)):
            rows.append(TreeNode(name=directories[depth], depth=depth, is_dir=True, size=None, status=None))
        previous = tuple(directories)
        size, status = entries[path]
        rows.append(
            TreeNode(
                name=filename,
                depth=len(directories),
                is_dir=False,
                size=size,
                status=status,
            )
        )
    return tuple(rows)


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
                committed=tuple(
                    AnalysedCommitted(change=change, text=committed_diff(change)) for change in case.committed
                ),
                tree=baseline_tree(case),
                back_link="/".join([*[".."] * depth, "index.html"]),
            )
        )
    return AnalysedReport(
        report=report,
        store_url=store_url,
        cases=tuple(cases),
        kinds=tuple(kind.value for kind in FileKind),
    )
