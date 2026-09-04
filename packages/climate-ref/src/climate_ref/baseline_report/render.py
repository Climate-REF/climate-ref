"""
Write the static HTML report.

Hands frozen objects to Jinja, registers the formatting filters the templates are allowed,
and writes the pages out.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from attrs import frozen
from jinja2 import Environment, PackageLoader, select_autoescape

from climate_ref.baseline_report.analyse import NOISE_REL_TOLERANCE, SHORT_DIGEST

if TYPE_CHECKING:
    from collections.abc import Sequence

    from climate_ref.baseline_report.analyse import AnalysedCase, AnalysedReport, KindCounts


def _format_bytes(size: int | None) -> str:
    """
    Render a byte count with thousands separators.

    Parameters
    ----------
    size
        The count, or ``None`` when the file is absent on that side.

    Returns
    -------
    :
        For example ``101,204 B``, or ``-`` when there is no count.
    """
    if size is None:
        return "-"
    return f"{size:,} B"


def _format_signed(delta: int | None) -> str:
    """
    Render a signed byte change.

    Parameters
    ----------
    delta
        The change, or ``None`` when the file exists on only one side.

    Returns
    -------
    :
        For example ``+1,024``, or ``-`` when there is no change to show.
    """
    if delta is None:
        return "-"
    return f"{delta:+,}"


def _format_num(value: float | int | None) -> str:
    """
    Render a statistic.

    Parameters
    ----------
    value
        The number, or ``None`` when it could not be computed. Counts arrive as ``int`` and
        measurements as ``float``, and each reads better in its own format.

    Returns
    -------
    :
        An integer with thousands separators, a float to six significant figures, or ``-``.
    """
    if value is None:
        return "-"
    if isinstance(value, int):
        return f"{value:,}"
    return f"{value:.6g}"


def _format_dash(value: str | None) -> str:
    """
    Render a string that may be absent.

    Parameters
    ----------
    value
        The string, or ``None``.

    Returns
    -------
    :
        The string, or ``-`` when there is not one.
    """
    return "-" if value is None else value


def _format_short(digest: str | None) -> str:
    """
    Render the readable prefix of a digest.

    Parameters
    ----------
    digest
        The digest, or ``None``.

    Returns
    -------
    :
        The first twelve hex characters, or ``-`` when there is no digest.
    """
    if digest is None:
        return "-"
    return digest[:SHORT_DIGEST]


def _build_env() -> Environment:
    """
    Build the Jinja environment the templates in this package are rendered with.

    Escaping keys off the template name, so ``comment.md.j2`` renders unescaped.
    It is markdown, where an escaped case label would show as HTML entities on GitHub.

    Returns
    -------
    :
        The environment, with the formatting filters the templates use.
    """
    env = Environment(
        loader=PackageLoader("climate_ref.baseline_report", "templates"),
        autoescape=select_autoescape(enabled_extensions=("html.j2", "html")),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["bytes"] = _format_bytes
    env.filters["signed"] = _format_signed
    env.filters["num"] = _format_num
    env.filters["dash"] = _format_dash
    env.filters["short"] = _format_short
    env.globals["noise_tolerance"] = _format_num(NOISE_REL_TOLERANCE)
    return env


_env = _build_env()


@frozen
class CommentLink:
    """One earlier report for the same pull request."""

    label: str
    """Short name for the report, typically the head sha it was built from."""

    url: str
    """Where that report is hosted."""


@frozen
class CommentRow:
    """One test case's row in the pull request comment table."""

    label: str
    """The test case's label, for example ``example/diag/case``."""

    versions: str
    """``v3 -> v4``, or ``new`` / ``removed`` when the case only exists on one side."""

    counts: tuple[str, ...]
    """The ``+a ~c -r`` shorthand per file kind, in the report's column order."""

    url: str
    """Link to the case's page in the hosted report."""


def _shorthand(counts: KindCounts) -> str:
    """
    Summarise one kind's changes as a short ``+a ~c -r`` string.

    Parameters
    ----------
    counts
        The tallied counts for one file kind.

    Returns
    -------
    :
        For example ``+1 ~2``, or ``none`` when nothing of that kind changed.
    """
    parts = [
        f"+{counts.added}" if counts.added else "",
        f"~{counts.changed}" if counts.changed else "",
        f"-{counts.removed}" if counts.removed else "",
    ]
    return " ".join(p for p in parts if p) or "none"


def _versions(case: AnalysedCase) -> str:
    """
    Describe how a case's version moved.

    Parameters
    ----------
    case
        The analysed case.

    Returns
    -------
    :
        ``v3 -> v4``, or ``new`` / ``removed`` when the case only exists on one side.
    """
    if case.change.is_removed:
        return "removed"
    if case.change.is_new:
        return "new"
    base, head = case.change.base, case.change.head
    assert base is not None and head is not None
    return f"v{base.test_case_version} -> v{head.test_case_version}"


def _comment_row(case: AnalysedCase, base_url: str) -> CommentRow:
    """
    Build one table row from an analysed case.

    Parameters
    ----------
    case
        The analysed case.
    base_url
        Where the report is hosted, without a trailing slash.

    Returns
    -------
    :
        The row.
    """
    return CommentRow(
        label=case.change.label,
        versions=_versions(case),
        counts=tuple(_shorthand(count) for count in case.counts),
        url=f"{base_url}/{case.change.label}/index.html",
    )


def render_comment(
    report: AnalysedReport,
    base_url: str,
    previous: Sequence[tuple[str, str]] = (),
) -> str:
    """
    Render the pull request comment as markdown.

    The comment is one row per changed case and a link into the hosted report,
    so it stays well inside GitHub's comment size limit however many files moved.

    Parameters
    ----------
    report
        The analysed report.
    base_url
        Where the report is hosted, for example ``https://reports.example/912/0c7e1d4abc12``.
        A trailing slash is ignored.
    previous
        ``(label, url)`` pairs for earlier reports on the same pull request.

    Returns
    -------
    :
        The comment's markdown, ending in the marker the CI job finds it by.
    """
    root = base_url.rstrip("/")
    return _env.get_template("comment.md.j2").render(
        report=report,
        cases=[_comment_row(case, root) for case in report.cases],
        index_url=f"{root}/index.html",
        previous=[CommentLink(label=label, url=url) for label, url in previous],
    )


def render_index(report: AnalysedReport) -> str:
    """
    Render the overview page.

    Parameters
    ----------
    report
        The analysed report.

    Returns
    -------
    :
        The page's HTML.
    """
    return _env.get_template("index.html.j2").render(report=report)


def render_case(report: AnalysedReport, case: AnalysedCase) -> str:
    """
    Render one test case's page.

    Parameters
    ----------
    report
        The analysed report, which carries the header details.
    case
        The case to render.

    Returns
    -------
    :
        The page's HTML.
    """
    return _env.get_template("case.html.j2").render(report=report, case=case)


def write_site(report: AnalysedReport, out_dir: Path) -> Path:
    """
    Write the overview ``index.html`` plus one ``index.html`` per case, under the case label.

    Parameters
    ----------
    report
        The analysed report.
    out_dir
        Directory to write the site into. Created if it does not exist.

    Returns
    -------
    :
        Path of the index page.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    index = out_dir / "index.html"
    index.write_text(render_index(report), encoding="utf-8")
    for case in report.cases:
        page = out_dir / case.change.label / "index.html"
        page.parent.mkdir(parents=True, exist_ok=True)
        page.write_text(render_case(report, case), encoding="utf-8")
    return index
