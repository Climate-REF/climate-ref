"""
Write the static HTML report.

Hands frozen objects to Jinja, registers the formatting filters the templates are allowed,
and writes the pages out.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from jinja2 import Environment, PackageLoader, select_autoescape

from climate_ref.baseline_report.analyse import SHORT_DIGEST

if TYPE_CHECKING:
    from climate_ref.baseline_report.analyse import AnalysedCase, AnalysedReport


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
    Build the Jinja environment the report is rendered with.

    Returns
    -------
    :
        The environment, with the three permitted filters registered.
    """
    env = Environment(
        loader=PackageLoader("climate_ref.baseline_report", "templates"),
        autoescape=select_autoescape(["html", "j2"]),
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters["bytes"] = _format_bytes
    env.filters["signed"] = _format_signed
    env.filters["short"] = _format_short
    return env


_env = _build_env()


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
