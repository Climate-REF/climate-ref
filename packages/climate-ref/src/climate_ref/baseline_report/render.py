"""
Write the static HTML report.

Python decides and templates place, so this module builds no HTML. It hands frozen objects to
Jinja, registers the three formatting filters the templates are allowed, and writes the pages out.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from jinja2 import Environment, PackageLoader, select_autoescape

if TYPE_CHECKING:
    from climate_ref.baseline_report.analyse import AnalysedCase, AnalysedReport

# Digest prefix shown in the report. Long enough to identify a blob, short enough to read.
_SHORT_DIGEST = 12


def _format_bytes(size: object) -> str:
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
    if not isinstance(size, int) or isinstance(size, bool):
        return "-"
    return f"{size:,} B"


def _format_num(value: object) -> str:
    """
    Render a number to four significant figures.

    ``g`` switches to scientific notation past 1e4 on its own, which is where a plain decimal
    stops being readable.

    Parameters
    ----------
    value
        The number, or anything else, which is passed through as text.

    Returns
    -------
    :
        The formatted number.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        return str(value)
    return f"{value:.4g}"


def _format_short(digest: object) -> str:
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
    if not isinstance(digest, str):
        return "-"
    return digest[:_SHORT_DIGEST]


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
    env.filters["num"] = _format_num
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
    Write the overview ``index.html`` plus one ``index.html`` per case, under the case slug.

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
        page = out_dir / case.change.slug / "index.html"
        page.parent.mkdir(parents=True, exist_ok=True)
        page.write_text(render_case(report, case), encoding="utf-8")
    return index
