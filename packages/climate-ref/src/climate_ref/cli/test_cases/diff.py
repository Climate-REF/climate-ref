"""
``ref test-cases diff``.

Renders an HTML report of every regression baseline that moved on this branch, with images
shown old and new side by side and text outputs diffed inline.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from loguru import logger

from climate_ref.cli._git_utils import get_repo_for_path
from climate_ref.cli.test_cases._app import app

if TYPE_CHECKING:
    from rich.console import Console

    from climate_ref.config import Config


@app.command(name="diff")
def diff_baselines(
    ctx: typer.Context,
    html_dir: Annotated[Path, typer.Option(help="Directory to write the HTML report into")],
    base: Annotated[
        str,
        typer.Option(help="Git ref to compare against (the PR base branch)"),
    ] = "origin/main",
    no_fetch: Annotated[
        bool,
        typer.Option("--no-fetch", help="Skip blob downloads and report sizes only"),
    ] = False,
) -> None:
    """
    Render an HTML report of the regression baselines changed on this branch.

    Compares every committed ``manifest.json`` to its counterpart on ``--base`` and writes one page
    per changed test case, with images shown old and new side by side.
    Exits 0 whether or not anything changed. This reports, it does not gate.

    Examples
    --------
        ref test-cases diff --html-dir build/baseline-diff
        ref test-cases diff --base origin/develop --html-dir build/baseline-diff
        ref test-cases diff --html-dir build/baseline-diff --no-fetch
    """
    from climate_ref.baseline_report.analyse import analyse
    from climate_ref.baseline_report.collect import collect
    from climate_ref.baseline_report.render import write_site
    from climate_ref_core.regression.store import build_native_store

    config: Config = ctx.obj.config
    console: Console = ctx.obj.console

    repo = get_repo_for_path(Path.cwd())
    if repo is None:
        logger.error("test-cases diff must be run inside a git repository")
        raise typer.Exit(code=1)

    report = collect(repo, base)
    store = build_native_store(config.native_store, writable=False)
    with tempfile.TemporaryDirectory() as workdir:
        analysed = analyse(report, store, fetch=not no_fetch, workdir=Path(workdir))
        index = write_site(analysed, html_dir)

    console.print(f"Wrote {len(analysed.cases)} case page(s) to {index}")
