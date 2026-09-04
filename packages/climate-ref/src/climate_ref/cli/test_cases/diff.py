"""
``ref test-cases diff``.

Renders an HTML report of every regression baseline that moved on this branch.
Images are shown old and new side by side and text outputs diffed inline.
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
def diff_baselines(  # noqa: PLR0913
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
    upload: Annotated[
        str | None,
        typer.Option(help="Upload the report under this key prefix, e.g. 912/0c7e1d4abc12"),
    ] = None,
    comment_output: Annotated[
        Path | None,
        typer.Option(help="Write the pull request comment markdown here"),
    ] = None,
) -> None:
    """
    Render an HTML report of the regression baselines changed on this branch.

    Compares every committed ``manifest.json`` to its counterpart on ``--base``
    and writes one page per changed test case.

    Exits 0 whether or not anything changed.

    Examples
    --------
        ref test-cases diff --html-dir build/baseline-diff
        ref test-cases diff --base origin/develop --html-dir build/baseline-diff
        ref test-cases diff --html-dir build/baseline-diff --no-fetch
        ref test-cases diff --html-dir build/baseline-diff --upload 912/0c7e1d4abc12
    """
    from climate_ref.baseline_report.analyse import analyse
    from climate_ref.baseline_report.collect import collect
    from climate_ref.baseline_report.render import render_comment, write_site
    from climate_ref.baseline_report.upload import upload_site
    from climate_ref_core.regression.report_store import build_report_store
    from climate_ref_core.regression.store import NativeStoreUnavailableError, build_native_store

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

    base_url = index.parent.resolve().as_uri()
    if upload is not None:
        try:
            report_store = build_report_store(config.report_store, writable=True)
            report_store.preflight()
            upload_site(html_dir, report_store, upload)
            base_url = report_store.url_for(upload)
        except (NotImplementedError, ValueError, ImportError, NativeStoreUnavailableError) as exc:
            logger.error(
                f"Could not upload the report: {exc} Check REF_REPORT_STORE_URL, and for a remote "
                "store REF_REPORT_STORE_ACCESS_KEY_ID / REF_REPORT_STORE_SECRET_ACCESS_KEY plus the "
                "'climate-ref-core[aws]' extra."
            )
            raise typer.Exit(code=1) from exc
        console.print(f"Uploaded the report to {base_url}/index.html")

    if comment_output is not None:
        comment_output.parent.mkdir(parents=True, exist_ok=True)
        comment_output.write_text(render_comment(analysed, base_url), encoding="utf-8")
        console.print(f"Wrote the pull request comment to {comment_output}")
