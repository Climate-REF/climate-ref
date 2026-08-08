"""
Native-store commands: ``ref test-cases sync``.

``sync`` warms the local cache with the native blobs referenced by committed
manifests.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from climate_ref.cli.test_cases._app import app
from climate_ref.cli.test_cases._common import VerbDriver
from climate_ref.config import Config

if TYPE_CHECKING:
    from rich.console import Console


@app.command(name="sync")
def sync_native(
    ctx: typer.Context,
    provider: Annotated[
        str | None,
        typer.Option(help="Limit sync to a single provider slug"),
    ] = None,
    diagnostic: Annotated[
        str | None,
        typer.Option(help="Limit sync to a single diagnostic slug"),
    ] = None,
    test_case: Annotated[
        str | None,
        typer.Option(help="Limit sync to a single test case name"),
    ] = None,
) -> None:
    """
    Fetch native baseline blobs referenced by committed manifests into the store cache.

    Reads each committed ``manifest.json``'s ``native`` block
    and ensures every referenced blob is present in the read store (public, credential-free).
    Blobs already cached are skipped (idempotent).
    A referenced digest the store cannot serve is a hard failure.

    Examples
    --------
        ref test-cases sync                  # Sync all providers
        ref test-cases sync --provider ilamb # Sync a single provider
    """
    import tempfile

    from climate_ref_core.regression.manifest import Manifest
    from climate_ref_core.regression.store import build_native_store

    config: Config = ctx.obj.config
    console: Console = ctx.obj.console

    driver = VerbDriver(ctx, provider=provider, diagnostic=diagnostic, test_case=test_case)
    store = build_native_store(config.native_store, writable=False)
    driver.exit_if_empty("No test cases found for the selected filters")

    fetched = 0
    skipped = 0

    for _diag, _tc, paths, case_id in driver.ready_cases(require_manifest=True):
        manifest = Manifest.load(paths.manifest)
        for relpath, entry in manifest.native.items():
            if store.has(entry.sha256):
                skipped += 1
                continue
            with tempfile.TemporaryDirectory() as tmp:
                try:
                    store.fetch(entry.sha256, Path(tmp) / "blob")
                except Exception as exc:
                    driver.fail(f"{case_id}: cannot serve native blob {entry.sha256} ({relpath}): {exc}")
                    continue
            fetched += 1

    console.print(f"[green]Synced native blobs:[/green] {fetched} fetched, {skipped} already cached")
    if driver.failures:
        console.print("[red]Failed to fetch referenced native blobs:[/red]")
        for failure in driver.failures:
            console.print(f"  - {failure}")
        raise typer.Exit(code=1)
