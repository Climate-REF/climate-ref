"""
Native-store commands: ``ref test-cases sync``.

``sync`` warms the local cache with the native blobs referenced by committed
manifests.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from loguru import logger

from climate_ref.cli.test_cases._app import app
from climate_ref.cli.test_cases._common import (
    _iter_test_cases,
    _validate_provider_in_registry,
    _validate_requested_filters,
)
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

    from climate_ref.provider_registry import ProviderRegistry
    from climate_ref_core.regression.manifest import Manifest
    from climate_ref_core.regression.store import build_native_store
    from climate_ref_core.testing import TestCasePaths

    config: Config = ctx.obj.config
    db = ctx.obj.database
    console: Console = ctx.obj.console

    registry = ProviderRegistry.build_from_config(config, db)
    _validate_provider_in_registry(registry, provider)
    _validate_requested_filters(registry, provider=provider, diagnostic=diagnostic, test_case=test_case)
    store = build_native_store(config.native_store, writable=False)

    # When a specific case is named, a missing manifest is a hard failure.
    named = bool(diagnostic or test_case)
    cases = list(_iter_test_cases(registry, provider=provider, diagnostic=diagnostic, test_case=test_case))

    if not cases:
        logger.warning("No test cases found for the selected filters")
        raise typer.Exit(code=0)

    fetched = 0
    skipped = 0
    failures: list[str] = []

    for diag, tc in cases:
        case_id = f"{diag.provider.slug}/{diag.slug}/{tc.name}"
        paths = TestCasePaths.from_diagnostic(diag, tc.name)
        if paths is None or not paths.manifest.exists():
            if named:
                logger.error(f"No manifest.json for {case_id}; run `ref test-cases mint` first")
                failures.append(case_id)
            continue
        manifest = Manifest.load(paths.manifest)
        for relpath, entry in manifest.native.items():
            if store.has(entry.sha256):
                skipped += 1
                continue
            with tempfile.TemporaryDirectory() as tmp:
                try:
                    store.fetch(entry.sha256, Path(tmp) / "blob")
                except Exception as exc:
                    failures.append(f"{case_id}: cannot serve native blob {entry.sha256} ({relpath}): {exc}")
                    continue
            fetched += 1

    console.print(f"[green]Synced native blobs:[/green] {fetched} fetched, {skipped} already cached")
    if failures:
        console.print("[red]Failed to fetch referenced native blobs:[/red]")
        for failure in failures:
            console.print(f"  - {failure}")
        raise typer.Exit(code=1)
