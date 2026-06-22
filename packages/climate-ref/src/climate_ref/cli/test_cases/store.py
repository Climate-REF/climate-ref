"""
Native-store commands: ``ref test-cases sync`` and ``ref test-cases check-store``.

``sync`` warms the local cache with the native blobs referenced by committed
manifests; ``check-store`` preflights the writable store's credentials before a
mint.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from loguru import logger

from climate_ref.cli.test_cases._app import app
from climate_ref.cli.test_cases._common import _iter_test_cases, _validate_provider_in_registry
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
    store = build_native_store(config.native_store, writable=False)

    # When a specific case is named, a missing manifest is a hard failure.
    named = bool(diagnostic or test_case)

    fetched = 0
    skipped = 0
    failures: list[str] = []

    for diag, tc in _iter_test_cases(registry, provider=provider, diagnostic=diagnostic, test_case=test_case):
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


@app.command(name="check-store")
def check_store(
    ctx: typer.Context,
) -> None:
    """
    Check connectivity and credentials for the writable native baseline store.

    Builds the writable store from the configuration and preflights it (an authenticated
    no-op probe) without running any diagnostics or uploading anything. Use this to confirm a
    mint will work — that the credentials (REF_NATIVE_STORE_PROFILE or the access-key env
    vars) and the bucket are correct — before a slow mint run.

    Examples
    --------
        ref test-cases check-store
        REF_NATIVE_STORE_PROFILE=my-profile ref test-cases check-store
    """
    from climate_ref_core.regression.store import NativeStoreUnavailableError, build_native_store

    config: Config = ctx.obj.config
    console: Console = ctx.obj.console

    try:
        store = build_native_store(config.native_store, writable=True)
    except (NotImplementedError, ValueError) as exc:
        logger.error(
            "Native store is not configured for writing. For the remote (R2) store set "
            "REF_NATIVE_STORE_S3_ENDPOINT_URL and REF_NATIVE_STORE_BUCKET, and authenticate via "
            "REF_NATIVE_STORE_ACCESS_KEY_ID / REF_NATIVE_STORE_SECRET_ACCESS_KEY or a named "
            f"REF_NATIVE_STORE_PROFILE; or set REF_NATIVE_STORE_URL to a local file:// path: {exc}"
        )
        raise typer.Exit(code=1) from exc

    try:
        store.preflight()
    except NativeStoreUnavailableError as exc:
        logger.error(str(exc))
        raise typer.Exit(code=1) from exc

    console.print("[green]Native store OK:[/green] credentials accepted and the store is reachable.")
