"""
``ref test-cases migrate-manifests``.

A one-shot, idempotent maintenance command that backfills every committed
``manifest.json`` to the current ``SCHEMA_VERSION``, stamping each case's
``diagnostic_version`` from the diagnostic's in-code ``Diagnostic.version`` only
when the field is absent. An already-recorded ``diagnostic_version`` is preserved,
so re-running never re-stamps a stale bundle and subverts the gate's staleness check.

It iterates the same ``_iter_test_cases`` the CI gate uses, so its coverage is
definitionally aligned with the gate (iterator parity). A pre-bump manifest is
schema-1 and ``Manifest.load`` would reject it, so the raw JSON is read directly
and re-serialised through ``Manifest(...).dump`` to keep the output byte-identical
to what a later ``mint`` would write.
"""

from __future__ import annotations

import json
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


@app.command(name="migrate-manifests")
def migrate_manifests(
    ctx: typer.Context,
    provider: Annotated[
        str | None,
        typer.Option(help="Limit the migration to a single provider slug"),
    ] = None,
    diagnostic: Annotated[
        str | None,
        typer.Option(help="Limit the migration to a single diagnostic slug"),
    ] = None,
    test_case: Annotated[
        str | None,
        typer.Option(help="Limit the migration to a single test case name"),
    ] = None,
) -> None:
    """
    Backfill every committed ``manifest.json`` to the current schema.

    For each test case with an existing manifest, backfills ``diagnostic_version``
    from the diagnostic's in-code ``Diagnostic.version`` when it is absent (an
    already-recorded value is preserved) and rewrites the manifest at the current
    ``SCHEMA_VERSION``. The operation is idempotent: re-running on an already-migrated
    manifest preserves the recorded value and produces identical bytes.

    Examples
    --------
        ref test-cases migrate-manifests
        ref test-cases migrate-manifests --provider ilamb
    """
    from climate_ref.provider_registry import ProviderRegistry
    from climate_ref_core.regression.manifest import SCHEMA_VERSION, Manifest, NativeEntry
    from climate_ref_core.testing import TestCasePaths

    config: Config = ctx.obj.config
    db = ctx.obj.database
    console: Console = ctx.obj.console

    registry = ProviderRegistry.build_from_config(config, db)
    _validate_provider_in_registry(registry, provider)
    _validate_requested_filters(registry, provider=provider, diagnostic=diagnostic, test_case=test_case)
    cases = list(_iter_test_cases(registry, provider=provider, diagnostic=diagnostic, test_case=test_case))

    migrated = 0
    skipped = 0

    for diag, tc in cases:
        case_id = f"{diag.provider.slug}/{diag.slug}/{tc.name}"
        paths = TestCasePaths.from_diagnostic(diag, tc.name)
        if paths is None or not paths.manifest.exists():
            skipped += 1
            continue

        # A pre-bump (schema-1) manifest is rejected by Manifest.load, so read the raw
        # JSON and re-construct, injecting diagnostic_version and the current schema while
        # carrying every other field through unchanged.
        data = json.loads(paths.manifest.read_text(encoding="utf-8"))
        native = {
            relpath: NativeEntry(sha256=entry["sha256"], size=entry["size"])
            for relpath, entry in data["native"].items()
        }
        # Only backfill diagnostic_version when it is absent. Preserving an already-recorded
        # value keeps this a pure backfill: re-running after an authorised version bump (where
        # the manifest legitimately lags the in-code Diagnostic.version until a re-mint) must
        # not silently re-stamp a stale bundle as current and subvert the gate's staleness check.
        diagnostic_version = data.get("diagnostic_version", diag.version)
        manifest = Manifest(
            schema=SCHEMA_VERSION,
            test_case_version=data["test_case_version"],
            diagnostic_version=diagnostic_version,
            committed=dict(data["committed"]),
            native=native,
            catalog_hash=data.get("catalog_hash"),
        )
        # Serialise via dump so the bytes (sort_keys + unconditional `catalog_hash: null`)
        # match exactly what a later mint would write.
        manifest.dump(paths.manifest)
        migrated += 1
        logger.info(f"Migrated {case_id} -> schema {SCHEMA_VERSION}, diagnostic_version {diagnostic_version}")

    console.print()
    console.print(
        f"[green]Migrated {migrated} manifest(s)[/green]"
        + (f"; skipped {skipped} case(s) with no manifest" if skipped else "")
    )
