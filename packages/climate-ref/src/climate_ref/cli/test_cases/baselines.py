"""
The native-baseline lifecycle verbs: ``replay``, ``mint`` and ``build``.

Each is a thin composition over the stages in
:mod:`climate_ref.cli.test_cases._stages`:

- ``replay`` materialises committed native from the store and checks it rebuilds
  the committed bundle within tolerance.
- ``mint`` (re)authors the canonical native baselines and uploads them.
- ``build`` rebuilds the committed bundle from an existing output slot without
  re-executing.
"""

from typing import TYPE_CHECKING, Annotated

import typer
from loguru import logger

from climate_ref.cli.test_cases._app import app
from climate_ref.cli.test_cases._common import (
    _iter_test_cases,
    _validate_provider_in_registry,
    _validate_requested_filters,
    _write_test_case_manifest,
)
from climate_ref.cli.test_cases._stages import (
    StageError,
    baseline_placeholders,
    native_is_stale,
    prepare_slot,
    promote_to_baseline,
    slot_native_relpaths,
    snapshot_native,
    stage_build,
    stage_compare,
    stage_execute,
    stage_materialise,
    stage_rebuild_from_slot,
    stage_upload,
    write_source_stamp,
)
from climate_ref.config import Config

if TYPE_CHECKING:
    from rich.console import Console


@app.command(name="replay")
def replay_test_case(  # noqa: PLR0912, PLR0915
    ctx: typer.Context,
    provider: Annotated[
        str,
        typer.Option(help="Provider slug (required, e.g., 'example', 'ilamb')"),
    ],
    diagnostic: Annotated[
        str | None,
        typer.Option(help="Specific diagnostic slug to replay"),
    ] = None,
    test_case: Annotated[
        str | None,
        typer.Option(help="Specific test case name to replay"),
    ] = None,
    label: Annotated[
        str,
        typer.Option(help="Output slot name under output/ (default: latest)"),
    ] = "latest",
) -> None:
    """
    Replay committed baselines from native blobs and compare to the committed bundle.

    Materialises the committed manifest's native blobs (public, credential-free)
    into a fresh output directory at their stored relative paths, re-runs ``build_execution_result``,
    and compares the regenerated committed bundle to the in-repo copy using the tolerant content comparator.

    Exits non-zero on drift.

    Examples
    --------
        ref test-cases replay --provider example
        ref test-cases replay --provider example --diagnostic global-mean-timeseries
    """
    from climate_ref.provider_registry import ProviderRegistry
    from climate_ref_core.regression import Manifest, verify_committed_integrity
    from climate_ref_core.regression.store import build_native_store
    from climate_ref_core.testing import TestCasePaths

    config: Config = ctx.obj.config
    db = ctx.obj.database
    console: Console = ctx.obj.console

    registry = ProviderRegistry.build_from_config(config, db)
    _validate_provider_in_registry(registry, provider)
    _validate_requested_filters(registry, provider=provider, diagnostic=diagnostic, test_case=test_case)
    store = build_native_store(config.native_store, writable=False)

    cases = list(_iter_test_cases(registry, provider=provider, diagnostic=diagnostic, test_case=test_case))
    if not cases:
        logger.warning(f"No test cases found for provider {provider!r}")
        raise typer.Exit(code=0)

    # When a specific case is named, a missing manifest/catalog is a hard failure.
    named = bool(diagnostic or test_case)

    successes = 0
    failures: list[str] = []

    for diag, tc in cases:
        paths = TestCasePaths.from_diagnostic(diag, tc.name)
        case_id = f"{provider}/{diag.slug}/{tc.name}"
        if paths is None:
            logger.warning(f"Could not determine test case directory for {case_id}")
            continue
        if not paths.manifest.exists():
            message = f"No manifest.json for {case_id}; run `ref test-cases mint` first"
            if named:
                logger.error(message)
                failures.append(case_id)
            else:
                logger.warning(message)
            continue
        if not paths.catalog.exists():
            logger.error(f"No catalog file for {case_id}; run `ref test-cases fetch` first")
            failures.append(case_id)
            continue

        manifest = Manifest.load(paths.manifest)

        # The byte-exact digest check is advisory that the committed baseline is not bitwise identical.
        # The tolerant bundle comparison below may still find them equivalent within tolerance.
        mismatches = verify_committed_integrity(manifest, paths.regression)
        if mismatches:
            logger.warning(
                f"{case_id}: committed baseline differs from the digests recorded in {paths.manifest}"
            )
            for mismatch in mismatches:
                logger.warning(f"  - {mismatch}")

        if not manifest.native:
            logger.error(
                f"{case_id}: manifest has no native baselines — not yet minted. "
                "Run `ref test-cases mint` first."
            )
            failures.append(case_id)
            continue

        slot = prepare_slot(paths, label)
        placeholders = baseline_placeholders(paths, config)
        try:
            source = stage_materialise(
                diag=diag,
                tc=tc,
                paths=paths,
                manifest=manifest,
                store=store,
                slot=slot,
                placeholders=placeholders,
            )
        except Exception as exc:
            logger.error(f"{case_id}: failed to materialise/rebuild native: {exc}")
            failures.append(case_id)
            continue

        stage_build(slot=slot, source=source, placeholders=placeholders)
        cmp_failures, compared = stage_compare(
            slot=slot, paths=paths, slug=diag.slug, expected=manifest.committed
        )
        write_source_stamp(
            slot,
            label=label,
            verb="replay",
            source="materialise",
            test_case_version=manifest.test_case_version,
        )
        if cmp_failures:
            logger.error(f"{case_id}: replay drift detected:\n" + "\n".join(cmp_failures))
            failures.append(case_id)
            continue

        successes += 1
        if mismatches:
            # The byte-level warning above was reconciled by the tolerant comparison.
            logger.info(
                f"Replay reconciled committed bundle: {case_id} "
                f"({len(manifest.native)} native file(s) materialised, "
                f"{len(compared)} bundle file(s) equivalent within tolerance)"
            )
        else:
            logger.info(
                f"Replay matched committed bundle: {case_id} "
                f"({len(manifest.native)} native file(s) materialised, "
                f"{len(compared)} bundle file(s) compared)"
            )

    console.print()
    if failures:
        console.print(f"[yellow]Replay: {successes} passed, {len(failures)} failed[/yellow]")
        console.print("[red]Failed replays:[/red]")
        for case in failures:
            console.print(f"  - {case}")
        raise typer.Exit(code=1)
    console.print(f"[green]All {successes} replay(s) matched the committed bundle[/green]")


@app.command(name="mint")
def mint_native(  # noqa: PLR0912, PLR0913, PLR0915
    ctx: typer.Context,
    provider: Annotated[
        str,
        typer.Option(help="Provider slug (required, e.g., 'example', 'ilamb')"),
    ],
    diagnostic: Annotated[
        str | None,
        typer.Option(help="Specific diagnostic slug to mint"),
    ] = None,
    test_case: Annotated[
        str | None,
        typer.Option(help="Specific test case name to mint"),
    ] = None,
    label: Annotated[
        str,
        typer.Option(help="Output slot name under output/ (default: latest)"),
    ] = "latest",
    from_replay: Annotated[
        bool,
        typer.Option(
            "--from-replay",
            help="Author from a replay of the stored native instead of re-running the diagnostic",
        ),
    ] = False,
    bump_version: Annotated[
        bool,
        typer.Option(help="Increment test_case_version when authoring the manifest"),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(help="Preflight the store and list what would be minted, without running or uploading"),
    ] = False,
) -> None:
    """
    Mint canonical native baselines

    Runs each test case, stores its native snapshot in the writable store,
    and authors the committed ``manifest.json``'s ``native`` block.

    This requires write credentials and is generally run by the CI.

    Examples
    --------
        ref test-cases mint --provider example
        ref test-cases mint --provider example --bump-version
        ref test-cases mint --provider example --dry-run
    """
    from climate_ref.provider_registry import ProviderRegistry
    from climate_ref_core.regression.manifest import Manifest
    from climate_ref_core.regression.store import NativeStoreUnavailableError, build_native_store
    from climate_ref_core.testing import TestCasePaths, load_datasets_from_yaml

    config: Config = ctx.obj.config
    db = ctx.obj.database
    console: Console = ctx.obj.console

    registry = ProviderRegistry.build_from_config(config, db)
    _validate_provider_in_registry(registry, provider)
    _validate_requested_filters(registry, provider=provider, diagnostic=diagnostic, test_case=test_case)
    cases = list(_iter_test_cases(registry, provider=provider, diagnostic=diagnostic, test_case=test_case))
    if not cases:
        logger.warning(f"No test cases found for provider {provider!r}")
        raise typer.Exit(code=0)

    try:
        store = build_native_store(config.native_store, writable=True)
    except (NotImplementedError, ValueError) as exc:
        logger.error(
            "Cannot mint: no writable native store is configured. For the remote (R2) store set "
            "REF_NATIVE_STORE_S3_ENDPOINT_URL and REF_NATIVE_STORE_BUCKET, and authenticate via "
            "REF_NATIVE_STORE_ACCESS_KEY_ID / REF_NATIVE_STORE_SECRET_ACCESS_KEY or a named "
            "REF_NATIVE_STORE_PROFILE; or set REF_NATIVE_STORE_URL to a local file:// path for "
            f"development: {exc}"
        )
        raise typer.Exit(code=1) from exc

    # Preflight the store (credentials / bucket reachability) before running any diagnostics,
    # so a misconfiguration fails fast instead of after the (slow) execution.
    try:
        store.preflight()
    except NativeStoreUnavailableError as exc:
        logger.error(f"Cannot mint: {exc}")
        raise typer.Exit(code=1) from exc

    if dry_run:
        # The store preflight has already passed at this point; report scope and stop before
        # running any diagnostics or uploading anything.
        console.print(f"[cyan]Dry run — would mint {len(cases)} test case(s):[/cyan]")
        for diag, tc in cases:
            console.print(f"  - {provider}/{diag.slug}/{tc.name}")
        console.print("[cyan]Store preflight passed; nothing was run or uploaded.[/cyan]")
        return

    minted = 0
    failures: list[str] = []

    for diag, tc in cases:
        case_id = f"{provider}/{diag.slug}/{tc.name}"
        paths = TestCasePaths.from_diagnostic(diag, tc.name)
        if paths is None:
            logger.warning(f"Could not determine test case directory for {case_id}")
            continue
        if not paths.catalog.exists():
            logger.error(f"No catalog file for {case_id}; run `ref test-cases fetch` first")
            failures.append(case_id)
            continue

        paths.create()
        previous = Manifest.load(paths.manifest) if paths.manifest.exists() else None
        # Validate the --from-replay precondition before wiping the slot, so a never-minted
        # case does not destroy a pre-existing output/<label>/ on its way to failing.
        if from_replay and (previous is None or not previous.native):
            logger.error(f"{case_id}: --from-replay needs an existing minted manifest")
            failures.append(case_id)
            continue

        slot = prepare_slot(paths, label)
        placeholders = baseline_placeholders(paths, config)

        # Populate the slot's native set: either re-execute the diagnostic, or (with
        # --from-replay) materialise the previously minted native from the store. The
        # writable store's fetch/has back the materialise, so no separate read store is needed.
        try:
            if from_replay and previous is not None:  # previous is non-None by the guard above
                source = stage_materialise(
                    diag=diag,
                    tc=tc,
                    paths=paths,
                    manifest=previous,
                    store=store,
                    slot=slot,
                    placeholders=placeholders,
                )
                source_kind = "materialise"
            else:
                datasets = load_datasets_from_yaml(paths.catalog)
                source = stage_execute(
                    config=config,
                    diag=diag,
                    tc=tc,
                    datasets=datasets,
                    slot=slot,
                    execution_dir=None,
                    clean=True,
                )
                source_kind = "execute"
        except StageError as exc:
            logger.error(f"{case_id}: {exc}")
            failures.append(case_id)
            continue
        except Exception as exc:
            logger.error(f"{case_id}: source stage failed during mint: {exc}")
            failures.append(case_id)
            continue

        committed = stage_build(slot=slot, source=source, placeholders=placeholders)
        if from_replay and previous is not None:
            # --from-replay reuses the already-minted native verbatim: stage_materialise hydrated
            # the slot's copy in place (placeholders -> concrete paths) while rebuilding, so a fresh
            # snapshot would re-author manifest.native with non-portable, slot-specific blobs even
            # though the canonical native baseline is unchanged. Preserve it -- only the committed
            # bundle is re-derived here -- which also makes the upload below a verified no-op.
            native = previous.native
        else:
            native = snapshot_native(slot, source=source, placeholders=placeholders)
        errors = stage_upload(
            slot=slot, native=native, store=store, previous=(previous.native if previous else {})
        )
        if errors:
            for error in errors:
                logger.error(f"{case_id}: {error}")
            failures.append(case_id)
            continue

        # Promote the rebuilt bundle and author the committed manifest: the native block
        # is written ONLY here.
        promote_to_baseline(slot, paths)
        if previous is not None:
            version = previous.test_case_version + 1 if bump_version else previous.test_case_version
        else:
            version = 1
        _write_test_case_manifest(
            paths,
            test_case_version=version,
            diagnostic_version=diag.version,
            committed=committed,
            native=native,
        )
        write_source_stamp(slot, label=label, verb="mint", source=source_kind, test_case_version=version)

        minted += 1
        logger.info(
            f"Minted native baseline: {case_id} "
            f"({len(native)} native file(s), {len(committed)} committed file(s), "
            f"test_case_version={version})"
        )

    console.print()
    if failures:
        console.print(f"[yellow]Mint: {minted} minted, {len(failures)} failed[/yellow]")
        console.print("[red]Failed mints:[/red]")
        for case in failures:
            console.print(f"  - {case}")
        raise typer.Exit(code=1)
    console.print(f"[green]Minted {minted} native baseline(s)[/green]")


@app.command(name="build")
def build_test_case(  # noqa: PLR0912, PLR0913, PLR0915
    ctx: typer.Context,
    provider: Annotated[
        str,
        typer.Option(help="Provider slug (required, e.g., 'example', 'ilamb')"),
    ],
    diagnostic: Annotated[
        str | None,
        typer.Option(help="Specific diagnostic slug to build"),
    ] = None,
    test_case: Annotated[
        str | None,
        typer.Option(help="Specific test case name to build"),
    ] = None,
    label: Annotated[
        str,
        typer.Option(help="Output slot to rebuild from (default: latest)"),
    ] = "latest",
    force_regen: Annotated[
        bool,
        typer.Option(help="Promote the rebuilt bundle to the tracked regression baseline"),
    ] = False,
) -> None:
    """
    Rebuild the committed bundle from an existing output slot, without re-executing.

    Reuses the native already materialised in ``output/<label>/`` (by a previous
    ``run`` / ``replay`` / ``mint``) to regenerate the slot's committed bundle. The tracked
    ``regression/`` baseline is only promoted when ``--force-regen`` is given or no baseline
    exists yet, so a rebuild never silently clobbers a committed baseline.

    Examples
    --------
        ref test-cases build --provider example
        ref test-cases build --provider example --label before --force-regen
    """
    from climate_ref.provider_registry import ProviderRegistry
    from climate_ref_core.regression.manifest import Manifest
    from climate_ref_core.testing import TestCasePaths

    config: Config = ctx.obj.config
    db = ctx.obj.database
    console: Console = ctx.obj.console

    registry = ProviderRegistry.build_from_config(config, db)
    _validate_provider_in_registry(registry, provider)
    _validate_requested_filters(registry, provider=provider, diagnostic=diagnostic, test_case=test_case)
    cases = list(_iter_test_cases(registry, provider=provider, diagnostic=diagnostic, test_case=test_case))
    if not cases:
        logger.warning(f"No test cases found for provider {provider!r}")
        raise typer.Exit(code=0)

    built = 0
    failures: list[str] = []

    for diag, tc in cases:
        case_id = f"{provider}/{diag.slug}/{tc.name}"
        paths = TestCasePaths.from_diagnostic(diag, tc.name)
        if paths is None:
            logger.warning(f"Could not determine test case directory for {case_id}")
            continue
        slot = paths.output_slot(label)
        if not slot.exists() or not slot_native_relpaths(slot):
            logger.error(f"{case_id}: no native in output slot {label!r}; run/replay/mint it first")
            failures.append(case_id)
            continue
        if not paths.catalog.exists():
            logger.error(f"No catalog file for {case_id}; run `ref test-cases fetch` first")
            failures.append(case_id)
            continue

        placeholders = baseline_placeholders(paths, config)
        try:
            source = stage_rebuild_from_slot(
                diag=diag, tc=tc, paths=paths, slot=slot, placeholders=placeholders
            )
        except Exception as exc:
            logger.error(f"{case_id}: failed to rebuild bundle from slot: {exc}")
            failures.append(case_id)
            continue

        committed = stage_build(slot=slot, source=source, placeholders=placeholders)
        previous = Manifest.load(paths.manifest) if paths.manifest.exists() else None
        version = previous.test_case_version if previous else 1

        if force_regen or not paths.regression.exists():
            promote_to_baseline(slot, paths)
            native = snapshot_native(slot, source=source, placeholders=placeholders)
            if previous is not None:
                _write_test_case_manifest(
                    paths,
                    test_case_version=previous.test_case_version,
                    diagnostic_version=previous.diagnostic_version,
                    committed=committed,
                    native=previous.native,
                    schema=previous.schema,
                )
                if native_is_stale(native, previous.native):
                    logger.warning(
                        f"{case_id}: committed bundle rebuilt but the native baseline differs; "
                        "re-mint with `ref test-cases mint`"
                    )
            else:
                _write_test_case_manifest(
                    paths,
                    test_case_version=1,
                    diagnostic_version=diag.version,
                    committed=committed,
                    native={},
                )
            logger.info(f"Promoted rebuilt bundle to regression baseline: {paths.regression}")
        else:
            logger.info(
                f"Wrote output slot {slot}/regression "
                "(committed baseline unchanged; use --force-regen to promote it)"
            )

        write_source_stamp(slot, label=label, verb="build", source="rebuild", test_case_version=version)
        built += 1

    console.print()
    if failures:
        console.print(f"[yellow]Build: {built} built, {len(failures)} failed[/yellow]")
        console.print("[red]Failed builds:[/red]")
        for case in failures:
            console.print(f"  - {case}")
        raise typer.Exit(code=1)
    console.print(f"[green]Built {built} committed bundle(s) from output slots[/green]")
