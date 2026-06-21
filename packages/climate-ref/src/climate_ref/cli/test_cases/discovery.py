"""
``ref test-cases fetch`` and ``ref test-cases list``.

Commands for discovering test cases and fetching their input data from ESGF.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

import typer
from loguru import logger
from rich.table import Table

from climate_ref.cli.test_cases._app import app
from climate_ref.cli.test_cases._catalog import _fetch_and_build_catalog
from climate_ref_core.exceptions import DatasetResolutionError

if TYPE_CHECKING:
    from climate_ref_core.diagnostics import Diagnostic


@app.command(name="fetch")
def fetch_test_data(  # noqa: PLR0912, PLR0913, PLR0915
    ctx: typer.Context,
    provider: Annotated[
        str | None,
        typer.Option(help="Specific provider to fetch data for (e.g., 'esmvaltool', 'ilamb')"),
    ] = None,
    diagnostic: Annotated[
        str | None,
        typer.Option(help="Specific diagnostic slug to fetch data for"),
    ] = None,
    test_case: Annotated[
        str | None,
        typer.Option(help="Specific test case name to fetch data for"),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(help="Show what would be fetched without downloading"),
    ] = False,
    only_missing: Annotated[
        bool,
        typer.Option(help="Only fetch data for test cases without existing catalogs"),
    ] = False,
    force: Annotated[
        bool,
        typer.Option(help="Force overwrite catalog even if unchanged"),
    ] = False,
) -> None:
    """
    Fetch test data from ESGF for running diagnostic tests.

    Downloads full-resolution ESGF data based on diagnostic test_data_spec.
    Use --provider or --diagnostic to limit scope.

    Examples
    --------
        ref test-cases fetch                   # Fetch all test data
        ref test-cases fetch --provider ilamb  # Fetch ILAMB test data only
        ref test-cases fetch --diagnostic ecs  # Fetch ECS diagnostic data
        ref test-cases fetch --only-missing    # Skip test cases with existing catalogs
    """
    from climate_ref.provider_registry import ProviderRegistry
    from climate_ref_core.testing import TestCasePaths

    config = ctx.obj.config
    db = ctx.obj.database

    # Build provider registry to access diagnostics
    registry = ProviderRegistry.build_from_config(config, db)

    # Check if the requested provider exists in the registry
    available_providers = [p.slug for p in registry.providers]
    if provider and provider not in available_providers:
        logger.error(f"Provider '{provider}' is not configured")
        if available_providers:
            logger.error(f"Available providers: {', '.join(sorted(available_providers))}")
        else:
            logger.error("No providers are configured. Check your configuration file.")
        logger.error("To add a provider, update your config file or set REF_DIAGNOSTIC_PROVIDERS")
        raise typer.Exit(code=1)

    # Collect diagnostics to process
    diagnostics_to_process: list[Diagnostic] = []

    for provider_instance in registry.providers:
        if provider and provider_instance.slug != provider:
            continue

        for diag in provider_instance.diagnostics():
            if diagnostic and diag.slug != diagnostic:
                continue
            if diag.test_data_spec is None:
                continue
            diagnostics_to_process.append(diag)

    if not diagnostics_to_process:
        if provider:
            logger.warning(f"No diagnostics with test_data_spec found for provider '{provider}'")
        else:
            logger.warning("No diagnostics with test_data_spec found")
        raise typer.Exit(code=0)

    logger.info(f"Found {len(diagnostics_to_process)} diagnostics with test data specifications")

    if dry_run:  # pragma: no cover
        for diag in diagnostics_to_process:
            logger.info(f"Would fetch data for: {diag.provider.slug}/{diag.slug}")
            if diag.test_data_spec:
                for tc in diag.test_data_spec.test_cases:
                    if test_case and tc.name != test_case:
                        continue
                    # Check if catalog exists when using --only-missing
                    if only_missing:
                        paths = TestCasePaths.from_diagnostic(diag, tc.name)
                        if paths and paths.catalog.exists():
                            logger.info(f"  Test case: {tc.name} - [SKIP: catalog exists]")
                            continue
                    logger.info(f"  Test case: {tc.name} - {tc.description}")
                    if tc.requests:
                        for req in tc.requests:
                            logger.info(f"    Request: {req.slug} ({req.source_type})")
        return

    # Process each diagnostic test case
    for diag in diagnostics_to_process:  # pragma: no cover
        logger.info(f"Fetching data for: {diag.provider.slug}/{diag.slug}")
        if diag.test_data_spec:
            for tc in diag.test_data_spec.test_cases:
                if test_case and tc.name != test_case:
                    continue
                # Skip if catalog exists when using --only-missing
                if only_missing:
                    paths = TestCasePaths.from_diagnostic(diag, tc.name)
                    if paths and paths.catalog.exists():
                        logger.info(f"  Skipping test case: {tc.name} (catalog exists)")
                        continue
                if tc.requests:
                    logger.info(f"  Processing test case: {tc.name}")
                    try:
                        _, catalog_written = _fetch_and_build_catalog(diag, tc, force=force)
                        if not catalog_written:
                            logger.info(f"  Catalog unchanged for {tc.name}")
                    except (DatasetResolutionError, ValueError) as e:
                        logger.warning(f"  Could not build catalog for {tc.name}: {e}")


@app.command(name="list")
def list_cases(
    ctx: typer.Context,
    provider: Annotated[
        str | None,
        typer.Option(help="Filter by provider"),
    ] = None,
) -> None:
    """
    List test cases for all diagnostics.

    Shows which test cases are defined for each diagnostic and their descriptions.
    Also shows whether catalog and regression data exist for each test case.
    """
    from climate_ref.provider_registry import ProviderRegistry
    from climate_ref_core.testing import TestCasePaths

    config = ctx.obj.config
    db = ctx.obj.database
    console = ctx.obj.console

    # Build provider registry to access diagnostics
    registry = ProviderRegistry.build_from_config(config, db)

    # Check if the requested provider exists in the registry
    available_providers = [p.slug for p in registry.providers]
    if provider and provider not in available_providers:
        logger.error(f"Provider '{provider}' is not configured")
        if available_providers:
            logger.error(f"Available providers: {', '.join(sorted(available_providers))}")
        else:
            logger.error("No providers are configured. Check your configuration file.")
        raise typer.Exit(code=1)

    table = Table(title="Test Data Specifications")
    table.add_column("Provider", style="cyan")
    table.add_column("Diagnostic", style="green")
    table.add_column("Test Case", style="yellow")
    table.add_column("Description")
    table.add_column("Requests", justify="right")
    table.add_column("Catalog", justify="center")
    table.add_column("Regression", justify="center")

    for provider_instance in registry.providers:
        if provider and provider_instance.slug != provider:
            continue

        for diag in provider_instance.diagnostics():
            if diag.test_data_spec is None:
                table.add_row(
                    provider_instance.slug,
                    diag.slug,
                    "-",
                    "(no test_data_spec)",
                    "0",
                    "-",
                    "-",
                )
                continue

            for tc in diag.test_data_spec.test_cases:
                num_requests = len(tc.requests) if tc.requests else 0

                # Check if catalog and regression data exist
                paths = TestCasePaths.from_diagnostic(diag, tc.name)
                if paths:
                    catalog_status = "[green]yes[/green]" if paths.catalog.exists() else "[red]no[/red]"
                    regression_status = "[green]yes[/green]" if paths.regression.exists() else "[red]no[/red]"
                else:
                    catalog_status = "[dim]-[/dim]"
                    regression_status = "[dim]-[/dim]"

                table.add_row(
                    provider_instance.slug,
                    diag.slug,
                    tc.name,
                    tc.description,
                    str(num_requests),
                    catalog_status,
                    regression_status,
                )

    console.print(table)
