"""
Test data management commands for diagnostic development.

These commands are intended for developers working on diagnostics and require
a source checkout of the project with test data directories available.
"""

import shutil
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer
from loguru import logger

from climate_ref.config import Config
from climate_ref.provider_registry import ProviderRegistry
from climate_ref_core.datasets import SourceDatasetType
from climate_ref_core.diagnostics import Diagnostic
from climate_ref_core.exceptions import (
    DatasetResolutionError,
    NoTestDataSpecError,
    TestCaseNotFoundError,
)

app = typer.Typer(help=__doc__)


@app.command(name="fetch")
def fetch_test_data(  # noqa: PLR0912
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
    output_directory: Annotated[
        Path | None,
        typer.Option(help="Output directory for test data"),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option(help="Show what would be fetched without downloading"),
    ] = False,
    symlink: Annotated[
        bool,
        typer.Option(help="Symlink files instead of copying (default: True)"),
    ] = True,
) -> None:
    """
    Fetch test data from ESGF for running diagnostic tests.

    Downloads full-resolution ESGF data based on diagnostic test_data_spec.
    Use --provider or --diagnostic to limit scope.

    Examples
    --------
        ref testing fetch                   # Fetch all test data
        ref testing fetch --provider ilamb  # Fetch ILAMB test data only
        ref testing fetch --diagnostic ecs  # Fetch ECS diagnostic data
    """
    from climate_ref.testing import TEST_DATA_DIR  # noqa: PLC0415
    from climate_ref_core.esgf import ESGFFetcher  # noqa: PLC0415

    config = ctx.obj.config
    db = ctx.obj.database

    # Determine output directory
    if output_directory is None:
        if TEST_DATA_DIR is None:
            logger.error("Test data directory not found. Please specify --output-directory")
            raise typer.Exit(code=1)
        output_directory = TEST_DATA_DIR / "esgf-data"

    # Build provider registry to access diagnostics
    registry = ProviderRegistry.build_from_config(config, db)

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
        logger.warning("No diagnostics with test_data_spec found")
        raise typer.Exit(code=0)

    logger.info(f"Found {len(diagnostics_to_process)} diagnostics with test data specifications")

    if dry_run:
        for diag in diagnostics_to_process:
            logger.info(f"Would fetch data for: {diag.provider.slug}/{diag.slug}")
            if diag.test_data_spec:
                for tc in diag.test_data_spec.test_cases:
                    if test_case and tc.name != test_case:
                        continue
                    logger.info(f"  Test case: {tc.name} - {tc.description}")
                    if tc.requests:
                        for req in tc.requests:
                            logger.info(f"    Request: {req.slug} ({req.source_type})")
        return

    # Create fetcher and process requests
    fetcher = ESGFFetcher(output_dir=output_directory)

    for diag in diagnostics_to_process:
        logger.info(f"Fetching data for: {diag.provider.slug}/{diag.slug}")
        if diag.test_data_spec:
            for tc in diag.test_data_spec.test_cases:
                if test_case and tc.name != test_case:
                    continue
                if tc.requests:
                    logger.info(f"  Processing test case: {tc.name}")
                    for req in tc.requests:
                        fetcher.fetch_request(req, symlink=symlink)


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
    """
    config = ctx.obj.config
    db = ctx.obj.database
    console = ctx.obj.console

    # Build provider registry to access diagnostics
    registry = ProviderRegistry.build_from_config(config, db)

    from rich.table import Table  # noqa: PLC0415

    table = Table(title="Test Data Specifications")
    table.add_column("Provider", style="cyan")
    table.add_column("Diagnostic", style="green")
    table.add_column("Test Case", style="yellow")
    table.add_column("Description")
    table.add_column("Requests", justify="right")

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
                )
                continue

            for tc in diag.test_data_spec.test_cases:
                num_requests = len(tc.requests) if tc.requests else 0
                table.add_row(
                    provider_instance.slug,
                    diag.slug,
                    tc.name,
                    tc.description,
                    str(num_requests),
                )

    console.print(table)


def _build_esgf_data_catalog() -> dict[SourceDatasetType, pd.DataFrame]:
    """Build data catalog from local ESGF test data."""
    from climate_ref.datasets.cmip6 import CMIP6DatasetAdapter  # noqa: PLC0415
    from climate_ref.datasets.obs4mips import Obs4MIPsDatasetAdapter  # noqa: PLC0415
    from climate_ref.testing import TEST_DATA_DIR  # noqa: PLC0415

    data_catalog: dict[SourceDatasetType, pd.DataFrame] = {}

    if TEST_DATA_DIR is None:
        return data_catalog

    esgf_data_dir = TEST_DATA_DIR / "esgf-data"

    cmip6_dir = esgf_data_dir / "CMIP6"
    if cmip6_dir.exists():
        cmip6_adapter = CMIP6DatasetAdapter()
        try:
            data_catalog[SourceDatasetType.CMIP6] = cmip6_adapter.find_local_datasets(cmip6_dir)
        except Exception as e:
            logger.warning(f"Could not load CMIP6 data: {e}")

    obs_dir = esgf_data_dir / "obs4MIPs"
    if obs_dir.exists():
        obs_adapter = Obs4MIPsDatasetAdapter()
        try:
            data_catalog[SourceDatasetType.obs4MIPs] = obs_adapter.find_local_datasets(obs_dir)
        except Exception as e:
            logger.warning(f"Could not load obs4MIPs data: {e}")

    return data_catalog


def _find_diagnostic(
    registry: ProviderRegistry, provider_slug: str, diagnostic_slug: str
) -> Diagnostic | None:
    """Find a diagnostic by provider and diagnostic slugs."""
    for provider_instance in registry.providers:
        if provider_instance.slug == provider_slug:
            for d in provider_instance.diagnostics():
                if d.slug == diagnostic_slug:
                    return d
    return None


@app.command(name="run")
def run_test_case(  # noqa: PLR0912
    ctx: typer.Context,
    provider: Annotated[
        str,
        typer.Option(help="Provider slug (e.g., 'example', 'ilamb')"),
    ],
    diagnostic: Annotated[
        str,
        typer.Option(help="Diagnostic slug (e.g., 'global-mean-timeseries')"),
    ],
    test_case: Annotated[
        str,
        typer.Option(help="Test case name (e.g., 'default')"),
    ] = "default",
    output_directory: Annotated[
        Path | None,
        typer.Option(help="Output directory for execution results"),
    ] = None,
    force_regen: Annotated[
        bool,
        typer.Option(help="Force regeneration of regression baselines"),
    ] = False,
) -> None:
    """
    Run a specific test case for a diagnostic.

    Executes the diagnostic using pre-defined datasets from the test_data_spec
    and optionally compares against regression baselines.

    Example:
        ref testing run --provider example --diagnostic global-mean-timeseries
        ref testing run --provider ilamb --diagnostic bias --test-case custom
    """
    from climate_ref.testing import (  # noqa: PLC0415
        TEST_DATA_DIR,
        TestCaseRunner,
    )

    config: Config = ctx.obj.config
    db = ctx.obj.database

    # Build provider registry and find the diagnostic
    registry = ProviderRegistry.build_from_config(config, db)
    diag = _find_diagnostic(registry, provider, diagnostic)

    if diag is None:
        logger.error(f"Diagnostic {provider}/{diagnostic} not found")
        raise typer.Exit(code=1)

    # Build data catalog from ESGF test data
    data_catalog = _build_esgf_data_catalog()

    # Create runner and execute
    runner = TestCaseRunner(config=config, data_catalog=data_catalog if data_catalog else None)

    logger.info(f"Running test case {test_case!r} for {provider}/{diagnostic}")

    try:
        result = runner.run(diag, test_case, output_directory)
    except NoTestDataSpecError:
        logger.error(f"Diagnostic {provider}/{diagnostic} has no test_data_spec")
        raise typer.Exit(code=1)
    except TestCaseNotFoundError:
        logger.error(f"Test case {test_case!r} not found for {provider}/{diagnostic}")
        if diag.test_data_spec:
            logger.error(f"Available test cases: {diag.test_data_spec.case_names}")
        raise typer.Exit(code=1)
    except DatasetResolutionError as e:
        logger.error(str(e))
        logger.error("Have you run 'ref testing fetch' first?")
        raise typer.Exit(code=1)
    except Exception as e:
        logger.error(f"Diagnostic execution failed: {e}")
        raise typer.Exit(code=1)

    if result.successful:
        logger.info("Diagnostic execution completed successfully")
        if result.metric_bundle_filename:
            logger.info(f"Metric bundle: {result.to_output_path(result.metric_bundle_filename)}")
        if result.output_bundle_filename:
            logger.info(f"Output bundle: {result.to_output_path(result.output_bundle_filename)}")
    else:
        logger.error("Diagnostic execution failed")
        raise typer.Exit(code=1)

    # Handle regression baseline comparison/regeneration
    if TEST_DATA_DIR is None:
        return

    regression_dir = TEST_DATA_DIR / "regression" / provider / diagnostic
    baseline_file = regression_dir / f"{test_case}_metric.json"

    if force_regen:
        if result.metric_bundle_filename:
            regression_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy(
                result.to_output_path(result.metric_bundle_filename),
                baseline_file,
            )
            logger.info(f"Updated regression baseline: {baseline_file}")
    elif baseline_file.exists():
        logger.info(f"Regression baseline exists at: {baseline_file}")
        logger.info("Use --force-regen to update the baseline")
