"""
Test data management commands for diagnostic development.

These commands are intended for developers working on diagnostics and require
a source checkout of the project with test data directories available.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

from climate_ref.cli._git_utils import collect_regression_file_info, get_repo_for_path
from climate_ref.cli._test_case_stages import (
    StageError,
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
from climate_ref.cli._utils import format_size
from climate_ref.config import Config
from climate_ref_core.exceptions import (
    DatasetResolutionError,
    InvalidDiagnosticException,
    NoTestDataSpecError,
    TestCaseNotFoundError,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    import pandas as pd

    from climate_ref.datasets import DatasetAdapter
    from climate_ref.provider_registry import ProviderRegistry
    from climate_ref_core.datasets import ExecutionDatasetCollection, SourceDatasetType
    from climate_ref_core.diagnostics import Diagnostic
    from climate_ref_core.regression.manifest import Manifest, NativeEntry
    from climate_ref_core.testing import TestCase, TestCasePaths


app = typer.Typer(help=__doc__)


def _build_catalog(dataset_adapter: DatasetAdapter, file_paths: list[Path]) -> pd.DataFrame:
    """
    Parses a list of datasets using a dataset adapter

    Parameters
    ----------
    file_paths
        List of files to build a catalog from

    Returns
    -------
    pd.DataFrame
        DataFrame catalog of datasets with metadata and paths
    """
    import pandas as pd

    # Collect unique parent directories since the adapter scans directories
    parent_dirs = list({fp.parent for fp in file_paths})

    catalog_dfs = []
    for parent_dir in parent_dirs:
        try:
            df = dataset_adapter.find_local_datasets(parent_dir)

            # Filter to only include the files we fetched
            fetched_files = {str(fp) for fp in file_paths}
            df = df[df["path"].isin(fetched_files)]
            if df.empty:
                logger.warning(f"No matching files found in catalog for {parent_dir}")
            catalog_dfs.append(df)
        except Exception as e:
            logger.warning(f"Failed to parse {parent_dir}: {e}")

    if not catalog_dfs:
        return pd.DataFrame()
    return pd.concat(catalog_dfs, ignore_index=True)


def _solve_test_case(
    diagnostic: Diagnostic,
    data_catalog: dict[SourceDatasetType, pd.DataFrame],
) -> ExecutionDatasetCollection:
    """
    Solve for test case datasets by applying the diagnostic's data requirements.

    Runs the solver to determine which datasets from the catalog
    satisfy the diagnostic's requirements.
    """
    from climate_ref.solver import solve_executions

    executions = list(solve_executions(data_catalog, diagnostic, diagnostic.provider))

    if not executions:
        raise ValueError(f"No valid executions found for diagnostic {diagnostic.slug}")

    return executions[0].datasets


def _fetch_and_build_catalog(
    diag: Diagnostic,
    tc: TestCase,
    *,
    force: bool = False,
) -> tuple[ExecutionDatasetCollection, bool]:
    """
    Fetch test data and build catalog.

    This function:
    1. Fetches ESGF data using ESGFFetcher (files stored in intake-esgf cache)
    2. Uses CMIP6DatasetAdapter to create a data catalog
    3. Solves for datasets using the diagnostic's data requirements
    4. Writes catalog YAML to .catalogs/{provider}/{diagnostic}/{test_case}.yaml
    5. Returns the solved datasets and whether the catalog was written

    By default, the catalog is only written if the content has changed.
    Use `force=True` to always write.

    Parameters
    ----------
    diag
        The diagnostic to fetch data for
    tc
        The test case to fetch data for
    force
        If True, always write the catalog even if unchanged

    Returns
    -------
    :
        Tuple of (datasets, catalog_was_written)
    """
    from climate_ref.datasets import (
        CMIP6DatasetAdapter,
        CMIP7DatasetAdapter,
        Obs4MIPsDatasetAdapter,
        PMPClimatologyDatasetAdapter,
    )
    from climate_ref_core.datasets import SourceDatasetType
    from climate_ref_core.esgf import ESGFFetcher
    from climate_ref_core.testing import TestCasePaths, save_datasets_to_yaml

    fetcher = ESGFFetcher()

    # Fetch all requests - returns DataFrame with metadata + paths
    combined = fetcher.fetch_for_test_case(tc.requests)

    if combined.empty:
        raise DatasetResolutionError(
            f"No datasets found for {diag.provider.slug}/{diag.slug} test case '{tc.name}'"
        )

    # Group paths by source type and use adapters to build proper catalog
    data_catalog: dict[SourceDatasetType, pd.DataFrame] = {}

    for source_type, group_df in combined.groupby("source_type"):
        file_paths = [Path(p) for p in group_df["path"].unique().tolist()]

        if source_type == "CMIP6":
            data_catalog[SourceDatasetType.CMIP6] = _build_catalog(CMIP6DatasetAdapter(), file_paths)

        elif source_type == "CMIP7":
            data_catalog[SourceDatasetType.CMIP7] = _build_catalog(CMIP7DatasetAdapter(), file_paths)

        elif source_type == "obs4MIPs":
            data_catalog[SourceDatasetType.obs4MIPs] = _build_catalog(Obs4MIPsDatasetAdapter(), file_paths)

        elif source_type == "PMPClimatology":
            data_catalog[SourceDatasetType.PMPClimatology] = _build_catalog(
                PMPClimatologyDatasetAdapter(), file_paths
            )

    if not data_catalog:
        raise DatasetResolutionError(
            f"No datasets found for {diag.provider.slug}/{diag.slug} test case '{tc.name}'"
        )

    # Solve for datasets
    datasets = _solve_test_case(diag, data_catalog)

    # Write catalog YAML to package-local test case directory
    catalog_written = False
    paths = TestCasePaths.from_diagnostic(diag, tc.name)
    if paths:
        paths.create()
        catalog_written = save_datasets_to_yaml(datasets, paths.catalog, force=force)

    return datasets, catalog_written


@app.command(name="fetch")
def fetch_test_data(  # noqa: PLR0912, PLR0915
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


def _print_regression_summary(  # pragma: no cover
    console: Console,
    regression_dir: Path,
    size_threshold_mb: float = 1.0,
) -> None:
    """
    Print a summary of the regression directory with file sizes and git status.

    Parameters
    ----------
    console
        Rich console for output
    regression_dir
        Path to the regression data directory
    size_threshold_mb
        Files larger than this (in MB) will be flagged
    """
    repo = get_repo_for_path(regression_dir)
    repo_root = Path(repo.working_dir) if repo else regression_dir

    threshold_bytes = int(size_threshold_mb * 1024 * 1024)
    file_info = collect_regression_file_info(regression_dir, repo, threshold_bytes)

    if not file_info:
        console.print("[yellow]No files in regression directory[/yellow]")
        return

    total_size = sum(f["size"] for f in file_info)
    large_files = sum(1 for f in file_info if f["is_large"])

    table = Table(title=f"Regression Data: {regression_dir.relative_to(repo_root)}")
    table.add_column("File", style="cyan", no_wrap=False)
    table.add_column("Size", justify="right")
    table.add_column("Git Status", justify="center")

    for info in file_info:
        # Format size with color
        size_str = format_size(info["size"])
        if info["is_large"]:
            size_str = f"[bold red]{size_str}[/bold red]"

        # Get git status with color
        git_status = info["git_status"]
        status_colors = {
            "new": "[green]new[/green]",
            "staged": "[green]staged[/green]",
            "modified": "[yellow]modified[/yellow]",
            "tracked": "[dim]tracked[/dim]",
            "untracked": "[red]untracked[/red]",
            "deleted": "[red]deleted[/red]",
            "unknown": "[dim]?[/dim]",
        }
        git_status_str = status_colors.get(git_status, f"[dim]{git_status}[/dim]")

        table.add_row(info["rel_path"], size_str, git_status_str)

    console.print(table)

    # Summary line
    summary = f"\n[bold]Total:[/bold] {len(file_info)} files, {format_size(total_size)}"
    if large_files > 0:
        summary += f" ([red]{large_files} files > {size_threshold_mb} MB[/red])"
    console.print(summary)


def _write_test_case_manifest(
    paths: TestCasePaths,
    *,
    test_case_version: int,
    committed: dict[str, str],
    native: dict[str, NativeEntry],
    schema: int | None = None,
) -> Manifest:
    """
    Construct and write a test case ``manifest.json``, recording the input catalog hash.

    Shared by ``run`` (which preserves the existing version and native block) and
    ``mint`` (which authors the native block and may bump the version); the two
    callers differ only in the ``test_case_version`` and ``native`` they supply.
    The ``catalog_hash`` is always (re)derived from the current ``catalog.yaml`` so
    the manifest stays coupled to the inputs that produced the committed bundle.
    """
    from climate_ref_core.regression.manifest import SCHEMA_VERSION, Manifest
    from climate_ref_core.testing import get_catalog_hash

    manifest = Manifest(
        schema=SCHEMA_VERSION if schema is None else schema,
        test_case_version=test_case_version,
        committed=dict(committed),
        native=native,
        catalog_hash=get_catalog_hash(paths.catalog),
    )
    manifest.dump(paths.manifest)
    return manifest


def _run_single_test_case(  # noqa: PLR0911, PLR0912, PLR0915
    config: Config,
    console: Console,
    diag: Diagnostic,
    tc: TestCase,
    *,
    execution_dir: Path | None,
    force_regen: bool,
    fetch: bool,
    size_threshold: float,
    clean: bool,
    label: str,
) -> bool:
    """
    Run a single test case for a diagnostic, writing its native into an output slot.

    Always (re)populates ``output/<label>/`` with the executed native set and its rebuilt
    committed bundle. The tracked ``regression/`` baseline is only updated ("promoted") when
    ``--force-regen`` is given or no baseline exists yet, so a labelled run never silently
    clobbers a committed baseline.

    Returns True if successful, False otherwise.
    """
    from climate_ref_core.regression.manifest import Manifest
    from climate_ref_core.testing import TestCasePaths, load_datasets_from_yaml

    provider_slug = diag.provider.slug
    diagnostic_slug = diag.slug
    test_case_name = tc.name
    case_id = f"{provider_slug}/{diagnostic_slug}/{test_case_name}"

    # Resolve datasets: either fetch from ESGF or load from the pre-built catalog.
    if fetch:
        logger.info(f"Fetching test data for {case_id}")
        try:
            datasets, _ = _fetch_and_build_catalog(diag, tc)
        except (DatasetResolutionError, InvalidDiagnosticException) as e:
            logger.exception(f"Failed to fetch data for {case_id}: {e}")
            return False

    paths = TestCasePaths.from_diagnostic(diag, test_case_name)
    if paths is None:
        logger.error(f"Could not determine test data directory for {provider_slug}/{diagnostic_slug}")
        return False

    if not fetch:
        if not paths.catalog.exists():
            logger.error(f"No catalog file found for {case_id}")
            logger.error("Run 'ref test-cases fetch' first or use --fetch flag")
            return False
        logger.info(f"Loading catalog from {paths.catalog}")
        datasets = load_datasets_from_yaml(paths.catalog)

    paths.create()
    slot = prepare_slot(paths, label)
    logger.info(f"Running test case {test_case_name!r} for {provider_slug}/{diagnostic_slug}")
    try:
        source = stage_execute(
            config=config,
            diag=diag,
            tc=tc,
            datasets=datasets,
            slot=slot,
            execution_dir=execution_dir,
            clean=clean,
        )
    except NoTestDataSpecError:
        logger.error(f"Diagnostic {provider_slug}/{diagnostic_slug} has no test_data_spec")
        return False
    except TestCaseNotFoundError:
        logger.error(f"Test case {test_case_name!r} not found for {provider_slug}/{diagnostic_slug}")
        if diag.test_data_spec:
            logger.error(f"Available test cases: {diag.test_data_spec.case_names}")
        return False
    except DatasetResolutionError as e:
        logger.error(str(e))
        logger.error("Have you run 'ref test-cases fetch' first?")
        return False
    except StageError:
        logger.error(f"Execution failed: {case_id}")
        return False
    except Exception as e:
        logger.error(f"Diagnostic execution failed for {case_id}: {e!s}")
        return False

    result = source.result
    logger.info(f"Execution completed: {case_id}")
    if result.metric_bundle_filename:
        logger.info(f"Metric bundle: {result.to_output_path(result.metric_bundle_filename)}")
    if result.output_bundle_filename:
        logger.info(f"Output bundle: {result.to_output_path(result.output_bundle_filename)}")

    # Rebuild the slot's committed bundle, then decide whether to promote it to the
    # tracked baseline. The native block is mint-owned, so a run preserves the previous
    # one (or seeds an empty set) and never authors native here.
    committed = stage_build(slot=slot, source=source, paths=paths)
    previous = Manifest.load(paths.manifest) if paths.manifest.exists() else None
    version = previous.test_case_version if previous else 1

    if force_regen or not paths.regression.exists():
        promote_to_baseline(slot, paths)
        native = snapshot_native(slot)
        if previous is not None:
            _write_test_case_manifest(
                paths,
                test_case_version=previous.test_case_version,
                committed=committed,
                native=previous.native,
                schema=previous.schema,
            )
            if native_is_stale(native, previous.native):
                logger.warning(
                    f"{case_id}: committed bundle regenerated but the native baseline differs; "
                    "re-mint with `ref test-cases mint` (or `mint --from-replay`)"
                )
        else:
            _write_test_case_manifest(paths, test_case_version=1, committed=committed, native={})
        logger.info(f"Updated regression baseline: {paths.regression}")
        _print_regression_summary(console, paths.regression, size_threshold)
    else:
        logger.info(
            f"Wrote output slot {slot} (committed baseline unchanged; use --force-regen to update it)"
        )

    write_source_stamp(slot, label=label, verb="run", source="execute", test_case_version=version)
    return True


@app.command(name="run")
def run_test_case(  # noqa: PLR0912, PLR0915
    ctx: typer.Context,
    provider: Annotated[
        str,
        typer.Option(help="Provider slug (required, e.g., 'example', 'ilamb')"),
    ],
    diagnostic: Annotated[
        str | None,
        typer.Option(help="Specific diagnostic slug to run (e.g., 'global-mean-timeseries')"),
    ] = None,
    test_case: Annotated[
        str | None,
        typer.Option(help="Specific test case name to run (e.g., 'default')"),
    ] = None,
    output_directory: Annotated[
        Path | None,
        typer.Option(help="Output directory for execution results"),
    ] = None,
    force_regen: Annotated[
        bool,
        typer.Option(help="Force regeneration of regression baselines"),
    ] = False,
    fetch: Annotated[
        bool,
        typer.Option(help="Fetch test data from ESGF before running"),
    ] = False,
    size_threshold: Annotated[
        float,
        typer.Option(help="Flag files larger than this size in MB (default: 1.0)"),
    ] = 1.0,
    dry_run: Annotated[
        bool,
        typer.Option(help="Show what would be run without executing"),
    ] = False,
    only_missing: Annotated[
        bool,
        typer.Option(help="Only run test cases without existing regression data"),
    ] = False,
    if_changed: Annotated[
        bool,
        typer.Option(help="Only run if catalog has changed since regression data was generated"),
    ] = False,
    clean: Annotated[
        bool,
        typer.Option(help="Delete existing output directory before running"),
    ] = False,
    label: Annotated[
        str,
        typer.Option(help="Output slot name under output/ (default: latest)"),
    ] = "latest",
) -> None:
    """
    Run test cases for diagnostics.

    Executes diagnostics using pre-defined datasets from the test_data_spec
    and optionally compares against regression baselines.

    Use --provider to select which provider's diagnostics to run (required).
    Use --diagnostic and --test-case to further narrow the scope.

    Examples
    --------
        ref test-cases run --provider ilamb              # Run all ILAMB test cases
        ref test-cases run --provider example --diagnostic global-mean-timeseries
        ref test-cases run --provider ilamb --test-case default --fetch
        ref test-cases run --provider pmp --only-missing # Skip test cases with regression data
        ref test-cases run --provider pmp --if-changed   # Only run if catalog changed
    """
    from climate_ref.provider_registry import ProviderRegistry
    from climate_ref_core.testing import (
        TestCasePaths,
        catalog_changed_since_regression,
    )

    config: Config = ctx.obj.config
    db = ctx.obj.database
    console: Console = ctx.obj.console

    # Build provider registry
    registry = ProviderRegistry.build_from_config(config, db)

    # Find the provider
    provider_instance = None
    for p in registry.providers:
        if p.slug == provider:
            provider_instance = p
            break

    if provider_instance is None:
        logger.error(f"Provider '{provider}' not found")
        available = [p.slug for p in registry.providers]
        logger.error(f"Available providers: {available}")
        raise typer.Exit(code=1)

    # Collect test cases to run
    test_cases_to_run: list[tuple[Diagnostic, TestCase]] = []
    skipped_cases: list[tuple[Diagnostic, TestCase]] = []

    for diag in provider_instance.diagnostics():
        if diagnostic and diag.slug != diagnostic:
            continue
        if diag.test_data_spec is None:
            continue

        for tc in diag.test_data_spec.test_cases:
            if test_case and tc.name != test_case:
                continue
            # Skip if regression exists when using --only-missing
            paths = TestCasePaths.from_diagnostic(diag, tc.name)
            if only_missing:
                if paths and paths.regression.exists():
                    skipped_cases.append((diag, tc))
                    continue
            # Skip if catalog hasn't changed when using --if-changed
            if if_changed:
                if paths and not catalog_changed_since_regression(paths):
                    skipped_cases.append((diag, tc))
                    continue
            test_cases_to_run.append((diag, tc))

    if not test_cases_to_run:
        logger.warning(f"No test cases found for provider '{provider}'")
        if diagnostic:
            logger.warning(f"  with diagnostic filter: {diagnostic}")
        if test_case:
            logger.warning(f"  with test case filter: {test_case}")
        if only_missing and skipped_cases:
            logger.info(f"  ({len(skipped_cases)} test case(s) skipped due to --only-missing)")
        raise typer.Exit(code=0)

    logger.info(f"Found {len(test_cases_to_run)} test case(s) to run")
    if skipped_cases:
        logger.info(f"Skipping {len(skipped_cases)} test case(s) with existing regression data")

    if dry_run:  # pragma: no cover
        table = Table(title="Test Cases to Run")
        table.add_column("Provider", style="cyan")
        table.add_column("Diagnostic", style="green")
        table.add_column("Test Case", style="yellow")
        table.add_column("Description")
        table.add_column("Status", justify="center")

        for diag, tc in test_cases_to_run:
            table.add_row(provider, diag.slug, tc.name, tc.description, "[green]will run[/green]")

        for diag, tc in skipped_cases:
            table.add_row(provider, diag.slug, tc.name, tc.description, "[dim]skip (regression exists)[/dim]")

        console.print(table)
        return

    # Run each test case
    successes = 0
    failures = 0
    failed_cases: list[str] = []

    for diag, tc in test_cases_to_run:
        success = _run_single_test_case(
            config=config,
            console=console,
            diag=diag,
            tc=tc,
            execution_dir=output_directory,
            force_regen=force_regen,
            fetch=fetch,
            size_threshold=size_threshold,
            clean=clean,
            label=label,
        )
        if success:
            successes += 1
        else:
            failures += 1
            failed_cases.append(f"{provider}/{diag.slug}/{tc.name}")

    # Print summary
    console.print()
    if failures == 0:
        console.print(f"[green]All {successes} test case(s) passed[/green]")
    else:
        console.print(f"[yellow]Results: {successes} passed, {failures} failed[/yellow]")
        console.print("[red]Failed test cases:[/red]")
        for case in failed_cases:
            console.print(f"  - {case}")
        raise typer.Exit(code=1)


def _iter_test_cases(
    registry: ProviderRegistry,
    *,
    provider: str | None = None,
    diagnostic: str | None = None,
    test_case: str | None = None,
) -> Iterator[tuple[Diagnostic, TestCase]]:
    """
    Yield ``(diagnostic, test_case)`` pairs from the registry, applying filters.

    Parameters
    ----------
    registry
        The provider registry to enumerate.
    provider
        Optional provider slug filter.
    diagnostic
        Optional diagnostic slug filter.
    test_case
        Optional test case name filter.

    Yields
    ------
    :
        Matching ``(diagnostic, test_case)`` pairs.
    """
    for provider_instance in registry.providers:
        if provider and provider_instance.slug != provider:
            continue
        for diag in provider_instance.diagnostics():
            if diagnostic and diag.slug != diagnostic:
                continue
            if diag.test_data_spec is None:
                continue
            for tc in diag.test_data_spec.test_cases:
                if test_case and tc.name != test_case:
                    continue
                yield diag, tc


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
        try:
            source = stage_materialise(
                diag=diag, tc=tc, paths=paths, manifest=manifest, store=store, slot=slot
            )
        except Exception as exc:
            logger.error(f"{case_id}: failed to materialise/rebuild native: {exc}")
            failures.append(case_id)
            continue

        stage_build(slot=slot, source=source, paths=paths)
        cmp_failures, compared = stage_compare(slot=slot, paths=paths, slug=diag.slug)
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
def mint_native(  # noqa: PLR0912, PLR0915
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

    registry = ProviderRegistry.build_from_config(config, db)
    cases = list(_iter_test_cases(registry, provider=provider, diagnostic=diagnostic, test_case=test_case))
    if not cases:
        logger.warning(f"No test cases found for provider {provider!r}")
        raise typer.Exit(code=0)

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

        # Populate the slot's native set: either re-execute the diagnostic, or (with
        # --from-replay) materialise the previously minted native from the store. The
        # writable store's fetch/has back the materialise, so no separate read store is needed.
        try:
            if from_replay and previous is not None:  # previous is non-None by the guard above
                source = stage_materialise(
                    diag=diag, tc=tc, paths=paths, manifest=previous, store=store, slot=slot
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

        committed = stage_build(slot=slot, source=source, paths=paths)
        native = snapshot_native(slot)
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
        _write_test_case_manifest(paths, test_case_version=version, committed=committed, native=native)
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
def build_test_case(  # noqa: PLR0912, PLR0915
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

        try:
            source = stage_rebuild_from_slot(diag=diag, tc=tc, paths=paths, slot=slot)
        except Exception as exc:
            logger.error(f"{case_id}: failed to rebuild bundle from slot: {exc}")
            failures.append(case_id)
            continue

        committed = stage_build(slot=slot, source=source, paths=paths)
        previous = Manifest.load(paths.manifest) if paths.manifest.exists() else None
        version = previous.test_case_version if previous else 1

        if force_regen or not paths.regression.exists():
            promote_to_baseline(slot, paths)
            native = snapshot_native(slot)
            if previous is not None:
                _write_test_case_manifest(
                    paths,
                    test_case_version=previous.test_case_version,
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
                _write_test_case_manifest(paths, test_case_version=1, committed=committed, native={})
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


def _provider_source_root(diag: Diagnostic, repo_root: Path) -> str | None:
    """
    Return the diagnostic's provider package source directory, relative to the repo root.

    Used to decide whether a changed file touches the diagnostic's extraction code.
    The returned path is POSIX-style so it can be compared against
    ``git diff --name-only`` output.

    Parameters
    ----------
    diag
        The diagnostic whose provider source directory is wanted.
    repo_root
        The repository working-tree root.

    Returns
    -------
    :
        The provider package source directory relative to ``repo_root``, or ``None``
        if it cannot be located or lies outside the repository.
    """
    import importlib.util

    top_package = type(diag).__module__.split(".")[0]
    try:
        spec = importlib.util.find_spec(top_package)
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.submodule_search_locations:
        return None
    package_dir = Path(next(iter(spec.submodule_search_locations))).resolve()
    try:
        return package_dir.relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return None


def _core_extraction_roots(repo_root: Path) -> list[str]:
    """
    Return the core paths whose change affects replay/execute for every test case.

    ``build_execution_result`` (the function replay/execute re-run) depends on more
    than the regression package: it builds and reads CMEC bundles via
    :mod:`climate_ref_core.pycmec`, persists curated outputs via
    :mod:`climate_ref_core.output_files`, and is dispatched through
    :mod:`climate_ref_core.diagnostics`. A change under any of these can alter the
    regenerated bundle, so all of them count as an extraction change.

    Detection is deliberately coarse and errs toward REPLAY (cheap, credential-free):
    a false positive only triggers an unnecessary replay, never a missed one.
    Roots are derived from the installed package location (rather than hardcoded
    paths) so they survive a package move; any root outside the repository
    (e.g. an installed wheel) is dropped.

    Parameters
    ----------
    repo_root
        The repository working-tree root.

    Returns
    -------
    :
        Repo-relative POSIX paths for the core extraction surfaces that lie inside the repo.
    """
    import climate_ref_core

    core_dir = Path(climate_ref_core.__file__).resolve().parent
    repo_root_resolved = repo_root.resolve()
    candidates = [
        core_dir / "regression",
        core_dir / "pycmec",
        core_dir / "output_files.py",
        core_dir / "diagnostics.py",
    ]
    roots: list[str] = []
    for candidate in candidates:
        try:
            roots.append(candidate.relative_to(repo_root_resolved).as_posix())
        except ValueError:
            continue
    return roots


@app.command(name="ci-gate")
def ci_gate(  # noqa: PLR0912, PLR0915
    ctx: typer.Context,
    base: Annotated[
        str,
        typer.Option(help="Git ref to compare against (the PR base branch)"),
    ] = "origin/main",
    provider: Annotated[
        str | None,
        typer.Option(help="Limit the gate to a single provider slug"),
    ] = None,
    diagnostic: Annotated[
        str | None,
        typer.Option(help="Limit the gate to a single diagnostic slug"),
    ] = None,
    test_case: Annotated[
        str | None,
        typer.Option(help="Limit the gate to a single test case name"),
    ] = None,
    output_json: Annotated[
        bool,
        typer.Option("--json", help="Emit the per-case decisions as JSON on stdout"),
    ] = False,
) -> None:
    """
    Decide how CI should verify each test case's regression baseline.

    Compares each committed ``manifest.json`` to its counterpart on the base branch
    and reports the action CI should take per case: ``replay`` (cheap, against the
    cached native baseline), ``execute`` (full re-run, when ``test_case_version`` was
    bumped), ``skip`` (nothing relevant changed), or ``fail`` (an unauthorised
    baseline change). Exits non-zero if any case is gated ``fail``.

    The ``--json`` output is intended for CI to dispatch ``replay``/``run`` jobs.

    Examples
    --------
        ref test-cases ci-gate                       # Gate all cases against origin/main
        ref test-cases ci-gate --base origin/develop
        ref test-cases ci-gate --provider example --json
    """
    import json as _json

    from git import GitCommandError

    from climate_ref.provider_registry import ProviderRegistry
    from climate_ref_core.regression.gate import Action, decide_coupling, paths_under
    from climate_ref_core.regression.manifest import Manifest, compute_committed_digests
    from climate_ref_core.testing import TestCasePaths, get_catalog_hash

    config: Config = ctx.obj.config
    db = ctx.obj.database
    console: Console = ctx.obj.console

    repo = get_repo_for_path(Path.cwd())
    if repo is None or repo.working_dir is None:
        logger.error("ci-gate must be run inside a git repository")
        raise typer.Exit(code=1)
    repo_root = Path(repo.working_dir)

    # Resolve the set of files changed on this branch relative to the base ref.
    # `base...HEAD` diffs against the merge-base, so unrelated base-branch churn
    # is excluded.
    try:
        diff_output = repo.git.diff("--name-only", f"{base}...HEAD")
    except GitCommandError as exc:
        logger.error(f"Could not diff against base ref {base!r}: {exc}")
        raise typer.Exit(code=1) from exc
    changed_files = [line.strip() for line in diff_output.splitlines() if line.strip()]

    # The core machinery behind build_execution_result affects every replay/execute,
    # so a change there counts as an extraction change for all cases. Extraction-change
    # detection is deliberately coarse: any change under a diagnostic's provider package
    # (see `_provider_source_root`) or under the core extraction surfaces counts for
    # every case in that provider. This errs toward REPLAY (cheap, credential-free),
    # never away from it.
    core_changed = paths_under(changed_files, _core_extraction_roots(repo_root))

    # Hoisted once: repo_root.resolve() is filesystem-touching, and a provider's source
    # root is identical for every case in that provider, so memoise it per provider slug
    # rather than recomputing find_spec on each case.
    repo_root_resolved = repo_root.resolve()
    source_root_cache: dict[str, str | None] = {}

    registry = ProviderRegistry.build_from_config(config, db)

    decisions: list[dict[str, str]] = []
    has_failure = False

    def record(case: str, action: Action, reason: str) -> None:
        nonlocal has_failure
        if action is Action.FAIL:
            has_failure = True
        decisions.append({"case": case, "action": action.value, "reason": reason})

    for diag, tc in _iter_test_cases(registry, provider=provider, diagnostic=diagnostic, test_case=test_case):
        case_id = f"{diag.provider.slug}/{diag.slug}/{tc.name}"
        paths = TestCasePaths.from_diagnostic(diag, tc.name)

        # A corrupt manifest authored in this change is a hard failure for that case,
        # not a crash for the whole gate.
        manifest: Manifest | None = None
        if paths is not None and paths.manifest.exists():
            try:
                manifest = Manifest.load(paths.manifest)
            except ValueError as exc:
                logger.error(f"{case_id}: invalid manifest.json: {exc}")
                record(case_id, Action.FAIL, f"invalid manifest.json: {exc}")
                continue

        base_manifest: Manifest | None = None
        if paths is not None:
            try:
                rel_manifest = paths.manifest.resolve().relative_to(repo_root_resolved).as_posix()
            except ValueError:
                rel_manifest = None
            if rel_manifest is not None:
                try:
                    base_text = repo.git.show(f"{base}:{rel_manifest}")
                except GitCommandError:
                    base_manifest = None
                else:
                    # A corrupt manifest on the base branch can't be compared against;
                    # fall back to seeding (REPLAY) rather than aborting the gate.
                    try:
                        base_manifest = Manifest.loads(base_text, source=f"{base}:{rel_manifest}")
                    except ValueError as exc:
                        logger.warning(
                            f"{case_id}: base manifest at {base}:{rel_manifest} is invalid "
                            f"({exc}); treating as newly added"
                        )
                        base_manifest = None

        provider_slug = diag.provider.slug
        if provider_slug not in source_root_cache:
            source_root_cache[provider_slug] = _provider_source_root(diag, repo_root)
        source_root = source_root_cache[provider_slug]
        extraction_roots = [r for r in (source_root,) if r]
        extraction_changed = core_changed or paths_under(changed_files, extraction_roots)

        # Verify the committed bundle on disk still matches the manifest digests.
        # A drift (edited/added/removed committed file without regenerating the
        # manifest) must fail closed rather than slip through as SKIP.
        committed_integrity_ok = True
        # Verify the input catalog still matches the manifest's recorded hash. A catalog
        # edit without regenerating the baseline leaves it silently stale; fail closed.
        # Legacy manifests without a catalog_hash have nothing to compare, so stay OK.
        catalog_integrity_ok = True
        if manifest is not None and paths is not None:
            committed_integrity_ok = compute_committed_digests(paths.regression) == manifest.committed
            if manifest.catalog_hash is not None:
                catalog_integrity_ok = get_catalog_hash(paths.catalog) == manifest.catalog_hash

        decision = decide_coupling(
            manifest,
            base_manifest,
            extraction_changed=extraction_changed,
            committed_integrity_ok=committed_integrity_ok,
            catalog_integrity_ok=catalog_integrity_ok,
        )
        record(case_id, decision.action, decision.reason)

    if output_json:
        console.print_json(_json.dumps(decisions))
    else:
        table = Table(title=f"CI coupling gate (base: {base})")
        table.add_column("Test case", style="cyan", no_wrap=True)
        table.add_column("Action")
        table.add_column("Reason")
        style_for = {
            Action.FAIL.value: "red",
            Action.EXECUTE.value: "yellow",
            Action.REPLAY.value: "green",
            Action.SKIP.value: "dim",
        }
        for entry in decisions:
            style = style_for[entry["action"]]
            table.add_row(entry["case"], f"[{style}]{entry['action']}[/{style}]", entry["reason"])
        console.print(table)

    if has_failure:
        raise typer.Exit(code=1)
