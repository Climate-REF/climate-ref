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
from git import InvalidGitRepositoryError, Repo
from loguru import logger
from rich.console import Console
from rich.table import Table

from climate_ref.config import Config
from climate_ref.datasets import (
    CMIP6DatasetAdapter,
    DatasetAdapter,
    Obs4MIPsDatasetAdapter,
    PMPClimatologyDatasetAdapter,
)
from climate_ref.provider_registry import ProviderRegistry
from climate_ref.solver import solve_executions
from climate_ref.testing import TestCaseRunner
from climate_ref_core.datasets import ExecutionDatasetCollection, SourceDatasetType
from climate_ref_core.diagnostics import Diagnostic
from climate_ref_core.esgf import ESGFFetcher
from climate_ref_core.exceptions import (
    DatasetResolutionError,
    NoTestDataSpecError,
    TestCaseNotFoundError,
)
from climate_ref_core.testing import (
    TestCase,
    TestCasePaths,
    load_datasets_from_yaml,
    save_datasets_to_yaml,
)

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
    executions = list(solve_executions(data_catalog, diagnostic, diagnostic.provider))

    if not executions:
        raise ValueError(f"No valid executions found for diagnostic {diagnostic.slug}")

    return executions[0].datasets


def _fetch_and_build_catalog(
    diag: Diagnostic,
    tc: TestCase,
) -> ExecutionDatasetCollection:
    """
    Fetch test data and build catalog.

    This function:
    1. Fetches ESGF data using ESGFFetcher (files stored in intake-esgf cache)
    2. Uses CMIP6DatasetAdapter to create a data catalog
    3. Solves for datasets using the diagnostic's data requirements
    4. Writes catalog YAML to .catalogs/{provider}/{diagnostic}/{test_case}.yaml
    5. Returns the solved datasets

    Parameters
    ----------
    diag
        The diagnostic to fetch data for
    tc
        The test case to fetch data for

    Returns
    -------
    ExecutionDatasetCollection
        The solved datasets for this test case
    """
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
    paths = TestCasePaths.from_diagnostic(diag, tc.name)
    if paths:
        paths.create()
        save_datasets_to_yaml(datasets, paths.catalog)

    return datasets


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
    dry_run: Annotated[
        bool,
        typer.Option(help="Show what would be fetched without downloading"),
    ] = False,
    only_missing: Annotated[
        bool,
        typer.Option(help="Only fetch data for test cases without existing catalogs"),
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
    config = ctx.obj.config
    db = ctx.obj.database

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
    for diag in diagnostics_to_process:
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
                        _fetch_and_build_catalog(diag, tc)
                    except DatasetResolutionError as e:
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
    config = ctx.obj.config
    db = ctx.obj.database
    console = ctx.obj.console

    # Build provider registry to access diagnostics
    registry = ProviderRegistry.build_from_config(config, db)

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


def _format_size(size_bytes: int | float) -> str:
    """Format file size in human-readable form."""
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _get_git_status(file_path: Path, repo: Repo) -> str:
    """Get git status for a file using GitPython."""
    try:
        rel_path = str(file_path.relative_to(repo.working_dir))

        # Check if untracked
        if rel_path in repo.untracked_files:
            return "new"

        # Check staged changes (index vs HEAD)
        staged_files = {item.a_path for item in repo.index.diff("HEAD")}
        if rel_path in staged_files:
            return "staged"

        # Check unstaged changes (working tree vs index)
        unstaged_files = {item.a_path for item in repo.index.diff(None)}
        if rel_path in unstaged_files:
            return "modified"

        # Check if file is tracked
        try:
            repo.git.ls_files("--error-unmatch", rel_path)
            return "tracked"
        except Exception:
            return "untracked"
    except Exception:
        return "unknown"


def _print_regression_summary(
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
    # Try to get git repo
    try:
        repo = Repo(regression_dir, search_parent_directories=True)
        repo_root = Path(repo.working_dir)
    except InvalidGitRepositoryError:
        repo = None
        repo_root = regression_dir

    # Collect file info
    files = sorted(regression_dir.rglob("*"))
    files = [f for f in files if f.is_file()]

    if not files:
        console.print("[yellow]No files in regression directory[/yellow]")
        return

    threshold_bytes = int(size_threshold_mb * 1024 * 1024)
    total_size = 0
    large_files = 0

    table = Table(title=f"Regression Data: {regression_dir.relative_to(repo_root)}")
    table.add_column("File", style="cyan", no_wrap=False)
    table.add_column("Size", justify="right")
    table.add_column("Git Status", justify="center")

    for file_path in files:
        size = file_path.stat().st_size
        total_size += size
        is_large = size > threshold_bytes

        # Format relative path
        rel_path = str(file_path.relative_to(regression_dir))

        # Format size with color
        size_str = _format_size(size)
        if is_large:
            size_str = f"[bold red]{size_str}[/bold red]"
            large_files += 1

        # Get git status with color
        git_status = _get_git_status(file_path, repo) if repo else "unknown"
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

        table.add_row(rel_path, size_str, git_status_str)

    console.print(table)

    # Summary line
    summary = f"\n[bold]Total:[/bold] {len(files)} files, {_format_size(total_size)}"
    if large_files > 0:
        summary += f" ([red]{large_files} files > {size_threshold_mb} MB[/red])"
    console.print(summary)


def _run_single_test_case(  # noqa: PLR0911, PLR0912, PLR0915
    config: Config,
    console: Console,
    diag: Diagnostic,
    tc: TestCase,
    output_directory: Path | None,
    force_regen: bool,
    fetch: bool,
    size_threshold: float,
) -> bool:
    """
    Run a single test case for a diagnostic.

    Returns True if successful, False otherwise.
    """
    provider_slug = diag.provider.slug
    diagnostic_slug = diag.slug
    test_case_name = tc.name

    # Resolve datasets: either fetch from ESGF or load from pre-built catalog
    if fetch:
        logger.info(f"Fetching test data for {provider_slug}/{diagnostic_slug}/{test_case_name}")
        try:
            datasets = _fetch_and_build_catalog(diag, tc)
        except DatasetResolutionError as e:
            logger.error(f"Failed to fetch data for {provider_slug}/{diagnostic_slug}/{test_case_name}: {e}")
            return False
    else:
        paths = TestCasePaths.from_diagnostic(diag, test_case_name)
        if paths is None:
            logger.error(f"Could not determine test data directory for {provider_slug}/{diagnostic_slug}")
            return False

        if not paths.catalog.exists():
            logger.error(f"No catalog file found for {provider_slug}/{diagnostic_slug}/{test_case_name}")
            logger.error("Run 'ref test-cases fetch' first or use --fetch flag")
            return False

        logger.info(f"Loading catalog from {paths.catalog}")
        datasets = load_datasets_from_yaml(paths.catalog)

    # Create runner and execute
    runner = TestCaseRunner(config=config, datasets=datasets)

    logger.info(f"Running test case {test_case_name!r} for {provider_slug}/{diagnostic_slug}")

    try:
        result = runner.run(diag, test_case_name, output_directory)
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
    except Exception as e:
        case_id = f"{provider_slug}/{diagnostic_slug}/{test_case_name}"
        logger.error(f"Diagnostic execution failed for {case_id}: {e!s}")
        return False

    if not result.successful:
        logger.error(f"Execution failed: {provider_slug}/{diagnostic_slug}/{test_case_name}")
        return False

    logger.info(f"Execution completed: {provider_slug}/{diagnostic_slug}/{test_case_name}")
    if result.metric_bundle_filename:
        logger.info(f"Metric bundle: {result.to_output_path(result.metric_bundle_filename)}")
    if result.output_bundle_filename:
        logger.info(f"Output bundle: {result.to_output_path(result.output_bundle_filename)}")

    # Handle regression baseline comparison/regeneration
    paths = TestCasePaths.from_diagnostic(diag, test_case_name)

    if paths is None:
        logger.warning("Could not determine test case directory for provider package")
        return True

    if force_regen:
        paths.create()

    if force_regen or not paths.regression.exists():
        # Save full output directory as regression data
        if paths.regression.exists():
            shutil.rmtree(paths.regression)
        paths.regression.mkdir(parents=True, exist_ok=True)
        shutil.copytree(result.definition.output_directory, paths.regression, dirs_exist_ok=True)

        # Replace absolute paths with placeholders for portability
        # We don't touch binary files, only text-based ones
        # TODO: Symlink regression datasets instead of any paths on users' systems
        for glob_pattern in ("*.json", "*.txt", "*.yaml", "*.yml"):
            for file in paths.regression.rglob(glob_pattern):
                content = file.read_text()
                content = content.replace(str(result.definition.output_directory), "<OUTPUT_DIR>")
                content = content.replace(str(paths.test_data_dir), "<TEST_DATA_DIR>")
                file.write_text(content)

        logger.info(f"Updated regression data: {paths.regression}")
        _print_regression_summary(console, paths.regression, size_threshold)
    elif paths.regression.exists():
        logger.info(f"Regression data exists at: {paths.regression}")
        logger.info("Use --force-regen to update the baseline")

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
    """
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
            if only_missing:
                paths = TestCasePaths.from_diagnostic(diag, tc.name)
                if paths and paths.regression.exists():
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

    if dry_run:
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
            output_directory=output_directory,
            force_regen=force_regen,
            fetch=fetch,
            size_threshold=size_threshold,
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
