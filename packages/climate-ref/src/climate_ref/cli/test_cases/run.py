"""
``ref test-cases run``.

Executes diagnostics for their declared test cases, writes the native into an
output slot, rebuilds the committed bundle, and (when asked) promotes it to the
tracked regression baseline.

``--from-slot`` skips the execution and reuses the native already in the slot,
so the bundle can be regenerated after an extraction-code change.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from loguru import logger
from rich.table import Table

from climate_ref.cli._git_utils import collect_regression_file_info, get_repo_for_path
from climate_ref.cli._utils import format_size
from climate_ref.cli.test_cases._app import app
from climate_ref.cli.test_cases._catalog import _fetch_and_build_catalog
from climate_ref.cli.test_cases._common import VerbDriver, VerbSummary
from climate_ref.cli.test_cases._stages import (
    SourceOutputs,
    StageError,
    baseline_placeholders,
    prepare_slot,
    promote_and_author_manifest,
    slot_native_relpaths,
    stage_build,
    stage_execute,
    stage_rebuild_from_slot,
)
from climate_ref.config import Config
from climate_ref_core.exceptions import (
    DatasetResolutionError,
    InvalidDiagnosticException,
    NoTestDataSpecError,
    TestCaseNotFoundError,
)

if TYPE_CHECKING:
    from rich.console import Console

    from climate_ref_core.diagnostics import Diagnostic
    from climate_ref_core.output_files import PlaceholderMap
    from climate_ref_core.testing import TestCase, TestCasePaths


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


def _rebuild_from_slot(
    *,
    diag: Diagnostic,
    tc: TestCase,
    paths: TestCasePaths,
    label: str,
    placeholders: PlaceholderMap,
) -> tuple[Path, SourceOutputs] | None:
    """
    Rebuild a test case's execution result from the native already in its output slot.

    Returns the slot and the rebuilt source outputs, or ``None`` when the slot holds no
    native or the catalog is missing (both logged as errors).
    """
    case_id = f"{diag.provider.slug}/{diag.slug}/{tc.name}"
    slot = paths.output_slot(label)
    if not slot.exists() or not slot_native_relpaths(slot):
        logger.error(f"{case_id}: no native in output slot {label!r}; run/replay/mint it first")
        return None
    if not paths.catalog.exists():
        logger.error(f"No catalog file for {case_id}; run `ref test-cases fetch` first")
        return None

    try:
        source = stage_rebuild_from_slot(diag=diag, tc=tc, paths=paths, slot=slot, placeholders=placeholders)
    except Exception as exc:
        logger.error(f"{case_id}: failed to rebuild bundle from slot: {exc}")
        return None
    logger.info(f"Rebuilt {case_id} from output slot {slot} without executing")
    return slot, source


def _run_single_test_case(  # noqa: PLR0911, PLR0912, PLR0913, PLR0915
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
    from_slot: bool,
) -> bool:
    """
    Run a single test case for a diagnostic, writing its native into an output slot.

    Always (re)populates ``output/<label>/`` with the executed native set and its rebuilt
    committed bundle. The tracked ``regression/`` baseline is only updated ("promoted") when
    ``--force-regen`` is given or no baseline exists yet, so a labelled run never silently
    clobbers a committed baseline.

    With ``from_slot`` the diagnostic is not executed at all: the native already in
    ``output/<label>/`` is reused and only the committed bundle is regenerated.

    Returns True if successful, False otherwise.
    """
    from climate_ref_core.testing import TestCasePaths, load_datasets_from_yaml

    provider_slug = diag.provider.slug
    diagnostic_slug = diag.slug
    test_case_name = tc.name
    case_id = f"{provider_slug}/{diagnostic_slug}/{test_case_name}"

    # Resolve datasets: either fetch from ESGF or load from the pre-built catalog.
    if fetch and not from_slot:
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

    placeholders = baseline_placeholders(paths, config)

    if from_slot:
        source_or_none = _rebuild_from_slot(
            diag=diag, tc=tc, paths=paths, label=label, placeholders=placeholders
        )
        if source_or_none is None:
            return False
        slot, source = source_or_none
    else:
        if not fetch:
            if not paths.catalog.exists():
                logger.error(f"No catalog file found for {case_id}")
                logger.error("Run 'ref test-cases fetch' first or use --fetch flag")
                return False
            logger.info(f"Loading catalog from {paths.catalog}")
            datasets = load_datasets_from_yaml(paths.catalog, paths.catalog_paths)

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
    committed = stage_build(slot=slot, source=source, placeholders=placeholders)

    if force_regen or not paths.regression.exists():
        promote_and_author_manifest(
            paths=paths,
            diag=diag,
            slot=slot,
            source=source,
            placeholders=placeholders,
            committed=committed,
            stale_message=(
                f"{case_id}: committed bundle regenerated but the native baseline differs; "
                "re-mint with `ref test-cases mint` (or `mint --from-replay`)"
            ),
        )
        logger.info(f"Updated regression baseline: {paths.regression}")
        _print_regression_summary(console, paths.regression, size_threshold)
    else:
        logger.info(
            f"Wrote output slot {slot} (committed baseline unchanged; use --force-regen to update it)"
        )

    return True


@app.command(name="run")
def run_test_case(  # noqa: PLR0912, PLR0913, PLR0915
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
        typer.Option(
            help=(
                "Scratch directory for the diagnostic execution results. "
                "The regression workflow also writes the gitignored output/<label> slot."
            )
        ),
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
    from_slot: Annotated[
        bool,
        typer.Option(
            "--from-slot",
            help="Rebuild the committed bundle from the native already in the output slot",
        ),
    ] = False,
) -> None:
    """
    Run test cases for diagnostics.

    Executes diagnostics using pre-defined datasets from the test_data_spec
    and optionally compares against regression baselines.

    Use --provider to select which provider's diagnostics to run (required).
    Use --diagnostic and --test-case to further narrow the scope.

    With --from-slot the diagnostic is not executed. The native already materialised in
    ``output/<label>/`` by a previous run, replay or mint is reused, and only the committed
    bundle is regenerated. This is how a bundle is refreshed after an extraction-code change.
    The execution options (--fetch, --output-directory and --clean) do not apply to it.

    Examples
    --------
        ref test-cases run --provider ilamb              # Run all ILAMB test cases
        ref test-cases run --provider example --diagnostic global-mean-timeseries
        ref test-cases run --provider ilamb --test-case default --fetch
        ref test-cases run --provider pmp --only-missing # Skip test cases with regression data
        ref test-cases run --provider pmp --if-changed   # Only run if catalog changed
        ref test-cases run --provider example --from-slot --label before --force-regen
    """
    from climate_ref_core.testing import (
        TestCasePaths,
        catalog_changed_since_regression,
    )

    config: Config = ctx.obj.config
    console: Console = ctx.obj.console

    driver = VerbDriver(ctx, provider=provider, diagnostic=diagnostic, test_case=test_case)

    # Collect test cases to run
    test_cases_to_run: list[tuple[Diagnostic, TestCase]] = []
    skipped_cases: list[tuple[Diagnostic, TestCase]] = []

    for diag, tc in driver.cases:
        paths = TestCasePaths.from_diagnostic(diag, tc.name)
        # Skip if regression exists when using --only-missing
        if only_missing and paths and paths.regression.exists():
            skipped_cases.append((diag, tc))
            continue
        # Skip if catalog hasn't changed when using --if-changed
        if if_changed and paths and not catalog_changed_since_regression(paths):
            skipped_cases.append((diag, tc))
            continue
        test_cases_to_run.append((diag, tc))

    if not test_cases_to_run:
        if only_missing and skipped_cases:
            logger.info(
                f"All {len(skipped_cases)} matching test case(s) skipped because "
                "regression baselines already exist"
            )
        elif if_changed and skipped_cases:
            logger.info(
                f"All {len(skipped_cases)} matching test case(s) skipped because catalogs are unchanged"
            )
        else:
            logger.warning(f"No test cases found for provider '{provider}'")
            if diagnostic:
                logger.warning(f"  with diagnostic filter: {diagnostic}")
            if test_case:
                logger.warning(f"  with test case filter: {test_case}")
        raise typer.Exit(code=0)

    logger.info(f"Found {len(test_cases_to_run)} test case(s) to run")
    if skipped_cases:
        if only_missing:
            logger.info(f"Skipping {len(skipped_cases)} test case(s) with existing regression data")
        elif if_changed:
            logger.info(f"Skipping {len(skipped_cases)} test case(s) with unchanged catalogs")

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

    if output_directory is not None and not from_slot:
        logger.info(
            f"Using {output_directory} as the execution scratch directory; rebuilt native/bundle files "
            f"will also be written to each test case's gitignored output/{label} slot"
        )

    for diag, tc in test_cases_to_run:
        case_id = f"{provider}/{diag.slug}/{tc.name}"
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
            from_slot=from_slot,
        )
        if success:
            driver.ok()
        else:
            driver.fail(case_id)

    driver.finish(
        VerbSummary(
            mixed="Results: {successes} passed, {failures} failed",
            failed_header="Failed test cases:",
            success="All {successes} test case(s) passed",
        )
    )
