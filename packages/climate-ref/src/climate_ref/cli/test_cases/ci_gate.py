"""
``ref test-cases ci-gate``.

Decides, per test case, how CI should verify its regression baseline (replay,
execute, skip, or fail) by comparing the committed ``manifest.json`` against the
base branch and detecting extraction-code changes.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from loguru import logger
from rich.table import Table

from climate_ref.cli._git_utils import get_repo_for_path
from climate_ref.cli.test_cases._app import app
from climate_ref.cli.test_cases._common import _iter_test_cases, _validate_provider_in_registry
from climate_ref.config import Config

if TYPE_CHECKING:
    from rich.console import Console

    from climate_ref_core.diagnostics import Diagnostic


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
def ci_gate(  # noqa: PLR0912, PLR0913, PLR0915
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
    _validate_provider_in_registry(registry, provider)

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
