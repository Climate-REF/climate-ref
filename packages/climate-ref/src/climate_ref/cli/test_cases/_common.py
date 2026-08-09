"""
Helpers shared across several ``ref test-cases`` commands.

``VerbDriver`` owns the per-case loop machinery every verb repeats
(registry construction, selector validation, case enumeration, skip policy, tally and summary).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import typer
from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Iterator

    from rich.console import Console

    from climate_ref.provider_registry import ProviderRegistry
    from climate_ref_core.diagnostics import Diagnostic
    from climate_ref_core.regression.manifest import Manifest, NativeEntry
    from climate_ref_core.testing import TestCase, TestCasePaths


def _validate_provider_in_registry(registry: ProviderRegistry, provider: str | None) -> None:
    """
    Validate that ``provider`` (if given) is configured in the registry.

    Logs a helpful error listing the available providers and exits with code 1
    when the requested provider is not present. A falsy ``provider`` is a no-op.
    """
    if not provider:
        return
    available_providers = [p.slug for p in registry.providers]
    if provider not in available_providers:
        logger.error(f"Provider '{provider}' is not configured")
        if available_providers:
            logger.error(f"Available providers: {', '.join(sorted(available_providers))}")
        else:
            logger.error("No providers are configured. Check your configuration file.")
        logger.error("To add a provider, update your config file or set REF_DIAGNOSTIC_PROVIDERS")
        raise typer.Exit(code=1)


def _validate_requested_filters(
    registry: ProviderRegistry,
    *,
    provider: str | None = None,
    diagnostic: str | None = None,
    test_case: str | None = None,
) -> None:
    """
    Fail fast when an explicit diagnostic or test-case selector matches nothing.

    Empty selections caused by skip flags (for example ``--only-missing``) are not handled here.
    Callers should decide whether those are successful no-ops.
    This helper only guards likely typos in user-supplied selectors.
    """
    if not diagnostic and not test_case:
        return

    provider_instances = [p for p in registry.providers if provider is None or p.slug == provider]
    diagnostics = [
        diag for provider_instance in provider_instances for diag in provider_instance.diagnostics()
    ]

    if diagnostic:
        matching_diagnostics = [diag for diag in diagnostics if diag.slug == diagnostic]
        if not matching_diagnostics:
            scope = f" for provider '{provider}'" if provider else ""
            logger.error(f"Diagnostic '{diagnostic}' was not found{scope}")
            available = sorted({diag.slug for diag in diagnostics})
            if available:
                logger.error(f"Available diagnostics: {', '.join(available)}")
            raise typer.Exit(code=1)
        diagnostics = matching_diagnostics

    if test_case:
        available_cases = sorted(
            {
                tc.name
                for diag in diagnostics
                if diag.test_data_spec is not None
                for tc in diag.test_data_spec.test_cases
            }
        )
        if test_case not in available_cases:
            scope_parts = []
            if provider:
                scope_parts.append(f"provider '{provider}'")
            if diagnostic:
                scope_parts.append(f"diagnostic '{diagnostic}'")
            scope = f" for {' and '.join(scope_parts)}" if scope_parts else ""
            logger.error(f"Test case '{test_case}' was not found{scope}")
            if available_cases:
                logger.error(f"Available test cases: {', '.join(available_cases)}")
            else:
                logger.error("No test cases are defined for the selected diagnostics")
            raise typer.Exit(code=1)


def _write_test_case_manifest(  # noqa: PLR0913
    paths: TestCasePaths,
    *,
    test_case_version: int,
    diagnostic_version: int,
    committed: dict[str, str],
    native: dict[str, NativeEntry],
    schema: int | None = None,
) -> Manifest:
    """
    Construct and write a test case ``manifest.json``, recording the input catalog hash.

    Shared by ``run`` (which preserves the existing version and native block) and
    ``mint`` (which authors the native block and may bump the version); the two
    callers differ only in the ``test_case_version`` and ``native`` they supply.
    Only ``mint`` advances ``diagnostic_version`` to the diagnostic's current
    ``Diagnostic.version``; ``run`` / ``build`` preserve the value already recorded.
    The ``catalog_hash`` is always (re)derived from the current ``catalog.yaml`` so
    the manifest stays coupled to the inputs that produced the committed bundle.
    """
    from climate_ref_core.regression.manifest import SCHEMA_VERSION, Manifest
    from climate_ref_core.testing import get_catalog_hash

    manifest = Manifest(
        schema=SCHEMA_VERSION if schema is None else schema,
        test_case_version=test_case_version,
        diagnostic_version=diagnostic_version,
        committed=dict(committed),
        native=native,
        catalog_hash=get_catalog_hash(paths.catalog),
    )
    manifest.dump(paths.manifest)
    return manifest


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


class VerbCase(NamedTuple):
    """A test case ready for a verb's loop body, with its paths resolved."""

    diag: Diagnostic
    tc: TestCase
    paths: TestCasePaths
    case_id: str


class VerbSummary(NamedTuple):
    """Summary for a per-case verb."""

    mixed: str
    """Yellow tally line when any case failed, e.g. ``"Replay: {successes} passed, {failures} failed"``."""

    failed_header: str
    """Red header above the failed case list, e.g. ``"Failed replays:"``."""

    success: str
    """Green line when every case succeeded, e.g. ``"All {successes} replay(s) matched ..."``."""


class VerbDriver:
    """
    Shared per-case driver for the ``ref test-cases`` verbs.

    The loop body stays with the verb and reports via :meth:`ok` and :meth:`fail`.
    """

    def __init__(
        self,
        ctx: typer.Context,
        *,
        provider: str | None,
        diagnostic: str | None,
        test_case: str | None,
    ) -> None:
        from climate_ref.provider_registry import ProviderRegistry

        self.console: Console = ctx.obj.console
        self.provider = provider
        # When a specific case is named, an unusable test case is a hard failure.
        self.named = bool(diagnostic or test_case)
        self.registry = ProviderRegistry.build_from_config(ctx.obj.config, ctx.obj.database)
        _validate_provider_in_registry(self.registry, provider)
        _validate_requested_filters(
            self.registry, provider=provider, diagnostic=diagnostic, test_case=test_case
        )
        self.cases = list(
            _iter_test_cases(self.registry, provider=provider, diagnostic=diagnostic, test_case=test_case)
        )
        self.successes = 0
        self.failures: list[str] = []

    def exit_if_empty(self, message: str | None = None) -> None:
        """Exit 0 with a warning when the selectors matched no test cases."""
        if self.cases:
            return
        logger.warning(message or f"No test cases found for provider {self.provider!r}")
        raise typer.Exit(code=0)

    def ready_cases(
        self, *, require_manifest: bool = False, require_catalog: bool = False
    ) -> Iterator[VerbCase]:
        """
        Yield the matched cases with paths resolved, applying the shared skip policy.

        An unlocatable test-data directory or a missing ``manifest.json`` is a hard failure
        when the case was named explicitly, and a warn-and-skip when sweeping.
        A missing catalog is always a hard failure.

        Skips and failures are recorded as the caller iterates,
        so the loop must run to exhaustion for the summary to see them all.
        """
        from climate_ref_core.testing import TestCasePaths

        for diag, tc in self.cases:
            case_id = f"{diag.provider.slug}/{diag.slug}/{tc.name}"
            paths = TestCasePaths.from_diagnostic(diag, tc.name)
            if paths is None:
                self._skip_or_fail(case_id, f"Could not determine test case directory for {case_id}")
                continue
            if require_manifest and not paths.manifest.exists():
                self._skip_or_fail(
                    case_id, f"No manifest.json for {case_id}. Run `ref test-cases mint` first"
                )
                continue
            if require_catalog and not paths.catalog.exists():
                self.fail(case_id, f"No catalog file for {case_id}. Run `ref test-cases fetch` first")
                continue
            yield VerbCase(diag, tc, paths, case_id)

    def ok(self) -> None:
        """Record a successful case."""
        self.successes += 1

    def fail(self, label: str, message: str | None = None) -> None:
        """
        Record a failure, logging ``message`` as an error when given.

        ``label`` is the entry listed under the summary's failed header, usually the case id.
        """
        if message:
            logger.error(message)
        self.failures.append(label)

    def _skip_or_fail(self, case_id: str, message: str) -> None:
        """Fail a case that was named explicitly, and warn-and-skip it when sweeping."""
        if self.named:
            self.fail(case_id, message)
        else:
            logger.warning(message)

    def finish(self, summary: VerbSummary) -> None:
        """Print the verb's summary and exit non-zero when any case failed."""
        self.console.print()

        if self.failures:
            mixed = summary.mixed.format(successes=self.successes, failures=len(self.failures))
            self.console.print(f"[yellow]{mixed}[/yellow]")
            self.console.print(f"[red]{summary.failed_header}[/red]")
            for case in self.failures:
                self.console.print(f"  - {case}")
            raise typer.Exit(code=1)
        self.console.print(f"[green]{summary.success.format(successes=self.successes)}[/green]")
