"""
Helpers shared across several ``ref test-cases`` commands.

``_iter_test_cases`` enumerates ``(diagnostic, test_case)`` pairs from the
provider registry (used by ``sync`` / ``replay`` / ``mint`` / ``build`` /
``ci-gate``) and ``_write_test_case_manifest`` authors the committed
``manifest.json`` (used by ``run`` / ``mint`` / ``build``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import typer
from loguru import logger

if TYPE_CHECKING:
    from collections.abc import Iterator

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
