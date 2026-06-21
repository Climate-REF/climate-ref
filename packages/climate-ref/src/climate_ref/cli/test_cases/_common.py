"""
Helpers shared across several ``ref test-cases`` commands.

``_iter_test_cases`` enumerates ``(diagnostic, test_case)`` pairs from the
provider registry (used by ``sync`` / ``replay`` / ``mint`` / ``build`` /
``ci-gate``) and ``_write_test_case_manifest`` authors the committed
``manifest.json`` (used by ``run`` / ``mint`` / ``build``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterator

    from climate_ref.provider_registry import ProviderRegistry
    from climate_ref_core.diagnostics import Diagnostic
    from climate_ref_core.regression.manifest import Manifest, NativeEntry
    from climate_ref_core.testing import TestCase, TestCasePaths


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
