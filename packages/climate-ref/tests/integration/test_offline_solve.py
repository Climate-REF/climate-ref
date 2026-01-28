"""
Tests to verify that solve operations work without network access.

These tests ensure that after provider setup is complete, no network access
is required during solve operations. This is critical for HPC environments
where compute nodes don't have internet access.

The tests run one diagnostic from each provider to verify that execution
works without network access. These tests are expected to FAIL until
provider lifecycle hooks are implemented that allow providers to pre-fetch
all required data via `ref providers setup`.
"""

import socket

import pytest
from pytest_socket import SocketBlockedError

from climate_ref.config import Config, DiagnosticProviderConfig
from climate_ref.database import Database
from climate_ref.provider_registry import ProviderRegistry
from climate_ref.solver import solve_required_executions


class TestNetworkBlockingWorks:
    """Verify that pytest-socket is correctly blocking network access."""

    @pytest.mark.disable_socket
    def test_socket_is_blocked(self):
        """Verify that socket connections are blocked by pytest-socket."""
        with pytest.raises(SocketBlockedError):
            # This should raise because socket is disabled
            socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    def test_socket_works_without_marker(self):
        """Verify that socket connections work without the marker."""
        # This should NOT raise
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.close()


@pytest.fixture
def config_with_providers(tmp_path, monkeypatch) -> Config:
    """
    Create a config that uses real providers (esmvaltool, pmp, ilamb).

    These providers have data registries that require network access
    to fetch reference data.
    """
    ref_config_dir = tmp_path / "climate_ref_offline_test"
    monkeypatch.setenv("REF_CONFIGURATION", str(ref_config_dir))

    cfg = Config.default()
    cfg.diagnostic_providers = [
        DiagnosticProviderConfig(provider="climate_ref_esmvaltool"),
        DiagnosticProviderConfig(provider="climate_ref_pmp"),
        DiagnosticProviderConfig(provider="climate_ref_ilamb"),
    ]
    cfg.executor.executor = "climate_ref.executor.SynchronousExecutor"
    cfg.save()

    return cfg


@pytest.fixture
def db_for_offline_test(config_with_providers) -> Database:
    """Create a fresh database for offline testing."""
    with Database.from_config(config_with_providers) as db:
        yield db


@pytest.mark.disable_socket
def test_provider_registry_build_requires_no_network(db_for_offline_test, config_with_providers):
    """
    Verify that building the provider registry works without network access.

    The provider registry should be buildable without network access,
    as all provider modules should be locally installed. This test imports
    providers that have dataset registries and verifies that the import
    and configuration process does not require network access.

    This is important because the provider registry is built on every
    solve operation, and solve operations should work on compute nodes
    without internet access.
    """
    # This should work offline - providers are installed locally
    provider_registry = ProviderRegistry.build_from_config(config_with_providers, db_for_offline_test)

    # Basic sanity check - we should have 3 providers
    assert len(provider_registry.providers) == 3

    # Verify we have the expected providers
    provider_slugs = {p.slug for p in provider_registry.providers}
    assert provider_slugs == {"esmvaltool", "pmp", "ilamb"}


@pytest.mark.disable_socket
def test_solve_dry_run_requires_no_network(db_seeded, config):
    """
    Verify that solve with dry_run=True works without network access.

    A dry run should not require any data to be fetched, as it only
    determines what executions would be created without actually
    executing them.
    """
    # Dry run should work offline
    solve_required_executions(
        config=config,
        db=db_seeded,
        dry_run=True,
    )


@pytest.mark.disable_socket
@pytest.mark.slow
def test_solve_one_per_provider_offline(db_seeded, config_with_providers):
    """
    Verify that running one diagnostic per provider works without network access.

    This test runs the solver with one_per_provider=True, which limits
    execution to a single diagnostic from each provider. This is a more
    comprehensive test that verifies the entire solve pipeline works offline.

    This test is expected to FAIL until provider lifecycle hooks are
    implemented. Once `ref providers setup` can pre-fetch all required
    data, this test should pass.
    """
    with Database.from_config(config_with_providers) as db:
        # Build provider registry
        provider_registry = ProviderRegistry.build_from_config(config_with_providers, db)

        # Verify we have all three providers
        assert len(provider_registry.providers) == 3

        # Run solve with one_per_provider to limit execution scope
        # This should work offline after setup has been run
        solve_required_executions(
            config=config_with_providers,
            db=db,
            dry_run=False,
            one_per_provider=True,
        )
