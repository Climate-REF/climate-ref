"""
Tests that the REF resolves its bundled data with no network and no writable filesystem.

The other tests in this area mock `requests.get`, which only proves that one call site
behaves. These block outbound sockets outright, so a request from any library fails,
and they point the cache at an unwritable directory so nothing can quietly be written.
"""

import socket

import pytest

from climate_ref.config import Config, refresh_ignore_datasets_file
from climate_ref_core.data import ResourceOrigin
from climate_ref_core.providers import CondaDiagnosticProvider
from climate_ref_core.pycmec.controlled_vocabulary import CV


class NetworkAccessError(OSError):
    """Raised when code under test tries to open a socket."""


@pytest.fixture
def no_network(monkeypatch):
    """
    Block every outbound socket for the duration of a test.

    Patching at the socket layer rather than at `requests` means a call from any
    library is caught, including one added by a future change.
    """

    def blocked(*args, **kwargs):
        raise NetworkAccessError(f"network access attempted: {args[:1]}")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)


@pytest.fixture
def read_only_cache(monkeypatch, tmp_path):
    """Point the dataset cache at a directory that cannot be created."""
    root = tmp_path / "readonly"
    root.mkdir()
    root.chmod(0o500)
    monkeypatch.setenv("REF_DATASET_CACHE_DIR", str(root / "cache"))
    yield root / "cache"
    root.chmod(0o700)


@pytest.fixture
def offline_config(monkeypatch, tmp_path, no_network, read_only_cache):
    """A default configuration on a host with no network and an unwritable cache."""
    monkeypatch.setenv("REF_CONFIGURATION", str(tmp_path / "climate_ref"))
    return Config.default()


def test_the_fixture_really_blocks_the_network(no_network):
    # Guards the guard. A fixture that silently stopped working would make every
    # test below pass for the wrong reason.
    with pytest.raises(NetworkAccessError):
        socket.create_connection(("example.invalid", 80))


def test_configuration_loads(offline_config, read_only_cache):
    assert offline_config.ignore_datasets_file is None
    assert offline_config.paths.dimensions_cv is None
    assert not read_only_cache.exists()


def test_controlled_vocabulary_resolves_from_the_package(offline_config):
    resource = offline_config.paths.dimensions_cv_resource

    assert resource.origin == ResourceOrigin.package
    assert CV.load(resource).dimensions


def test_grey_list_resolves_from_the_package(offline_config):
    # Refreshing is attempted and fails on both counts, which must not be fatal.
    refresh_ignore_datasets_file(offline_config)

    resource = offline_config.ignore_datasets_resource
    assert resource.origin == ResourceOrigin.package
    assert resource.read_text()


def test_refreshing_the_grey_list_writes_nothing(offline_config, read_only_cache):
    refresh_ignore_datasets_file(offline_config)

    assert not read_only_cache.exists()


def test_provider_configure_applies_the_grey_list(offline_config, provider):
    # `configure` is on the solve path, so it must not need the network either.
    provider.configure(offline_config)

    assert offline_config.ignore_datasets_resource.origin == ResourceOrigin.package


def test_repeated_resolution_stays_offline(offline_config):
    # Resolution happens on every access, so a later access must not reach out either.
    for _ in range(3):
        assert offline_config.ignore_datasets_resource.origin == ResourceOrigin.package
        assert offline_config.paths.dimensions_cv_resource.origin == ResourceOrigin.package


def test_a_conda_provider_does_not_need_the_network_to_configure(offline_config):
    # A plain conda provider only records where its prefix will be.
    # `climate_ref_pmp` additionally resolves its conda executable here, which does
    # need the network. See test_offline_known_gaps below.
    conda_provider = CondaDiagnosticProvider("offline-test", "v0")
    conda_provider.configure(offline_config)

    assert conda_provider.prefix == offline_config.paths.software / "conda"


@pytest.mark.xfail(
    reason="climate_ref_pmp downloads micromamba during configure, which is on the solve path",
    raises=Exception,
    strict=True,
)
def test_offline_known_gaps(offline_config):
    """
    Records that building the provider registry still needs the network.

    `PMPDiagnosticProvider.configure` calls `get_conda_exe`, which downloads micromamba
    when it is missing or more than a week old.
    Remove this test once that call is deferred to environment setup.
    """
    pmp = pytest.importorskip("climate_ref_pmp")

    pmp.PMPDiagnosticProvider("PMP", "v0").configure(offline_config)
