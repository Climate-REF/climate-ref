"""
Tests that the REF resolves its bundled data with no network and no writable filesystem.

The other tests in this area mock `requests.get`, which only proves that one call site
behaves. These block outbound sockets outright, so a request from any library fails,
and they point the cache at an unwritable directory so nothing can quietly be written.
"""

import socket

import pytest
import requests

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

    attempts: list[object] = []

    def blocked(*args, **kwargs):
        attempts.append(args[-1] if args else None)
        raise NetworkAccessError("network access attempted")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)
    return attempts


@pytest.fixture
def read_only_cache(monkeypatch, tmp_path):
    """
    Point the dataset cache somewhere it cannot be created.

    A regular file is used rather than a directory with the write bit cleared,
    because permissions do not stop a privileged user and do not mean the same
    thing on Windows. Creating a directory beneath a file always fails.
    """
    root = tmp_path / "not-a-directory"
    root.touch()
    monkeypatch.setenv("REF_DATASET_CACHE_DIR", str(root))
    return root


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
    assert no_network


def test_configuration_loads(offline_config, read_only_cache):
    assert offline_config.ignore_datasets_file is None
    assert offline_config.paths.dimensions_cv is None
    assert read_only_cache.is_file()


def test_controlled_vocabulary_resolves_from_the_package(offline_config, no_network):
    resource = offline_config.paths.dimensions_cv_resource

    assert resource.origin == ResourceOrigin.package
    assert CV.load(resource).dimensions
    # The controlled vocabulary has no remote copy, so nothing should even try.
    assert no_network == []


def test_grey_list_resolves_from_the_package(offline_config):
    # Refreshing is attempted and fails on both counts, which must not be fatal.
    refresh_ignore_datasets_file(offline_config)

    resource = offline_config.ignore_datasets_resource
    assert resource.origin == ResourceOrigin.package
    assert resource.read_text()


def test_refreshing_the_grey_list_writes_nothing(offline_config, read_only_cache):
    refresh_ignore_datasets_file(offline_config)

    assert read_only_cache.is_file()


def test_grey_list_resolves_from_the_package_with_a_writable_cache(monkeypatch, tmp_path, no_network):
    """
    The air-gapped case with an ordinary filesystem.

    `refresh_ignore_datasets_file` creates the cache directory before it fetches,
    so the read-only case above never reaches the network.
    This one does, and the download is what fails.
    """
    monkeypatch.setenv("REF_CONFIGURATION", str(tmp_path / "climate_ref"))
    monkeypatch.setenv("REF_DATASET_CACHE_DIR", str(tmp_path / "cache"))
    config = Config.default()

    refresh_ignore_datasets_file(config)

    assert no_network, "the download should have been attempted and blocked"
    assert config.ignore_datasets_resource.origin == ResourceOrigin.package
    assert config.ignore_datasets_resource.read_text()


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
    raises=requests.exceptions.ConnectionError,
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
