from pathlib import Path

import climate_ref_pmp
import pytest


@pytest.fixture(scope="module")
def pdo_example_dir() -> Path:
    """Captured PMP PDO output used as input fixture by the driver/metric unit tests."""
    return Path(__file__).parent / "test-data" / "pdo-example"


@pytest.fixture
def provider(tmp_path, monkeypatch):
    # The provider is a module-level singleton shared with later tests in the same session,
    # so its state is restored afterwards rather than left pointing at the mock executable
    provider = climate_ref_pmp.provider
    prefix = tmp_path / "conda"
    prefix.mkdir()
    conda_exe = prefix / "mock_micromamba"
    conda_exe.touch()
    monkeypatch.setattr(provider, "_prefix", prefix)
    monkeypatch.setattr(provider, "_conda_exe", conda_exe)
    provider.env_path.mkdir()

    return provider
