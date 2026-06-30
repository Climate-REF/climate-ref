from pathlib import Path

import climate_ref_pmp
import pytest


@pytest.fixture(scope="module")
def pdo_example_dir() -> Path:
    """Captured PMP PDO output used as input fixture by the driver/metric unit tests."""
    return Path(__file__).parent / "test-data" / "pdo-example"


@pytest.fixture
def provider(tmp_path):
    provider = climate_ref_pmp.provider
    provider.prefix = tmp_path / "conda"
    provider.prefix.mkdir()
    provider._conda_exe = provider.prefix / "mock_micromamba"
    provider._conda_exe.touch()
    provider.env_path.mkdir()

    return provider
