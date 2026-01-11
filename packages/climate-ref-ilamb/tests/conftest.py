"""
Shared fixtures for ILAMB provider tests.
"""

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Path to the package-local test data directory."""
    return Path(__file__).parent / "test-data"


@pytest.fixture(scope="session")
def catalog_dir(test_data_dir: Path) -> Path:
    """Path to the package-local catalogs directory."""
    return test_data_dir / "catalogs"


@pytest.fixture(scope="session")
def regression_dir(test_data_dir: Path) -> Path:
    """Path to the package-local regression data directory."""
    return test_data_dir / "regression"
