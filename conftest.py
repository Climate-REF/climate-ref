"""
Re-useable fixtures etc. for tests that are shared across the whole project

See https://docs.pytest.org/en/7.1.x/reference/fixtures.html#conftest-py-sharing-fixtures-across-multiple-files
"""

from pathlib import Path

import esgpull
import pytest


@pytest.fixture
def esgf_data_dir() -> Path:
    pull = esgpull.Esgpull()

    return pull.config.paths.data


@pytest.fixture
def test_dataset(esgf_data_dir) -> Path:
    return (
        esgf_data_dir
        / "CMIP6"
        / "ScenarioMIP"
        / "CSIRO"
        / "ACCESS-ESM1-5"
        / "ssp126"
        / "r1i1p1f1"
        / "Amon"
        / "tas"
        / "gn"
        / "v20210318"
    )
