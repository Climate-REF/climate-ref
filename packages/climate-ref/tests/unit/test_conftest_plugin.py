"""Tests for the shared pytest plugin shipped with ``climate_ref``."""

from pathlib import Path

import pytest

from climate_ref.config import BUNDLED_IGNORE_DATASETS, DEFAULT_IGNORE_DATASETS_FILENAME, Config
from climate_ref.conftest_plugin import _use_local_ignore_datasets_file, packaged_ignore_datasets_file
from climate_ref_core.data import ResourceOrigin

# The canonical copy, served over `DEFAULT_IGNORE_DATASETS_URL` from the default branch.
# Only present in a source checkout, so tests using it skip for an installed wheel.
REPO_ROOT_IGNORE_FILE = Path(__file__).parents[4] / DEFAULT_IGNORE_DATASETS_FILENAME


def test_packaged_ignore_datasets_file_is_resolvable():
    # Materialised from the installed package, so this also holds for a wheel install.
    assert packaged_ignore_datasets_file().is_file()


def test_packaged_ignore_datasets_file_matches_repo_root():
    # The root copy is what `DEFAULT_IGNORE_DATASETS_URL` serves, so the two must not drift apart.
    # `scripts/sync_grey_list.py` keeps them identical, and runs as a pre-commit hook.
    if not REPO_ROOT_IGNORE_FILE.is_file():
        pytest.skip("not running from a source checkout")

    assert BUNDLED_IGNORE_DATASETS.read_text() == REPO_ROOT_IGNORE_FILE.read_text(encoding="utf-8"), (
        "The packaged grey list has drifted from the canonical copy. "
        "Run `python scripts/sync_grey_list.py` to update it."
    )


def test_use_local_ignore_datasets_file_disables_fetching():
    cfg = Config.default()

    _use_local_ignore_datasets_file(cfg)

    assert cfg.ignore_datasets_file == packaged_ignore_datasets_file()
    # Pinned as an override, so a stale cache on the host cannot leak into a test.
    assert cfg.ignore_datasets_resource.origin == ResourceOrigin.override
    # An empty URL short-circuits `refresh_ignore_datasets_file`, keeping tests offline.
    assert cfg.ignore_datasets_url == ""
