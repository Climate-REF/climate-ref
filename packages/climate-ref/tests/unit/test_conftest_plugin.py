"""Tests for the shared pytest plugin shipped with ``climate_ref``."""

from pathlib import Path

from climate_ref.config import DEFAULT_IGNORE_DATASETS_FILENAME, Config
from climate_ref.conftest_plugin import _use_local_ignore_datasets_file, packaged_ignore_datasets_file

# The canonical copy, served over `DEFAULT_IGNORE_DATASETS_URL` from the default branch.
REPO_ROOT_IGNORE_FILE = Path(__file__).parents[4] / DEFAULT_IGNORE_DATASETS_FILENAME


def test_packaged_ignore_datasets_file_is_resolvable():
    # Resolved from the installed package, so this also holds for a wheel install.
    assert packaged_ignore_datasets_file().is_file()


def test_packaged_ignore_datasets_file_matches_repo_root():
    # The root copy is what `DEFAULT_IGNORE_DATASETS_URL` serves, so the two must not drift apart.
    assert REPO_ROOT_IGNORE_FILE.is_file()
    assert packaged_ignore_datasets_file().read_bytes() == REPO_ROOT_IGNORE_FILE.read_bytes()


def test_use_local_ignore_datasets_file_disables_fetching():
    cfg = Config.default()

    _use_local_ignore_datasets_file(cfg)

    assert cfg.ignore_datasets_file == packaged_ignore_datasets_file()
    # An empty URL short-circuits `refresh_ignore_datasets_file`, keeping tests offline.
    assert cfg.ignore_datasets_url == ""
