"""
Tests for the historical extension in `scripts/generate_cmip7_catalog.py`.

The script is not an importable module, so it is loaded by path.
"""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

SCRIPT = Path(__file__).parents[2] / "scripts" / "generate_cmip7_catalog.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("generate_cmip7_catalog", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script():
    return _load_script()


def _catalog(rows):
    return pd.DataFrame(
        [{"experiment_id": experiment_id, "end_time": end_time} for experiment_id, end_time in rows]
    )


def test_extends_every_calendar(script):
    """A 360_day model stamps mid-December at 00:00:00, so matching the full timestamp drops it."""
    catalog = _catalog(
        [
            ("historical", "2014-12-16 12:00:00"),
            ("historical", "2014-12-16 00:00:00"),
        ]
    )

    extended = script._extend_historical_end(catalog, "2021-12-16 12:00:00")

    assert extended == 2
    assert list(catalog["end_time"]) == ["2021-12-16 12:00:00"] * 2


def test_leaves_short_runs_and_other_experiments_alone(script):
    """Only full-length historical runs are lifted, so nothing newly satisfies a timerange."""
    catalog = _catalog(
        [
            ("historical", "1854-12-16 12:00:00"),
            ("ssp585", "2014-12-16 12:00:00"),
            ("historical", None),
        ]
    )

    extended = script._extend_historical_end(catalog, "2021-12-16 12:00:00")

    assert extended == 0
    assert list(catalog["end_time"][:2]) == ["1854-12-16 12:00:00", "2014-12-16 12:00:00"]
    # The fx row carries no end_time and must stay that way.
    assert catalog["end_time"].isna().iloc[2]
