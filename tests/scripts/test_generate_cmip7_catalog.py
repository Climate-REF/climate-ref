"""
Tests for `scripts/generate_cmip7_catalog.py`.

No network access is performed: the selection works on hand-written STAC items
and the header parsing on the decimated CMIP7 files in `tests/test-data/cmip7`.
"""

import importlib.util
from pathlib import Path

import netCDF4
import pandas as pd
import pytest

from climate_ref.config import Config
from climate_ref.datasets import CMIP7DatasetAdapter
from climate_ref.datasets.cmip7_parsers import parse_cmip7_dataset
from climate_ref.datasets.utils import sort_data_catalog

REPO_ROOT = Path(__file__).parents[2]
SCRIPT = REPO_ROOT / "scripts" / "generate_cmip7_catalog.py"
CMIP7_DATA = REPO_ROOT / "tests" / "test-data" / "cmip7"
WANTED = {("tas_tavg-h2m-hxy-u", "mon", "glb")}


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("generate_cmip7_catalog", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _item(experiment_id, variant_label, parent=None, branded="tas_tavg-h2m-hxy-u", **properties):
    parent_experiment_id, parent_variant_label = parent or (None, None)
    return {
        "id": f"{experiment_id}.{variant_label}.{branded}",
        "properties": {
            "cmip7:source_id": "MODEL",
            "cmip7:experiment_id": experiment_id,
            "cmip7:variant_label": variant_label,
            "cmip7:variable_branded_name": branded,
            "cmip7:frequency": "mon",
            "cmip7:region": "glb",
            "cmip7:parent_source_id": "MODEL" if parent else None,
            "cmip7:parent_experiment_id": parent_experiment_id,
            "cmip7:parent_variant_label": parent_variant_label,
            **properties,
        },
        "assets": {},
    }


def _selected(script, items, max_variants=1):
    return {item["id"] for item in script.select_datasets(items, WANTED, max_variants)}


def test_keeps_the_lowest_ensemble_members(script):
    items = [_item("historical", label) for label in ("r10i1p1f1", "r2i1p1f1", "r1i1p1f1")]

    assert _selected(script, items, max_variants=2) == {
        "historical.r1i1p1f1.tas_tavg-h2m-hxy-u",
        "historical.r2i1p1f1.tas_tavg-h2m-hxy-u",
    }


def test_adds_parents_of_selected_members(script):
    """A child's parent member is kept even if it is not among the lowest ones, and so is its parent."""
    items = [
        _item("piControl", "r1i1p1f1"),
        _item("piControl", "r5i1p1f1"),
        _item("historical", "r1i1p1f1", parent=("piControl", "r5i1p1f1")),
        _item("historical", "r3i1p1f1", parent=("piControl", "r1i1p1f1")),
        _item("scen7-h", "r1i1p1f1", parent=("historical", "r3i1p1f1")),
    ]

    assert _selected(script, items) == {item["id"] for item in items}


def test_leaves_out_other_variables_and_retracted_datasets(script):
    items = [
        _item("historical", "r1i1p1f1"),
        _item("historical", "r1i1p1f1", branded="tas_tmaxavg-h2m-hxy-u"),
        _item("piControl", "r1i1p1f1", retracted=True),
    ]

    assert _selected(script, items) == {"historical.r1i1p1f1.tas_tavg-h2m-hxy-u"}


def test_data_files_skips_globus_links(script):
    path = "MIP-DRS7/CMIP7/CMIP/INST/MODEL/historical/r1i1p1f1/glb/mon/tas/tavg-h2m-hxy-u/gn/v1/tas.nc"
    item = {
        "assets": {
            "tas.nc": {"href": f"https://node.example/thredds/fileServer/{path}", "roles": ["data"]},
            "globus": {
                "href": f"https://app.globus.org/file-manager?origin_path=/{path}",
                "roles": ["data"],
            },
            "thumbnail": {"href": "https://node.example/thumb.png", "roles": ["thumbnail"]},
        }
    }

    assert script.data_files(item) == [(path, f"https://node.example/thredds/fileServer/{path}")]


def test_build_catalog_matches_a_local_scan(script, config: Config):
    """Rows parsed from headers make the same catalog as the complete parser scanning the files."""
    files = sorted(CMIP7_DATA.rglob("*.nc"))
    rows = []
    for file in files:
        with netCDF4.Dataset(file) as ds:
            rows.append(parse_cmip7_dataset(ds, str(file)))

    config.cmip7_parser = "complete"
    scanned = CMIP7DatasetAdapter(config=config).find_local_datasets(CMIP7_DATA)
    built = script.build_catalog(rows)

    assert list(built.columns) == [
        *CMIP7DatasetAdapter.dataset_specific_metadata,
        "branded_variable",
        *CMIP7DatasetAdapter.file_specific_metadata,
    ]
    pd.testing.assert_frame_equal(
        sort_data_catalog(built).reset_index(drop=True),
        sort_data_catalog(scanned[built.columns]).reset_index(drop=True),
    )


def test_read_header_matches_the_complete_parser(script):
    """Reading through fsspec and h5netcdf gives what netCDF4 gives for the same file."""
    pytest.importorskip("h5netcdf")
    file = next(CMIP7_DATA.rglob("tas_*.nc"))
    path = str(file.relative_to(CMIP7_DATA))

    row = script.read_header(file.as_uri(), path)

    with netCDF4.Dataset(file) as ds:
        expected = parse_cmip7_dataset(ds, f"{{data_dir}}/{path}")
    assert row == {key: script._plain(value) for key, value in expected.items()}


def test_complete_datasets_leaves_out_partly_read_datasets(script):
    """A dataset with a file that could not be read would make the catalog claim a gap that is not there."""
    files_by_dataset = {
        "complete": [("a/1.nc", "https://node/a/1.nc"), ("a/2.nc", "https://node/a/2.nc")],
        "partial": [("b/1.nc", "https://node/b/1.nc"), ("b/2.nc", "https://node/b/2.nc")],
    }
    rows = {f"{{data_dir}}/{path}": {"path": path} for path in ("a/1.nc", "a/2.nc", "b/1.nc")}

    assert script.complete_datasets(files_by_dataset, rows) == [{"path": "a/1.nc"}, {"path": "a/2.nc"}]
