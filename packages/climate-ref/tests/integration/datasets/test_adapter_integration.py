"""Integration tests for the CMIP6 and CMIP7 dataset adapters on real files.

The unit tests mock the complete parser or only create empty files in a DRS tree.
These tests run both parsers on real NetCDF files (decimated CMIP6 sample data and
decimated published CMIP7 data) and push what they find through a real database.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from climate_ref.database import Database
from climate_ref.datasets.cmip6 import _primary_activity
from climate_ref.datasets.utils import sort_data_catalog


def _register(database: Database, adapter, catalog: pd.DataFrame) -> None:
    with database.session.begin():
        for _, dataset in catalog.groupby(adapter.slug_column):
            adapter.register_dataset(database, dataset)


@pytest.mark.parametrize("parser", ["complete", "drs"])
def test_find_local_datasets(parser, adapter_config, adapter_local_catalogs):
    """Each parser finds the datasets with every metadata column and the right finalisation state."""
    adapter = adapter_config.adapter_cls()
    catalog = adapter_local_catalogs[parser]
    assert not catalog.empty

    required = set(adapter.dataset_specific_metadata + adapter.file_specific_metadata)
    assert required <= set(catalog.columns), f"Missing columns: {required - set(catalog.columns)}"

    for instance_id in catalog["instance_id"]:
        parts = instance_id.split(".")
        assert parts[0] == adapter_config.instance_id_prefix, instance_id
        assert len(parts) == adapter_config.instance_id_part_count, instance_id

    if parser == "complete":
        assert catalog["finalised"].all()
        for field in adapter_config.complete_parser_core_fields:
            assert catalog[field].notna().all(), f"Complete parser should populate '{field}'"
        assert len(adapter.validate_data_catalog(catalog)) == len(catalog)
    else:
        assert (~catalog["finalised"]).all()
        for field in adapter.columns_requiring_finalisation:
            if field in catalog.columns:
                assert catalog[field].isna().all(), f"The DRS parser should leave '{field}' NA"


def test_drs_and_complete_agree(adapter_config, adapter_local_catalogs):
    """
    Both parsers find the same datasets and the same values for the DRS fields.

    ``activity_id`` is compared by its primary activity alone: a file that belongs to
    several lists them all in its attribute (``"C4MIP CDRMIP"``), while a DRS path can
    only name the one it is published under. Comparing them through
    :func:`_primary_activity` -- the same rule the ``instance_id`` is built with --
    keeps the field under test, and fails if a dataset ever turns up whose DRS activity
    is not the first the attribute names, which is the assumption that rule rests on.
    """
    adapter = adapter_config.adapter_cls()
    drs_catalog = adapter_local_catalogs["drs"]
    complete_catalog = adapter_local_catalogs["complete"]

    assert set(drs_catalog["instance_id"]) == set(complete_catalog["instance_id"])

    # Only check fields present in both catalogs (e.g. CMIP6 has no mip_era)
    candidate_fields = [*adapter.dataset_id_metadata, "version", "mip_era"]
    fields = [f for f in candidate_fields if f in drs_catalog.columns and f in complete_catalog.columns]
    for instance_id in drs_catalog["instance_id"].unique():
        drs_row = drs_catalog[drs_catalog["instance_id"] == instance_id].iloc[0]
        complete_row = complete_catalog[complete_catalog["instance_id"] == instance_id].iloc[0]
        for field in fields:
            drs_value = _primary_activity(field, drs_row[field])
            assert drs_value == _primary_activity(field, complete_row[field]), (
                f"Field '{field}' differs for {instance_id}: "
                f"DRS={drs_row[field]!r} vs complete={complete_row[field]!r}"
            )


@pytest.mark.parametrize("parser", ["complete", "drs"])
def test_round_trip(parser, config, adapter_config, adapter_local_catalogs):
    """Registering and reloading a catalog gives the same frame, including the derived columns."""
    setattr(config, adapter_config.parser_config_attr, parser)
    catalog = adapter_local_catalogs[parser]

    with Database.from_config(config, run_migrations=True) as database:
        adapter = adapter_config.adapter_cls()
        _register(database, adapter, catalog)
        loaded = adapter.load_catalog(database)

    for column in adapter.derived_metadata:
        assert column in loaded.columns, f"Derived column '{column}' missing: {sorted(loaded.columns)}"

    def normalise(df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop(columns=adapter_config.roundtrip_exclude_columns, errors="ignore")
        # Normalise null values for a consistent comparison
        return sort_data_catalog(df).reset_index(drop=True).fillna(np.nan).infer_objects()

    pd.testing.assert_frame_equal(normalise(catalog), normalise(loaded), check_like=True)


def test_finalise_datasets(config, adapter_config, adapter_local_catalogs):
    """Datasets ingested with the DRS parser are finalised from their files."""
    setattr(config, adapter_config.parser_config_attr, "drs")

    with Database.from_config(config, run_migrations=True) as database:
        adapter = adapter_config.adapter_cls(config=config)
        _register(database, adapter, adapter_local_catalogs["drs"])

        db_catalog = adapter.load_catalog(database)
        assert not db_catalog["finalised"].any()

        target = db_catalog["instance_id"].iloc[0]
        result = adapter.finalise_datasets(database, db_catalog[db_catalog["instance_id"] == target].copy())

        assert result["finalised"].all()
        expected = adapter_local_catalogs["complete"].set_index("path")
        for field in adapter_config.complete_parser_core_fields:
            assert (result[field].to_numpy() == expected.loc[result["path"], field].to_numpy()).all(), field
