import datetime

import numpy as np
import pandas as pd
import pytest

from climate_ref.database import Database
from climate_ref.datasets.cmip6 import (
    CMIP6DatasetAdapter,
    _apply_fixes,
)
from climate_ref.datasets.cmip6_parsers import parse_cmip6_complete, parse_cmip6_drs
from climate_ref.datasets.utils import clean_branch_time, parse_datetime


def testparse_datetime():
    pd.testing.assert_series_equal(
        parse_datetime(pd.Series(["2021-01-01 00:00:00", "1850-01-17 00:29:59.999993", None])),
        pd.Series(
            [datetime.datetime(2021, 1, 1, 0, 0), datetime.datetime(1850, 1, 17, 0, 29, 59, 999993), None],
            dtype="object",
        ),
    )


@pytest.mark.parametrize("parsing_func", [parse_cmip6_complete, parse_cmip6_drs])
def test_parse_exception(parsing_func):
    result = parsing_func("missing_file")

    assert result["INVALID_ASSET"] == "missing_file"
    assert "TRACEBACK" in result


def testclean_branch_time():
    inp = pd.Series(["0D", "12", "12.0", "12.000", "12.0000", "12.00000", None, np.nan])
    exp = pd.Series([0.0, 12.0, 12.0, 12.0, 12.0, 12.0, np.nan, np.nan])

    pd.testing.assert_series_equal(clean_branch_time(inp), exp)


class TestCMIP6Adapter:
    def test_catalog_empty(self, db):
        adapter = CMIP6DatasetAdapter()
        df = adapter.load_catalog(db)
        assert df.empty

    @pytest.mark.parametrize("cmip6_parser", ["complete", "drs"])
    def test_load_catalog(self, cmip6_parser, db_seeded, catalog_regression, sample_data_dir, config):
        config.cmip6_parser = cmip6_parser

        adapter = CMIP6DatasetAdapter(config=config)
        df = adapter.load_catalog(db_seeded)

        for k in adapter.dataset_specific_metadata + adapter.file_specific_metadata:
            assert k in df.columns

        # The order of the rows may be flakey due to sqlite ordering and the created time resolution
        catalog_regression(
            df.sort_values(["instance_id", "start_time"]), basename=f"cmip6_catalog_db_{cmip6_parser}"
        )

    def test_load_catalog_multiple_versions(self, config, db_seeded, catalog_regression, sample_data_dir):
        adapter = CMIP6DatasetAdapter()
        data_catalog = adapter.load_catalog(db_seeded)
        target_ds = "CMIP6.CMIP.CSIRO.ACCESS-ESM1-5.historical.r1i1p1f1.Amon.tas.gn.v20191115"
        target_metadata = data_catalog[data_catalog["instance_id"] == target_ds].copy()

        # Make an old version
        target_metadata.loc[:, "version"] = "v20000101"
        target_metadata.loc[:, "instance_id"] = target_ds.replace("v20191115", "v20000101")
        with db_seeded.session.begin():
            adapter.register_dataset(config, db_seeded, target_metadata)

        # An older version should not be in the catalog
        pd.testing.assert_frame_equal(
            data_catalog.sort_values(["instance_id", "start_time"]),
            adapter.load_catalog(db_seeded).sort_values(["instance_id", "start_time"]),
        )

        # Make a new version
        target_metadata.loc[:, "version"] = "v20230101"
        new_instance_id = target_ds.replace("v20191115", "v20230101")
        target_metadata.loc[:, "instance_id"] = new_instance_id
        with db_seeded.session.begin():
            adapter.register_dataset(config, db_seeded, target_metadata)

        # The new version should be in the catalog
        latest_data_catalog = adapter.load_catalog(db_seeded)
        latest_instance_ids = latest_data_catalog.instance_id.unique().tolist()
        assert target_ds not in latest_instance_ids
        assert new_instance_id in latest_instance_ids

    @pytest.mark.parametrize("cmip6_parser", ["complete", "drs"])
    def test_round_trip(self, cmip6_parser, config, sample_data_dir):
        config.cmip6_parser = cmip6_parser

        with Database.from_config(config, run_migrations=True) as database:
            catalog = CMIP6DatasetAdapter(config=config).find_local_datasets(sample_data_dir / "CMIP6")

            # Indexes and ordering may be different
            adapter = CMIP6DatasetAdapter()
            with database.session.begin():
                for instance_id, data_catalog_dataset in catalog.groupby(adapter.slug_column):
                    adapter.register_dataset(config, database, data_catalog_dataset)

            local_data_catalog = (
                catalog.drop(columns=["time_range"])
                .sort_values(["instance_id", "start_time"])
                .reset_index(drop=True)
            )

            db_data_catalog = (
                adapter.load_catalog(database)
                .sort_values(["instance_id", "start_time"])
                .reset_index(drop=True)
            )

            # Normalize null values - convert None to np.nan for consistent comparison
            # Opt into future pandas behavior to avoid deprecation warnings
            with pd.option_context("future.no_silent_downcasting", True):
                local_normalized = local_data_catalog.fillna(np.nan).infer_objects()
                db_normalized = db_data_catalog.fillna(np.nan).infer_objects()

            pd.testing.assert_frame_equal(
                local_normalized,
                db_normalized,
                check_like=True,
            )

    @pytest.mark.parametrize("cmip6_parser", ["complete", "drs"])
    def test_load_local_datasets(self, config, cmip6_parser, sample_data_dir, catalog_regression):
        # Set the parser in the config
        config.cmip6_parser = cmip6_parser

        # Parse the local datasets
        adapter = CMIP6DatasetAdapter(config=config)
        data_catalog = adapter.find_local_datasets(sample_data_dir / "CMIP6")

        if cmip6_parser == "complete":
            assert data_catalog["finalised"].all()
        else:
            assert (~data_catalog["finalised"]).all()

        # TODO: add time_range to the db?
        assert sorted(data_catalog.columns.tolist()) == sorted(
            [*adapter.dataset_specific_metadata, *adapter.file_specific_metadata, "time_range"]
        )

        catalog_regression(
            data_catalog.sort_values(["instance_id", "start_time"]),
            basename=f"cmip6_catalog_local_{cmip6_parser}",
        )


def test_apply_fixes():
    df = pd.DataFrame(
        {
            "instance_id": ["dataset_001", "dataset_001", "dataset_002"],
            "parent_variant_label": ["r1i1p1f1", "r1i1p1f2", "r1i1p1f2"],
            "variant_label": ["r1i1p1f1", "r1i1p1f1", "r1i1p1f2"],
            "branch_time_in_child": ["0D", "12", "12.0"],
            "branch_time_in_parent": [None, np.nan, "12.0"],
        }
    )

    res = _apply_fixes(df)

    exp = pd.DataFrame(
        {
            "instance_id": ["dataset_001", "dataset_001", "dataset_002"],
            "parent_variant_label": ["r1i1p1f1", "r1i1p1f1", "r1i1p1f2"],
            "variant_label": ["r1i1p1f1", "r1i1p1f1", "r1i1p1f2"],
            "branch_time_in_child": [0.0, 12.0, 12.0],
            "branch_time_in_parent": [np.nan, np.nan, 12.0],
        }
    )
    pd.testing.assert_frame_equal(res, exp)
