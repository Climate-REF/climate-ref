from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from climate_ref.database import Database
from climate_ref.datasets.cmip6 import (
    CMIP6DatasetAdapter,
    _apply_fixes,
)
from climate_ref.datasets.cmip6_parsers import parse_cmip6_complete, parse_cmip6_drs


@pytest.mark.parametrize("parsing_func", [parse_cmip6_complete, parse_cmip6_drs])
def test_parse_exception(parsing_func):
    result = parsing_func("missing_file")

    assert result["INVALID_ASSET"] == "missing_file"
    assert "TRACEBACK" in result


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
            adapter.register_dataset(db_seeded, target_metadata)

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
            adapter.register_dataset(db_seeded, target_metadata)

        # The new version should be in the catalog
        latest_data_catalog = adapter.load_catalog(db_seeded)
        latest_instance_ids = latest_data_catalog.instance_id.unique().tolist()
        assert target_ds not in latest_instance_ids
        assert new_instance_id in latest_instance_ids

    @pytest.mark.parametrize("cmip6_parser", ["complete", "drs"])
    def test_round_trip(self, cmip6_parser, config, cmip6_local_catalogs):
        config.cmip6_parser = cmip6_parser
        catalog = cmip6_local_catalogs[cmip6_parser]

        with Database.from_config(config, run_migrations=True) as database:
            # Indexes and ordering may be different
            adapter = CMIP6DatasetAdapter()
            with database.session.begin():
                for instance_id, data_catalog_dataset in catalog.groupby(adapter.slug_column):
                    adapter.register_dataset(database, data_catalog_dataset)

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

    def test_finalise_datasets(self, config, sample_data_dir):
        """Test that DRS-ingested (unfinalised) datasets get finalised with full metadata."""
        config.cmip6_parser = "drs"

        with Database.from_config(config, run_migrations=True) as database:
            # Ingest via DRS parser (fast, no file I/O, finalised=False)
            adapter = CMIP6DatasetAdapter(config=config)
            drs_catalog = adapter.find_local_datasets(sample_data_dir / "CMIP6")
            assert (~drs_catalog["finalised"]).all(), "DRS parser should produce unfinalised datasets"

            with database.session.begin():
                for _instance_id, data_catalog_dataset in drs_catalog.groupby(adapter.slug_column):
                    adapter.register_dataset(database, data_catalog_dataset)

            # Load catalog from DB - should be unfinalised
            db_catalog = adapter.load_catalog(database)
            assert not db_catalog["finalised"].any(), "DB catalog should have unfinalised datasets"

            # Pick a small subset to finalise (just one instance_id)
            target_instance = db_catalog["instance_id"].iloc[0]
            subset = db_catalog[db_catalog["instance_id"] == target_instance].copy()

            # Finalise the subset (no outer begin() needed —
            # _persist_finalised_metadata manages its own transactions)
            result = adapter.finalise_datasets(database, subset)

            # Verify the result is finalised and has full metadata
            assert result["finalised"].all(), "Finalised datasets should have finalised=True"

            # The complete parser fills in fields that DRS leaves as NA
            # (e.g. frequency should be populated from the file, not just inferred)
            assert result["source_id"].notna().all()
            assert result["experiment_id"].notna().all()

    @pytest.mark.parametrize("cmip6_parser", ["complete", "drs"])
    def test_load_local_datasets(self, config, cmip6_parser, catalog_regression, cmip6_local_catalogs):
        config.cmip6_parser = cmip6_parser
        adapter = CMIP6DatasetAdapter(config=config)
        data_catalog = cmip6_local_catalogs[cmip6_parser]

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


class TestGetParsingFunction:
    def test_returns_complete_parser(self, config):
        config.cmip6_parser = "complete"
        adapter = CMIP6DatasetAdapter(config=config)
        assert adapter.get_parsing_function() is parse_cmip6_complete

    def test_returns_drs_parser(self, config):
        config.cmip6_parser = "drs"
        adapter = CMIP6DatasetAdapter(config=config)
        assert adapter.get_parsing_function() is parse_cmip6_drs


class TestPersistFinalisedMetadata:
    """CMIP6-specific _persist_finalised_metadata tests (shared tests in test_mixin.py)."""

    def test_handles_db_exception_gracefully(self, config, sample_data_dir):
        """Database exceptions during persist are caught, logged, and DataFrame rolled back."""
        config.cmip6_parser = "drs"
        with Database.from_config(config, run_migrations=True) as database:
            adapter = CMIP6DatasetAdapter(config=config)

            # Register some real data
            drs_catalog = adapter.find_local_datasets(sample_data_dir / "CMIP6")
            with database.session.begin():
                for _, group in drs_catalog.groupby(adapter.slug_column):
                    adapter.register_dataset(database, group)

            db_catalog = adapter.load_catalog(database)
            target = db_catalog["instance_id"].iloc[0]
            subset = db_catalog[db_catalog["instance_id"] == target].copy()
            subset["finalised"] = True

            # Force a DB error inside the persist loop
            def exploding_begin():
                raise RuntimeError("simulated DB failure")

            with patch.object(database.session, "begin", side_effect=exploding_begin):
                # Should not raise - exception is caught internally
                adapter._persist_finalised_metadata(database, subset, subset.index)

            # DataFrame should be rolled back to finalised=False to stay consistent with DB
            assert not subset["finalised"].any(), (
                "DataFrame should be rolled back to finalised=False when DB persist fails"
            )


class TestCMIP6AdapterInstanceId:
    """Tests for instance_id construction and column handling in find_local_datasets."""

    def test_instance_id_follows_drs_format(self, cmip6_local_catalogs):
        """instance_id follows 'CMIP6.<8 DRS fields>.<version>' convention."""
        catalog = cmip6_local_catalogs["drs"]

        for _, row in catalog.iterrows():
            parts = row["instance_id"].split(".")
            assert parts[0] == "CMIP6", f"instance_id must start with 'CMIP6': {row['instance_id']}"
            assert len(parts) == 10, f"instance_id must have 10 dot-separated parts: {row['instance_id']}"
            assert parts[1] == row["activity_id"]
            assert parts[2] == row["institution_id"]
            assert parts[3] == row["source_id"]
            assert parts[4] == row["experiment_id"]
            assert parts[5] == row["member_id"]
            assert parts[6] == row["table_id"]
            assert parts[7] == row["variable_id"]
            assert parts[8] == row["grid_label"]
            assert parts[9] == row["version"]

    def test_drs_and_complete_share_common_datasets(self, cmip6_local_catalogs):
        """Both parsers discover a common set of datasets from the same directory."""
        drs_ids = set(cmip6_local_catalogs["drs"]["instance_id"].unique())
        complete_ids = set(cmip6_local_catalogs["complete"]["instance_id"].unique())
        overlap = drs_ids & complete_ids
        assert len(overlap) > 0, "Parsers should discover at least some common datasets"
        # Both should produce non-trivial catalogs
        assert len(drs_ids) > 1
        assert len(complete_ids) > 1

    def test_drs_catalog_has_all_required_columns(self, cmip6_local_catalogs):
        """DRS parser creates all required metadata columns (filling gaps with NA)."""
        adapter = CMIP6DatasetAdapter()
        catalog = cmip6_local_catalogs["drs"]

        all_required = set(adapter.dataset_specific_metadata + adapter.file_specific_metadata)
        assert all_required.issubset(set(catalog.columns)), (
            f"Missing columns: {all_required - set(catalog.columns)}"
        )

    def test_complete_parser_populates_core_metadata(self, cmip6_local_catalogs):
        """Complete parser provides non-NA values for core metadata fields."""
        catalog = cmip6_local_catalogs["complete"]

        core_fields = ["source_id", "experiment_id", "variable_id", "frequency", "grid_label"]
        for field in core_fields:
            assert catalog[field].notna().all(), f"Complete parser should populate '{field}'"
        assert catalog["finalised"].all(), "Complete parser should mark all datasets as finalised"

    def test_drs_parser_marks_unfinalised(self, cmip6_local_catalogs):
        """DRS parser marks all datasets as unfinalised."""
        catalog = cmip6_local_catalogs["drs"]
        assert (~catalog["finalised"]).all(), "DRS parser should set finalised=False"

    def test_adapter_default_config(self):
        """Adapter uses default config and n_jobs=1 when not specified."""
        adapter = CMIP6DatasetAdapter()
        assert adapter.n_jobs == 1
        assert adapter.config is not None

    def test_adapter_custom_n_jobs(self, config):
        """Adapter stores the provided n_jobs value."""
        adapter = CMIP6DatasetAdapter(n_jobs=4, config=config)
        assert adapter.n_jobs == 4
