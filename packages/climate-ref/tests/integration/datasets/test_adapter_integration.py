"""Integration tests for CMIP6/CMIP7 dataset adapters.

Tests that exercise real file I/O, database round-trips, and finalisation logic
for the CMIP6 and CMIP7 dataset adapters.

Both the "drs" and "complete" parsers are tested to ensure they can be used interchangeably and produce
consistent results.
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from climate_ref.database import Database


class TestLocalDatasets:
    """Local dataset discovery tests, parameterised over adapter types."""

    @pytest.mark.parametrize("parser", ["complete", "drs"])
    def test_find_local_datasets_columns_and_finalised(self, parser, adapter_config, adapter_local_catalogs):
        """find_local_datasets returns all metadata columns and correct finalised status."""
        catalog = adapter_local_catalogs[parser]
        adapter = adapter_config.adapter_cls()

        all_required = set(adapter.dataset_specific_metadata + adapter.file_specific_metadata)
        assert all_required.issubset(set(catalog.columns)), (
            f"Missing columns: {all_required - set(catalog.columns)}"
        )

        if parser == "complete":
            assert catalog["finalised"].all()
        else:
            assert (~catalog["finalised"]).all()

    def test_instance_id_format(self, adapter_config, adapter_local_catalogs):
        """instance_id has the correct prefix and part count."""
        catalog = adapter_local_catalogs["drs"]

        for _, row in catalog.iterrows():
            parts = row["instance_id"].split(".")
            assert parts[0] == adapter_config.instance_id_prefix, (
                f"instance_id must start with '{adapter_config.instance_id_prefix}': {row['instance_id']}"
            )
            assert len(parts) == adapter_config.instance_id_part_count, (
                f"instance_id must have {adapter_config.instance_id_part_count} "
                f"dot-separated parts: {row['instance_id']}"
            )

    def test_drs_catalog_has_all_required_columns(self, adapter_config, adapter_local_catalogs):
        """DRS parser creates all required metadata columns (filling gaps with NA)."""
        adapter = adapter_config.adapter_cls()
        catalog = adapter_local_catalogs["drs"]

        all_required = set(adapter.dataset_specific_metadata + adapter.file_specific_metadata)
        assert all_required.issubset(set(catalog.columns)), (
            f"Missing columns: {all_required - set(catalog.columns)}"
        )

    def test_complete_parser_populates_core_metadata(self, adapter_config, adapter_local_catalogs):
        """Complete parser provides non-NA values for core metadata fields."""
        catalog = adapter_local_catalogs["complete"]

        for field in adapter_config.complete_parser_core_fields:
            assert catalog[field].notna().all(), f"Complete parser should populate '{field}'"
        assert catalog["finalised"].all(), "Complete parser should mark all datasets as finalised"

    def test_drs_parser_marks_unfinalised(self, adapter_config, adapter_local_catalogs):
        """DRS parser marks all datasets as unfinalised."""
        catalog = adapter_local_catalogs["drs"]
        assert (~catalog["finalised"]).all(), "DRS parser should set finalised=False"

    def test_drs_parser_leaves_non_drs_as_na(self, adapter_config, adapter_local_catalogs):
        """DRS parser leaves columns_requiring_finalisation as NA."""
        catalog = adapter_local_catalogs["drs"]
        adapter = adapter_config.adapter_cls()

        for field_name in adapter.columns_requiring_finalisation:
            if field_name in catalog.columns:
                assert catalog[field_name].isna().all(), (
                    f"Non-DRS field '{field_name}' should be NA for DRS parser"
                )

    def test_drs_and_complete_share_common_datasets(self, adapter_local_catalogs):
        """Both parsers discover a common set of datasets from the same directory."""
        drs_ids = set(adapter_local_catalogs["drs"]["instance_id"].unique())
        complete_ids = set(adapter_local_catalogs["complete"]["instance_id"].unique())
        overlap = drs_ids & complete_ids
        assert len(overlap) > 0, "Parsers should discover at least some common datasets"
        assert len(drs_ids) > 0
        assert len(complete_ids) > 0

    def test_drs_then_complete_produces_same_core_metadata(self, adapter_config, adapter_local_catalogs):
        """DRS and complete parsers produce the same values for core DRS fields."""
        adapter = adapter_config.adapter_cls()
        drs_catalog = adapter_local_catalogs["drs"]
        complete_catalog = adapter_local_catalogs["complete"]

        # Only check fields present in both catalogs (e.g. CMIP6 has no mip_era)
        candidate_fields = [*adapter.dataset_id_metadata, "version", "mip_era"]
        drs_fields = [
            f for f in candidate_fields if f in drs_catalog.columns and f in complete_catalog.columns
        ]
        common_ids = set(drs_catalog["instance_id"]) & set(complete_catalog["instance_id"])
        assert len(common_ids) > 0, "Need at least one common dataset"

        for instance_id in common_ids:
            drs_row = drs_catalog[drs_catalog["instance_id"] == instance_id].iloc[0]
            complete_row = complete_catalog[complete_catalog["instance_id"] == instance_id].iloc[0]
            for field in drs_fields:
                assert str(drs_row[field]) == str(complete_row[field]), (
                    f"Field '{field}' differs for {instance_id}: "
                    f"DRS={drs_row[field]!r} vs complete={complete_row[field]!r}"
                )

    def test_validate_catalog(self, adapter_config, adapter_local_catalogs):
        """validate_data_catalog() passes on a complete-parsed catalog."""
        adapter = adapter_config.adapter_cls()
        catalog = adapter_local_catalogs["complete"]
        validated = adapter.validate_data_catalog(catalog)
        assert len(validated) > 0


class TestRoundTripAndFinalisation:
    """Round-trip and finalisation tests, parameterised over adapter types."""

    @pytest.mark.parametrize("parser", ["complete", "drs"])
    def test_round_trip(self, parser, config, adapter_config, adapter_local_catalogs):
        """Ingest, register, and reload a dataset - verify DataFrame equality."""
        setattr(config, adapter_config.parser_config_attr, parser)
        catalog = adapter_local_catalogs[parser]

        with Database.from_config(config, run_migrations=True) as database:
            adapter = adapter_config.adapter_cls()
            with database.session.begin():
                for instance_id, data_catalog_dataset in catalog.groupby(adapter.slug_column):
                    adapter.register_dataset(database, data_catalog_dataset)

            local_data_catalog = (
                catalog.drop(columns=adapter_config.non_roundtrip_columns, errors="ignore")
                .sort_values(["instance_id", "start_time"])
                .reset_index(drop=True)
            )

            db_data_catalog = (
                adapter.load_catalog(database)
                .drop(columns=adapter_config.non_roundtrip_columns, errors="ignore")
                .sort_values(["instance_id", "start_time"])
                .reset_index(drop=True)
            )

            # Normalize null values for consistent comparison
            with pd.option_context("future.no_silent_downcasting", True):
                local_normalized = local_data_catalog.fillna(np.nan).infer_objects()
                db_normalized = db_data_catalog.fillna(np.nan).infer_objects()

            pd.testing.assert_frame_equal(
                local_normalized,
                db_normalized,
                check_like=True,
            )

    def test_finalise_datasets(self, config, adapter_config, adapter_data_dir):
        """DRS ingest -> register -> finalise -> verify metadata populated."""
        setattr(config, adapter_config.parser_config_attr, "drs")

        with Database.from_config(config, run_migrations=True) as database:
            adapter = adapter_config.adapter_cls(config=config)
            drs_catalog = adapter.find_local_datasets(adapter_data_dir)
            assert (~drs_catalog["finalised"]).all(), "DRS parser should produce unfinalised datasets"

            with database.session.begin():
                for _instance_id, data_catalog_dataset in drs_catalog.groupby(adapter.slug_column):
                    adapter.register_dataset(database, data_catalog_dataset)

            db_catalog = adapter.load_catalog(database)
            assert not db_catalog["finalised"].any(), "DB catalog should have unfinalised datasets"

            target_instance = db_catalog["instance_id"].iloc[0]
            subset = db_catalog[db_catalog["instance_id"] == target_instance].copy()

            result = adapter.finalise_datasets(database, subset)

            assert result["finalised"].all(), "Finalised datasets should have finalised=True"
            assert result["source_id"].notna().all()
            assert result["experiment_id"].notna().all()

    def test_handles_db_exception_gracefully(self, config, adapter_config, adapter_data_dir):
        """Database exceptions during persist are caught and DataFrame rolled back."""
        setattr(config, adapter_config.parser_config_attr, "drs")

        with Database.from_config(config, run_migrations=True) as database:
            adapter = adapter_config.adapter_cls(config=config)
            drs_catalog = adapter.find_local_datasets(adapter_data_dir)

            with database.session.begin():
                for _, group in drs_catalog.groupby(adapter.slug_column):
                    adapter.register_dataset(database, group)

            db_catalog = adapter.load_catalog(database)
            target = db_catalog["instance_id"].iloc[0]
            subset = db_catalog[db_catalog["instance_id"] == target].copy()
            subset["finalised"] = True

            def exploding_begin():
                raise RuntimeError("simulated DB failure")

            with patch.object(database.session, "begin", side_effect=exploding_begin):
                adapter._persist_finalised_metadata(database, subset, subset.index)

            assert not subset["finalised"].any(), (
                "DataFrame should be rolled back to finalised=False when DB persist fails"
            )
