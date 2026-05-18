"""
Integration tests for the idempotency of CMIP6 dataset ingestion.

Re-running ``ingest_datasets`` on an unchanged directory must be a complete
no-op: no UPDATED states, no "Updating file metadata" warnings.

These tests previously failed because ``DatasetFile.start_time`` /
``end_time`` are stored as ``str`` (via the @validates coercion on the model)
but ``register_dataset`` compared the DB-loaded ``str`` against a freshly
parsed ``cftime.datetime`` from ``find_local_datasets``.
The cross-type ``!=`` was always True, so every file was flagged as changed
on every run.
"""

from __future__ import annotations

import logging

from climate_ref.datasets import ingest_datasets
from climate_ref.datasets.cmip6 import CMIP6DatasetAdapter


class TestReingestIdempotency:
    """Re-ingesting the same directory must be a no-op."""

    def test_second_ingest_reports_no_updated_files(self, config, db, sample_data_dir):
        """The second ingest of the same directory updates zero files and zero datasets."""
        adapter = CMIP6DatasetAdapter(config=config)
        data_dir = sample_data_dir / "CMIP6"

        stats1 = ingest_datasets(adapter, data_dir, db)
        assert stats1.datasets_created > 0, "first ingest must create at least one dataset"
        assert stats1.files_added > 0, "first ingest must add at least one file"

        stats2 = ingest_datasets(adapter, data_dir, db)

        assert stats2.datasets_created == 0, (
            f"second ingest created {stats2.datasets_created} dataset(s); expected 0"
        )
        assert stats2.files_added == 0, f"second ingest added {stats2.files_added} file(s); expected 0"
        assert stats2.files_updated == 0, (
            f"second ingest reported {stats2.files_updated} file(s) as updated; expected 0. "
            "Root cause: file metadata comparison treats DB str vs freshly parsed "
            "cftime.datetime as always unequal."
        )
        assert stats2.datasets_updated == 0, (
            f"second ingest reported {stats2.datasets_updated} dataset(s) as updated; expected 0"
        )

    def test_second_ingest_is_quiet(self, config, db, sample_data_dir, caplog):
        """The second ingest must not emit 'Updating file metadata' warnings."""
        adapter = CMIP6DatasetAdapter(config=config)
        data_dir = sample_data_dir / "CMIP6"

        ingest_datasets(adapter, data_dir, db)

        caplog.clear()
        with caplog.at_level(logging.WARNING):
            ingest_datasets(adapter, data_dir, db)

        updating = [r for r in caplog.records if "Updating file metadata" in r.message]
        assert updating == [], (
            f"re-ingest emitted {len(updating)} spurious 'Updating file metadata' "
            "warning(s) on identical input:\n" + "\n".join(r.message for r in updating)
        )
