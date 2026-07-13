import numpy as np
import pandas as pd
import pytest

from climate_ref.datasets.cmip6 import CMIP6DatasetAdapter, _apply_fixes
from climate_ref.datasets.utils import sort_data_catalog


class TestCMIP6Adapter:
    @pytest.mark.parametrize("cmip6_parser", ["complete", "drs"])
    def test_load_catalog(self, cmip6_parser, db_seeded, catalog_regression, sample_data_dir, config):
        config.cmip6_parser = cmip6_parser

        adapter = CMIP6DatasetAdapter(config=config)
        df = adapter.load_catalog(db_seeded)

        for k in adapter.dataset_specific_metadata + adapter.file_specific_metadata:
            assert k in df.columns

        catalog_regression(
            sort_data_catalog(df),
            basename=f"cmip6_catalog_db_{cmip6_parser}",
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

        pd.testing.assert_frame_equal(
            sort_data_catalog(data_catalog),
            sort_data_catalog(adapter.load_catalog(db_seeded)),
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

        # The limit is applied AFTER deduplicating to the latest version, so it bounds the number of
        # datasets actually returned. The target now has three versions in the DB; a limit equal to the
        # deduplicated dataset count must still return every dataset (a limit applied before dedup would
        # spend slots on the target's superseded versions and silently drop other datasets).
        n_datasets = len(adapter.load_catalog(db_seeded, include_files=False))
        limited = adapter.load_catalog(db_seeded, include_files=False, limit=n_datasets)
        assert len(limited) == n_datasets
        assert new_instance_id in limited.instance_id.tolist()


class TestCMIP6IterLocalDatasets:
    def test_streaming_matches_whole_tree(self, sample_data, sample_data_dir):
        """``iter_local_datasets`` must yield the same rows as ``find_local_datasets``."""
        adapter = CMIP6DatasetAdapter()
        cmip6_root = sample_data_dir / "CMIP6"

        whole = adapter.find_local_datasets(cmip6_root)
        streamed = pd.concat(list(adapter.iter_local_datasets(cmip6_root, chunk_size=5)))

        # The streaming path may interleave chunks differently, so normalise both.
        pd.testing.assert_frame_equal(
            sort_data_catalog(whole.reset_index(drop=True)),
            sort_data_catalog(streamed.reset_index(drop=True)),
        )

    def test_streaming_yields_nonempty_chunks(self, sample_data, sample_data_dir):
        adapter = CMIP6DatasetAdapter()
        chunks = list(adapter.iter_local_datasets(sample_data_dir / "CMIP6", chunk_size=3))
        assert chunks, "expected at least one chunk for the sample archive"
        for chunk in chunks:
            assert not chunk.empty
            assert "instance_id" in chunk.columns

    def test_streaming_skips_empty_enriched_chunk(self, monkeypatch, tmp_path):
        """Chunks whose post-enrichment DataFrame is empty are not yielded."""
        adapter = CMIP6DatasetAdapter()
        data_dir = tmp_path / "CMIP6"
        data_dir.mkdir()
        (data_dir / "test.nc").touch()

        empty_df = pd.DataFrame()

        def _fake_iter(**kwargs):
            yield empty_df

        monkeypatch.setattr("climate_ref.datasets.cmip6.iter_built_catalogs", _fake_iter)
        monkeypatch.setattr(adapter, "_enrich_parsed_catalog", lambda df: df)

        chunks = list(adapter.iter_local_datasets(data_dir, chunk_size=10))
        assert chunks == []


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
