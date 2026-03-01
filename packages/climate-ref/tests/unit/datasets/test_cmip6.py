import numpy as np
import pandas as pd
import pytest

from climate_ref.datasets.cmip6 import CMIP6DatasetAdapter, _apply_fixes


class TestCMIP6Adapter:
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
