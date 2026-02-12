"""
Tests for the solve_helpers module.
"""

from __future__ import annotations

import json

import pandas as pd
from climate_ref_example import provider as example_provider

from climate_ref.solve_helpers import (
    format_solve_results_json,
    format_solve_results_table,
    generate_catalog,
    load_solve_catalog,
    solve_results_for_regression,
    solve_to_results,
    write_catalog_parquet,
)
from climate_ref_core.datasets import SourceDatasetType


class TestGenerateCatalog:
    def test_generate_catalog_cmip6(self, sample_data_dir):
        catalog = generate_catalog("cmip6", [sample_data_dir / "CMIP6"])
        assert isinstance(catalog, pd.DataFrame)
        assert len(catalog) > 0
        assert "instance_id" in catalog.columns

    def test_generate_catalog_obs4mips(self, sample_data_dir):
        catalog = generate_catalog("obs4mips", [sample_data_dir / "obs4REF", sample_data_dir / "obs4MIPs"])
        assert isinstance(catalog, pd.DataFrame)
        assert len(catalog) > 0

    def test_generate_catalog_empty_dir(self, tmp_path):
        catalog = generate_catalog("cmip6", [tmp_path])
        assert isinstance(catalog, pd.DataFrame)
        assert len(catalog) == 0

    def test_generate_catalog_strip_prefix(self, sample_data_dir):
        prefix = str(sample_data_dir)
        catalog = generate_catalog("cmip6", [sample_data_dir / "CMIP6"], strip_path_prefix=prefix)
        assert len(catalog) > 0
        if "path" in catalog.columns:
            for path in catalog["path"]:
                assert not path.startswith(prefix)
                assert "{data_dir}" in path


class TestWriteAndLoadCatalog:
    def test_round_trip(self, tmp_path, sample_data_dir):
        catalog = generate_catalog("cmip6", [sample_data_dir / "CMIP6"])
        out_path = tmp_path / "test_catalog.parquet"
        write_catalog_parquet(catalog, out_path)

        assert out_path.exists()
        loaded = pd.read_parquet(out_path)
        # Parquet may coerce time columns to datetime64, so skip dtype check
        pd.testing.assert_frame_equal(catalog, loaded, check_dtype=False)

    def test_load_solve_catalog_missing_dir(self, tmp_path):
        result = load_solve_catalog(tmp_path / "nonexistent")
        assert result is None

    def test_load_solve_catalog_empty_dir(self, tmp_path):
        catalog_dir = tmp_path / "empty-catalog"
        catalog_dir.mkdir()
        result = load_solve_catalog(catalog_dir)
        assert result is None

    def test_load_solve_catalog_with_files(self, tmp_path, sample_data_dir):
        catalog_dir = tmp_path / "catalog"
        catalog_dir.mkdir()

        cmip6_catalog = generate_catalog("cmip6", [sample_data_dir / "CMIP6"])
        write_catalog_parquet(cmip6_catalog, catalog_dir / "cmip6_catalog.parquet")

        result = load_solve_catalog(catalog_dir)
        assert result is not None
        assert SourceDatasetType.CMIP6 in result
        # Parquet may coerce time columns to datetime64, so skip dtype check
        pd.testing.assert_frame_equal(result[SourceDatasetType.CMIP6], cmip6_catalog, check_dtype=False)


class TestSolveToResults:
    def test_solve_to_results_example_provider(self, esgf_data_catalog):
        results = solve_to_results(esgf_data_catalog, providers=[example_provider])
        assert isinstance(results, list)
        assert len(results) > 0

        for r in results:
            assert "provider" in r
            assert "diagnostic" in r
            assert "dataset_key" in r
            assert "selectors" in r
            assert "datasets" in r
            assert r["provider"] == "example"

    def test_solve_to_results_sorted(self, esgf_data_catalog):
        results = solve_to_results(esgf_data_catalog, providers=[example_provider])
        keys = [(r["provider"], r["diagnostic"], r["dataset_key"]) for r in results]
        assert keys == sorted(keys)

    def test_solve_to_results_datasets_sorted(self, esgf_data_catalog):
        results = solve_to_results(esgf_data_catalog, providers=[example_provider])
        for r in results:
            for instance_ids in r["datasets"].values():
                assert instance_ids == sorted(instance_ids)


class TestFormatSolveResultsTable:
    def test_empty_results(self):
        assert format_solve_results_table([]) == "No executions found."

    def test_format_contains_provider_and_diagnostic(self, esgf_data_catalog):
        results = solve_to_results(esgf_data_catalog, providers=[example_provider])
        table = format_solve_results_table(results)
        assert "example" in table
        assert "global-mean-timeseries" in table
        assert "Summary:" in table
        assert "executions" in table

    def test_format_contains_dataset_keys(self, esgf_data_catalog):
        results = solve_to_results(esgf_data_catalog, providers=[example_provider])
        table = format_solve_results_table(results)
        for r in results:
            assert r["dataset_key"] in table


class TestFormatSolveResultsJson:
    def test_valid_json(self, esgf_data_catalog):
        results = solve_to_results(esgf_data_catalog, providers=[example_provider])
        json_str = format_solve_results_json(results)
        parsed = json.loads(json_str)
        assert isinstance(parsed, list)
        assert len(parsed) == len(results)

    def test_json_has_expected_keys(self, esgf_data_catalog):
        results = solve_to_results(esgf_data_catalog, providers=[example_provider])
        parsed = json.loads(format_solve_results_json(results))
        for entry in parsed:
            assert "provider" in entry
            assert "diagnostic" in entry
            assert "dataset_key" in entry
            assert "selectors" in entry
            assert "datasets" in entry


class TestSolveResultsForRegression:
    def test_regression_format(self, esgf_data_catalog):
        results = solve_to_results(esgf_data_catalog, providers=[example_provider])
        regression = solve_results_for_regression(results)
        assert isinstance(regression, dict)

        for dataset_key, sources in regression.items():
            assert isinstance(dataset_key, str)
            assert isinstance(sources, dict)
            for source_type, instance_ids in sources.items():
                assert isinstance(source_type, str)
                assert isinstance(instance_ids, list)

    def test_regression_keys_match_results(self, esgf_data_catalog):
        results = solve_to_results(esgf_data_catalog, providers=[example_provider])
        regression = solve_results_for_regression(results)
        result_keys = {r["dataset_key"] for r in results}
        assert set(regression.keys()) == result_keys
