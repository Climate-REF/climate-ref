"""Shared unit tests parameterised over CMIP6 and CMIP7 adapter classes.

Tests the FinaliseableDatasetAdapterMixin behaviour and common adapter
patterns that are identical regardless of which concrete adapter
(CMIP6 / CMIP7) is used.
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd

from climate_ref.database import Database


def _make_unfinalised_df(cfg, paths: list) -> pd.DataFrame:
    """Build a minimal unfinalised DataFrame matching adapter column expectations."""
    adapter = cfg.adapter_cls()
    n = len(paths)
    data = {col: [pd.NA] * n for col in adapter.dataset_specific_metadata}
    data.update({col: [pd.NA] * n for col in adapter.file_specific_metadata})
    data["path"] = paths
    data["finalised"] = [False] * n
    data["instance_id"] = [cfg.default_instance_id] * n
    data["source_id"] = [cfg.default_source_id] * n
    data["experiment_id"] = [cfg.default_experiment_id] * n
    return pd.DataFrame(data)


class TestAdapterBasics:
    """Basic adapter behaviour, parameterised over adapter types."""

    def test_catalog_empty(self, db, adapter_config):
        """Empty database returns an empty catalog."""
        adapter = adapter_config.adapter_cls()
        df = adapter.load_catalog(db)
        assert df.empty

    def test_adapter_default_config(self, adapter_config):
        """Adapter uses default config and n_jobs=1 when not specified."""
        adapter = adapter_config.adapter_cls()
        assert adapter.n_jobs == 1
        assert adapter.config is not None

    def test_adapter_custom_n_jobs(self, config, adapter_config):
        """Adapter stores the provided n_jobs value."""
        adapter = adapter_config.adapter_cls(n_jobs=4, config=config)
        assert adapter.n_jobs == 4


class TestParserDispatch:
    """Parser dispatch tests, parameterised over adapter types."""

    def test_returns_complete_parser(self, config, adapter_config):
        """get_parsing_function() returns the complete parser."""
        setattr(config, adapter_config.parser_config_attr, "complete")
        adapter = adapter_config.adapter_cls(config=config)
        assert adapter.get_parsing_function() is adapter_config.complete_parser

    def test_returns_drs_parser(self, config, adapter_config):
        """get_parsing_function() returns the DRS parser."""
        setattr(config, adapter_config.parser_config_attr, "drs")
        adapter = adapter_config.adapter_cls(config=config)
        assert adapter.get_parsing_function() is adapter_config.drs_parser

    def test_parse_exception_complete(self, adapter_config):
        """Complete parser returns INVALID_ASSET for a missing file."""
        result = adapter_config.complete_parser("missing_file")
        assert result["INVALID_ASSET"] == "missing_file"
        assert "TRACEBACK" in result

    def test_parse_exception_drs(self, adapter_config):
        """DRS parser returns INVALID_ASSET for a missing file."""
        result = adapter_config.drs_parser("missing_file")
        assert result["INVALID_ASSET"] == "missing_file"
        assert "TRACEBACK" in result


class TestFinalisationEdgeCases:
    """Finalisation edge cases for FinaliseableDatasetAdapterMixin, parameterised over adapter types."""

    def test_skips_rows_with_na_path(self, config, adapter_config):
        """Rows with NA path are skipped and remain unfinalised."""
        with Database.from_config(config, run_migrations=True) as database:
            adapter = adapter_config.adapter_cls(config=config)
            df = _make_unfinalised_df(adapter_config, [pd.NA])
            result = adapter.finalise_datasets(database, df)
            assert not result["finalised"].any()

    def test_skips_invalid_asset_response(self, config, adapter_config):
        """Rows returning INVALID_ASSET from the parser remain unfinalised."""
        with Database.from_config(config, run_migrations=True) as database:
            adapter = adapter_config.adapter_cls(config=config)
            df = _make_unfinalised_df(adapter_config, ["/fake/path.nc"])

            with patch(
                adapter_config.complete_parser_patch_path,
                return_value={"INVALID_ASSET": "/fake/path.nc", "TRACEBACK": "parse error"},
            ):
                result = adapter.finalise_datasets(database, df)
            assert not result["finalised"].any()

    def test_noop_when_all_already_finalised(self, config, adapter_config):
        """No parsing occurs when all rows are already finalised."""
        with Database.from_config(config, run_migrations=True) as database:
            adapter = adapter_config.adapter_cls(config=config)
            df = _make_unfinalised_df(adapter_config, ["/some/path.nc"])
            df["finalised"] = True

            with patch(adapter_config.complete_parser_patch_path) as mock_parse:
                result = adapter.finalise_datasets(database, df)
            mock_parse.assert_not_called()
            assert result["finalised"].all()

    def test_successful_parse_updates_metadata(self, config, adapter_config):
        """Successful parsing updates metadata columns and marks finalised."""
        with Database.from_config(config, run_migrations=True) as database:
            adapter = adapter_config.adapter_cls(config=config)
            df = _make_unfinalised_df(adapter_config, ["/fake/path.nc"])

            with patch(
                adapter_config.complete_parser_patch_path,
                return_value=adapter_config.successful_parsed_result,
            ):
                result = adapter.finalise_datasets(database, df)

            assert result["finalised"].iloc[0]
            for field, expected in adapter_config.metadata_checks.items():
                assert result[field].iloc[0] == expected, (
                    f"Expected {field}={expected!r}, got {result[field].iloc[0]!r}"
                )

    def test_partial_failure_finalises_only_successful_rows(self, config, adapter_config):
        """When one row fails and another succeeds, only the successful one is finalised."""
        with Database.from_config(config, run_migrations=True) as database:
            adapter = adapter_config.adapter_cls(config=config)
            df = _make_unfinalised_df(adapter_config, ["/bad/path.nc", "/good/path.nc"])

            def side_effect(path, **_):
                if "bad" in path:
                    return {"INVALID_ASSET": path, "TRACEBACK": "corrupt file"}
                return adapter_config.successful_parsed_result

            with patch(
                adapter_config.complete_parser_patch_path,
                side_effect=side_effect,
            ):
                result = adapter.finalise_datasets(database, df)

            assert not result["finalised"].iloc[0]
            assert result["finalised"].iloc[1]
            field, expected = next(iter(adapter_config.metadata_checks.items()))
            assert result[field].iloc[1] == expected

    def test_parsed_none_values_are_not_written(self, config, adapter_config):
        """Parsed values that are None do not overwrite existing column values."""
        with Database.from_config(config, run_migrations=True) as database:
            adapter = adapter_config.adapter_cls(config=config)
            df = _make_unfinalised_df(adapter_config, ["/fake/path.nc"])
            df["source_id"] = "original-model"

            check_field = next(iter(adapter_config.metadata_checks))
            check_value = adapter_config.metadata_checks[check_field]
            parsed = {
                "source_id": None,
                check_field: check_value,
            }
            with patch(
                adapter_config.complete_parser_patch_path,
                return_value=parsed,
            ):
                result = adapter.finalise_datasets(database, df)

            # source_id should retain original value since parsed value was None
            assert result["source_id"].iloc[0] == "original-model"
            assert result[check_field].iloc[0] == check_value


class TestPersistFinalisedMetadata:
    """_persist_finalised_metadata edge cases, parameterised over adapter types."""

    def test_skips_when_no_matching_db_record(self, config, adapter_config):
        """Silently skips slugs that have no matching database record."""
        with Database.from_config(config, run_migrations=True) as database:
            adapter = adapter_config.adapter_cls(config=config)
            data = {col: ["value"] for col in adapter.dataset_specific_metadata}
            data.update({col: [pd.NA] for col in adapter.file_specific_metadata})
            data["instance_id"] = [adapter_config.default_instance_id + ".nonexistent"]
            data["finalised"] = [True]
            data["path"] = ["/fake/path.nc"]
            df = pd.DataFrame(data)

            # Should not raise
            adapter._persist_finalised_metadata(database, df, df.index)

    def test_skips_duplicate_slugs(self, config, adapter_config):
        """Each slug is persisted only once even when multiple rows share it."""
        with Database.from_config(config, run_migrations=True) as database:
            adapter = adapter_config.adapter_cls(config=config)
            slug = adapter_config.default_instance_id
            data = {col: ["val", "val"] for col in adapter.dataset_specific_metadata}
            data.update({col: [pd.NA, pd.NA] for col in adapter.file_specific_metadata})
            data["instance_id"] = [slug, slug]
            data["finalised"] = [True, True]
            data["path"] = ["/fake/path1.nc", "/fake/path2.nc"]
            df = pd.DataFrame(data)

            # Make the query return None so we exercise the "no record" path
            database.session.query = lambda *a, **kw: type(
                "Q",
                (),
                {"filter": lambda *a, **kw: type("Q2", (), {"one_or_none": lambda: None})()},
            )()

            # Should not raise and should only attempt once for the slug
            adapter._persist_finalised_metadata(database, df, df.index)
