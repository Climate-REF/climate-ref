import datetime
import json
from pathlib import Path

import pytest
from sqlalchemy import select

from climate_ref.datasets.cmip6 import CMIP6DatasetAdapter
from climate_ref.models import Dataset
from climate_ref.models.dataset import CMIP6Dataset, DatasetFile


def test_ingest_help(invoke_cli):
    result = invoke_cli(["datasets", "ingest", "--help"])

    assert "Ingest a directory of dataset" in result.stdout


class TestDatasetsList:
    def test_list(self, db_seeded, invoke_cli):
        result = invoke_cli(["datasets", "list"])
        assert "experi…" in result.stdout

    def test_list_json_is_untruncated(self, db_seeded, invoke_cli):
        import json

        result = invoke_cli(["datasets", "list", "--format", "json"])

        payload = json.loads(result.stdout)
        assert isinstance(payload, list)
        assert payload
        # The table output truncates wide columns (e.g. "experi…"); JSON must not.
        assert "experiment_id" in payload[0]
        assert "experi…" not in result.stdout

    def test_list_limit(self, db_seeded, invoke_cli):
        result = invoke_cli(["datasets", "list", "--limit", "1", "--column", "variable_id"])
        assert len(result.stdout.strip().split("\n")) == 3  # header + spacer + 1 row

    def test_list_column(self, db_seeded, invoke_cli):
        result = invoke_cli(["datasets", "list", "--column", "variable_id"])
        assert "variable_id" in result.stdout
        assert "grid" not in result.stdout

    def test_list_column_invalid(self, db_seeded, invoke_cli):
        invoke_cli(["datasets", "list", "--column", "wrong"], expected_exit_code=1)

    def test_list_column_base_field(self, db_seeded, invoke_cli):
        """A base field (not a facet) is selectable, e.g. ``slug``."""
        existing = db_seeded.session.query(CMIP6Dataset).first()
        result = invoke_cli(["datasets", "list", "--column", "slug"])
        assert existing.slug in result.stdout

    def test_list_column_on_empty_result(self, db_seeded, invoke_cli):
        """``--column`` on an empty result set exits 0 rather than erroring on a columnless frame."""
        result = invoke_cli(
            ["datasets", "list", "--dataset-filter", "source_id=DOES-NOT-EXIST", "--column", "slug"]
        )
        assert result.exit_code == 0

    def test_list_dataset_filter(self, db_seeded, invoke_cli):
        result = invoke_cli(
            ["datasets", "list", "--dataset-filter", "variable_id=tas", "--column", "variable_id"]
        )
        # All rows should contain "tas", none should contain other variables
        lines = result.stdout.strip().split("\n")
        # Skip header and separator lines
        data_lines = [line.strip() for line in lines[2:] if line.strip()]
        assert all("tas" in line for line in data_lines)

    def test_list_dataset_filter_multiple_values(self, db_seeded, invoke_cli):
        result = invoke_cli(
            [
                "datasets",
                "list",
                "--dataset-filter",
                "variable_id=tas",
                "--dataset-filter",
                "variable_id=pr",
                "--column",
                "variable_id",
            ]
        )
        lines = result.stdout.strip().split("\n")
        data_lines = [line.strip() for line in lines[2:] if line.strip()]
        assert all("tas" in line or "pr" in line for line in data_lines)

    def test_list_dataset_filter_invalid_format(self, db_seeded, invoke_cli):
        invoke_cli(
            ["datasets", "list", "--dataset-filter", "bad_filter"],
            expected_exit_code=2,
        )

    def test_list_dataset_filter_invalid_facet(self, db_seeded, invoke_cli):
        invoke_cli(
            ["datasets", "list", "--dataset-filter", "nonexistent_facet=value"],
            expected_exit_code=1,
        )

    def test_list_only_latest_version(self, db_seeded, invoke_cli):
        """``datasets list`` shows only the latest version of each dataset (v10 wins over v2)."""
        existing = db_seeded.session.query(CMIP6Dataset).first()
        base = {
            "dataset_type": existing.dataset_type,
            "activity_id": existing.activity_id,
            "experiment_id": existing.experiment_id,
            "institution_id": existing.institution_id,
            "source_id": "LATEST-TEST",
            "member_id": existing.member_id,
            "table_id": existing.table_id,
            "variable_id": existing.variable_id,
            "grid_label": existing.grid_label,
            "variant_label": existing.member_id,
        }
        for version in ("v2", "v10"):
            slug = f"latest.test.{version}"
            db_seeded.session.add(
                CMIP6Dataset(slug=slug, instance_id=slug, version=version, finalised=True, **base)
            )
        db_seeded.session.commit()

        result = invoke_cli(
            ["datasets", "list", "--dataset-filter", "source_id=LATEST-TEST", "--column", "version"]
        )
        assert "v10" in result.stdout
        assert "v2" not in result.stdout

    def test_list_limit_warns_when_truncated(self, db_seeded, invoke_cli):
        """A ``--limit`` smaller than the match count warns, matching the other list commands."""
        total = db_seeded.session.query(CMIP6Dataset).count()
        assert total > 1, "need multiple datasets for the warning to trigger"
        result = invoke_cli(["datasets", "list", "--limit", "1"])
        assert f"of {total} filtered results" in result.stderr

    def test_list_include_files_limit_bounds_files_not_datasets(self, db_seeded, invoke_cli):
        """``--limit`` bounds *files* (not datasets) when ``--include-files`` is set.

        Add a second file to one of the seeded datasets so at least one dataset has >1 file,
        then confirm ``--limit`` set to fewer than the total file count still returns that many file rows
        (i.e. the limit was not spent solely on datasets).
        """
        dataset = db_seeded.session.query(CMIP6Dataset).first()
        db_seeded.session.add(
            DatasetFile(
                dataset_id=dataset.id,
                path="extra-file-for-limit-test.nc",
                start_time="2000-01-01",
                end_time="2000-12-31",
            )
        )
        db_seeded.session.commit()

        total_files = db_seeded.session.query(DatasetFile).count()
        assert total_files >= 2, "need at least 2 files in the DB for this test to be meaningful"

        result = invoke_cli(["datasets", "list", "--include-files", "--limit", "1", "--column", "path"])
        data_lines = [line for line in result.stdout.strip().split("\n")[2:] if line.strip()]
        assert len(data_lines) == 1

    def test_list_excludes_retracted_by_default(self, db_seeded, invoke_cli):
        """Default ``datasets list`` matches solve-time eligibility: a retracted dataset is hidden."""
        existing = db_seeded.session.query(CMIP6Dataset).first()
        slug = existing.slug
        existing.retracted_at = datetime.datetime.now(datetime.UTC).replace(tzinfo=None)
        db_seeded.session.commit()

        result = invoke_cli(["datasets", "list", "--column", "slug"])
        assert slug not in result.stdout

    def test_list_include_retracted_does_not_mask_live_row(self, db_seeded, invoke_cli):
        """``--include-retracted`` must show both the retracted row and the live row sharing its partition."""
        existing = db_seeded.session.query(CMIP6Dataset).first()
        base = {
            "dataset_type": existing.dataset_type,
            "activity_id": existing.activity_id,
            "experiment_id": existing.experiment_id,
            "institution_id": existing.institution_id,
            "source_id": "RETRACT-MASK-TEST",
            "member_id": existing.member_id,
            "table_id": existing.table_id,
            "variable_id": existing.variable_id,
            "grid_label": existing.grid_label,
            "variant_label": existing.member_id,
        }
        live = CMIP6Dataset(
            slug="retract.mask.v2", instance_id="retract.mask.v2", version="v2", finalised=True, **base
        )
        retracted = CMIP6Dataset(
            slug="retract.mask.v10", instance_id="retract.mask.v10", version="v10", finalised=True, **base
        )
        db_seeded.session.add_all([live, retracted])
        db_seeded.session.commit()
        retracted.retracted_at = datetime.datetime.now(datetime.UTC).replace(tzinfo=None)
        db_seeded.session.commit()

        result = invoke_cli(
            [
                "datasets",
                "list",
                "--include-retracted",
                "--dataset-filter",
                "source_id=RETRACT-MASK-TEST",
                "--column",
                "version",
            ]
        )
        assert "v2" in result.stdout
        assert "v10" in result.stdout

    def test_list_retracted_at_column_present(self, db_seeded, invoke_cli):
        """The ``retracted_at`` column is present in the output frame regardless of
        ``--include-retracted``, so retraction state is visible whenever a retracted row does show."""
        result = invoke_cli(["datasets", "list", "--format", "json"])
        payload = json.loads(result.stdout)
        assert "retracted_at" in payload[0]

        result = invoke_cli(["datasets", "list", "--include-retracted", "--format", "json"])
        payload = json.loads(result.stdout)
        assert "retracted_at" in payload[0]


class TestDatasetsStats:
    def test_stats_basic(self, db_seeded, invoke_cli):
        result = invoke_cli(["datasets", "stats"])
        # db_seeded has CMIP6 datasets
        assert "cmip6" in result.stdout
        assert "dataset_type" in result.stdout
        assert "datasets" in result.stdout
        assert "files" in result.stdout
        assert "finalised" in result.stdout
        assert "unfinalised" in result.stdout

    def test_stats_no_data(self, db, invoke_cli):
        result = invoke_cli(["datasets", "stats"])
        assert "No datasets found." in result.stdout

    def test_stats_filter_by_source_type(self, db_seeded, invoke_cli):
        result = invoke_cli(["datasets", "stats", "--source-type", "cmip6"])
        assert "cmip6" in result.stdout

    def test_stats_filter_no_results(self, db_seeded, invoke_cli):
        result = invoke_cli(["datasets", "stats", "--source-type", "pmp-climatology"])
        assert "No datasets found." in result.stdout

    def test_stats_group_by(self, db_seeded, invoke_cli):
        result = invoke_cli(["datasets", "stats", "--source-type", "cmip6", "--group-by", "source_id"])
        assert "source_id" in result.stdout
        # db_seeded has ACCESS-ESM1-5 datasets
        assert "ACCESS-ESM1-5" in result.stdout

    def test_stats_group_by_variable_id(self, db_seeded, invoke_cli):
        result = invoke_cli(["datasets", "stats", "--source-type", "cmip6", "--group-by", "variable_id"])
        assert "variable_id" in result.stdout

    def test_stats_group_by_requires_source_type(self, db_seeded, invoke_cli):
        result = invoke_cli(["datasets", "stats", "--group-by", "source_id"], expected_exit_code=1)
        assert "--group-by requires --source-type" in result.stderr

    def test_stats_group_by_invalid_column(self, db_seeded, invoke_cli):
        result = invoke_cli(
            ["datasets", "stats", "--source-type", "cmip6", "--group-by", "experiment_id"],
            expected_exit_code=1,
        )
        assert "Invalid --group-by value" in result.stderr
        assert "source_id, variable_id" in result.stderr


class TestDatasetsListColumns:
    def test_list_columns(self, db_seeded, invoke_cli):
        result = invoke_cli(["datasets", "list-columns"])
        assert result.stdout.strip() == "\n".join(
            sorted(CMIP6DatasetAdapter().load_catalog(db_seeded, include_files=False).columns.to_list())
        )

    def test_list_include_files(self, db_seeded, invoke_cli):
        result = invoke_cli(["datasets", "list-columns", "--include-files"])
        assert result.stdout.strip() == "\n".join(
            sorted(CMIP6DatasetAdapter().load_catalog(db_seeded, include_files=True).columns.to_list())
        )
        assert "start_time" in result.stdout


class TestDatasetsRetract:
    def test_retract_marks_dataset_retracted(self, db_seeded, invoke_cli):
        existing = db_seeded.session.query(CMIP6Dataset).first()
        slug = existing.slug

        result = invoke_cli(["datasets", "retract", slug])
        assert f"Retracted dataset {slug!r}" in result.stdout

        db_seeded.session.expire_all()
        refreshed = db_seeded.session.query(CMIP6Dataset).filter_by(slug=slug).one()
        assert refreshed.retracted_at is not None

    def test_retract_missing_slug_exits_1(self, db_seeded, invoke_cli):
        result = invoke_cli(["datasets", "retract", "no-such-slug"], expected_exit_code=1)
        assert "No dataset found with slug" in result.stderr

    def test_retract_is_idempotent(self, db_seeded, invoke_cli):
        """Retracting an already-retracted dataset is a no-op, not an error."""
        existing = db_seeded.session.query(CMIP6Dataset).first()
        slug = existing.slug

        invoke_cli(["datasets", "retract", slug])
        db_seeded.session.expire_all()
        first_retracted_at = db_seeded.session.query(CMIP6Dataset).filter_by(slug=slug).one().retracted_at

        result = invoke_cli(["datasets", "retract", slug])
        assert result.exit_code == 0
        assert "already retracted" in result.stdout

        db_seeded.session.expire_all()
        second_retracted_at = db_seeded.session.query(CMIP6Dataset).filter_by(slug=slug).one().retracted_at
        assert first_retracted_at == second_retracted_at

    def test_unretract_restores_solve_eligibility(self, db_seeded, invoke_cli):
        existing = db_seeded.session.query(CMIP6Dataset).first()
        slug = existing.slug

        invoke_cli(["datasets", "retract", slug])
        result = invoke_cli(["datasets", "unretract", slug])
        assert f"Unretracted dataset {slug!r}" in result.stdout

        db_seeded.session.expire_all()
        refreshed = db_seeded.session.query(CMIP6Dataset).filter_by(slug=slug).one()
        assert refreshed.retracted_at is None

    def test_unretract_not_retracted_is_noop(self, db_seeded, invoke_cli):
        existing = db_seeded.session.query(CMIP6Dataset).first()
        slug = existing.slug

        result = invoke_cli(["datasets", "unretract", slug])
        assert result.exit_code == 0
        assert "is not retracted" in result.stdout

    def test_unretract_missing_slug_exits_1(self, db_seeded, invoke_cli):
        result = invoke_cli(["datasets", "unretract", "no-such-slug"], expected_exit_code=1)
        assert "No dataset found with slug" in result.stderr

    def test_list_surfaces_retraction_state(self, db_seeded, invoke_cli):
        """``ref datasets list --include-retracted`` includes retracted rows, flagged via ``retracted_at``."""
        existing = db_seeded.session.query(CMIP6Dataset).first()
        slug = existing.slug
        invoke_cli(["datasets", "retract", slug])

        result = invoke_cli(
            [
                "datasets",
                "list",
                "--include-retracted",
                "--format",
                "json",
                "--column",
                "slug",
                "--column",
                "retracted_at",
            ]
        )
        payload = json.loads(result.stdout)

        retracted_row = next(r for r in payload if r["slug"] == slug)
        assert retracted_row["retracted_at"] is not None

        other_rows = [r for r in payload if r["slug"] != slug]
        assert other_rows
        assert all(r["retracted_at"] is None for r in other_rows)


class TestIngest:
    data_dir = Path("CMIP6") / "ScenarioMIP" / "CSIRO" / "ACCESS-ESM1-5" / "ssp126" / "r1i1p1f1"

    def test_ingest(self, sample_data_dir, db, invoke_cli):
        invoke_cli(["datasets", "ingest", str(sample_data_dir / self.data_dir), "--source-type", "cmip6"])

        expected_dataset_count = 6
        assert db.session.query(Dataset).count() == expected_dataset_count
        assert db.session.query(CMIP6Dataset).count() == expected_dataset_count
        assert db.session.query(DatasetFile).count() == expected_dataset_count

    def test_ingest_multiple(self, sample_data_dir, db, invoke_cli):
        invoke_cli(
            [
                "datasets",
                "ingest",
                str(sample_data_dir / "CMIP6/ScenarioMIP"),
                str(sample_data_dir / "CMIP6/CMIP"),
                "--source-type",
                "cmip6",
            ]
        )
        query_paths = select(DatasetFile.path)
        dataset_file_paths = db.session.scalars(query_paths).all()
        assert len(dataset_file_paths)

        for dataset_file in dataset_file_paths:
            assert Path(dataset_file).exists()
            assert dataset_file.startswith(str(sample_data_dir / "CMIP6")) or dataset_file.startswith(
                str(sample_data_dir / "CMIP6/ScenarioMIP")
            )

    def test_ingest_and_solve(self, sample_data_dir, db, invoke_cli):
        result = invoke_cli(
            [
                "--log-level",
                "info",
                "datasets",
                "ingest",
                str(sample_data_dir / self.data_dir),
                "--source-type",
                "cmip6",
                "--solve",
                "--dry-run",
            ],
        )
        assert "Solving for diagnostics that require recalculation." in result.stderr

    def test_ingest_multiple_times(self, sample_data_dir, db, invoke_cli):
        invoke_cli(
            [
                "datasets",
                "ingest",
                str(sample_data_dir / self.data_dir / "Amon" / "tas"),
                "--source-type",
                "cmip6",
            ],
        )

        assert db.session.query(Dataset).count() == 1
        assert db.session.query(DatasetFile).count() == 1

        invoke_cli(
            [
                "datasets",
                "ingest",
                str(sample_data_dir / self.data_dir / "Amon" / "tas"),
                "--source-type",
                "cmip6",
            ],
        )

        assert db.session.query(Dataset).count() == 1

        invoke_cli(
            [
                "datasets",
                "ingest",
                str(sample_data_dir / self.data_dir / "Amon" / "rsut"),
                "--source-type",
                "cmip6",
            ],
        )

        assert db.session.query(Dataset).count() == 2

    def test_ingest_missing(self, sample_data_dir, db, invoke_cli):
        result = invoke_cli(
            [
                "datasets",
                "ingest",
                str(sample_data_dir / "missing"),
                "--source-type",
                "cmip6",
            ],
            expected_exit_code=1,
        )

        # Logs the missing directory and exits non-zero so cron/k8s see the failure.
        assert f"File or directory {sample_data_dir / 'missing'} does not exist" in result.stderr
        assert "Ingestion failed for 1 of 1 input(s)" in result.stderr

    def test_ingest_partial_failure_exits_nonzero(self, sample_data_dir, db, invoke_cli):
        # One valid dir, one missing dir: ingestion still runs for the valid dir,
        # but the CLI exits non-zero so the failure is observable in cron/k8s.
        result = invoke_cli(
            [
                "datasets",
                "ingest",
                str(sample_data_dir / self.data_dir),
                str(sample_data_dir / "missing"),
                "--source-type",
                "cmip6",
            ],
            expected_exit_code=1,
        )

        expected_dataset_count = 6
        assert db.session.query(Dataset).count() == expected_dataset_count
        assert "Ingestion failed for 1 of 2 input(s)" in result.stderr

    def test_ingest_chunk_size(self, sample_data_dir, db, invoke_cli):
        invoke_cli(
            [
                "datasets",
                "ingest",
                str(sample_data_dir / self.data_dir),
                "--source-type",
                "cmip6",
                "--chunk-size",
                "2",
            ]
        )

        expected_dataset_count = 6
        assert db.session.query(Dataset).count() == expected_dataset_count
        assert db.session.query(CMIP6Dataset).count() == expected_dataset_count
        assert db.session.query(DatasetFile).count() == expected_dataset_count

    def test_ingest_chunk_size_dry_run(self, sample_data_dir, db, invoke_cli):
        invoke_cli(
            [
                "datasets",
                "ingest",
                str(sample_data_dir / self.data_dir),
                "--source-type",
                "cmip6",
                "--chunk-size",
                "2",
                "--dry-run",
            ]
        )

        assert db.session.query(Dataset).count() == 0

    def test_ingest_chunk_size_zero_rejected(self, sample_data_dir, db, invoke_cli):
        result = invoke_cli(
            [
                "datasets",
                "ingest",
                str(sample_data_dir / self.data_dir),
                "--source-type",
                "cmip6",
                "--chunk-size",
                "0",
            ],
            expected_exit_code=2,
        )
        assert "chunk_size must be >= 1" in result.stderr

    def test_ingest_chunk_size_negative_rejected(self, sample_data_dir, db, invoke_cli):
        result = invoke_cli(
            [
                "datasets",
                "ingest",
                str(sample_data_dir / self.data_dir),
                "--source-type",
                "cmip6",
                "--chunk-size",
                "-3",
            ],
            expected_exit_code=2,
        )
        assert "chunk_size must be >= 1" in result.stderr

    def test_ingest_chunk_size_unsupported_adapter_falls_back(self, sample_data_dir, db, invoke_cli):
        """``--chunk-size`` on an adapter without ``iter_local_datasets`` warns and falls back."""
        obs_dir = sample_data_dir / "obs4MIPs"
        result = invoke_cli(
            [
                "datasets",
                "ingest",
                str(obs_dir),
                "--source-type",
                "obs4mips",
                "--chunk-size",
                "5",
                "--dry-run",
            ]
        )
        assert "does not support streaming ingest" in result.stderr

    def test_ingest_dryrun(self, sample_data_dir, db, invoke_cli):
        invoke_cli(
            [
                "datasets",
                "ingest",
                str(sample_data_dir / "CMIP6"),
                "--source-type",
                "cmip6",
                "--dry-run",
            ]
        )

        # Check that no data was loaded
        assert db.session.query(Dataset).count() == 0


class TestFetchSampleData:
    def test_fetch_defaults(self, mocker, invoke_cli):
        mock_fetch = mocker.patch("climate_ref.testing.fetch_sample_data")
        invoke_cli(
            [
                "datasets",
                "fetch-sample-data",
            ]
        )

        mock_fetch.assert_called_once_with(force_cleanup=False, symlink=False)

    def test_fetch(self, mocker, invoke_cli):
        mock_fetch = mocker.patch("climate_ref.testing.fetch_sample_data")
        invoke_cli(
            [
                "datasets",
                "fetch-sample-data",
                "--force-cleanup",
                "--symlink",
            ]
        )

        mock_fetch.assert_called_once_with(force_cleanup=True, symlink=True)


@pytest.fixture(scope="function")
def mock_obs4ref(mocker):
    mock_data_registry = mocker.patch("climate_ref.cli.datasets.dataset_registry_manager")
    mock_fetch = mocker.patch("climate_ref.cli.datasets.fetch_all_files")

    return mock_data_registry, mock_fetch


class TestFetchObs4REFData:
    def test_fetch_defaults(self, mock_obs4ref, invoke_cli, tmp_path):
        mock_data_registry, mock_fetch = mock_obs4ref

        invoke_cli(["datasets", "fetch-data", "--registry", "obs4ref", "--output-directory", str(tmp_path)])

        mock_fetch.assert_called_once_with(
            mock_data_registry["obs4ref"], "obs4ref", tmp_path, symlink=False, verify=True
        )

    def test_fetch_without_output_directory(self, mock_obs4ref, invoke_cli, tmp_path):
        mock_data_registry, mock_fetch = mock_obs4ref

        invoke_cli(["datasets", "fetch-data", "--registry", "obs4ref"])

        mock_fetch.assert_called_once_with(
            mock_data_registry["obs4ref"], "obs4ref", None, symlink=False, verify=True
        )

    def test_fetch_no_verify(self, mock_obs4ref, invoke_cli, tmp_path):
        mock_data_registry, mock_fetch = mock_obs4ref

        invoke_cli(["datasets", "fetch-data", "--registry", "obs4ref", "--no-verify"])

        mock_fetch.assert_called_once_with(
            mock_data_registry["obs4ref"], "obs4ref", None, symlink=False, verify=False
        )

    def test_fetch_missing(self, mock_obs4ref, invoke_cli, tmp_path):
        mock_data_registry, _mock_fetch = mock_obs4ref
        mock_data_registry.__getitem__.side_effect = KeyError

        invoke_cli(["datasets", "fetch-data", "--registry", "missing"], expected_exit_code=1)

    def test_fetch_symlink(self, mock_obs4ref, invoke_cli, tmp_path):
        mock_data_registry, mock_fetch = mock_obs4ref
        invoke_cli(
            [
                "datasets",
                "fetch-data",
                "--registry",
                "obs4ref",
                "--output-directory",
                str(tmp_path),
                "--symlink",
            ]
        )

        mock_fetch.assert_called_once_with(
            mock_data_registry["obs4ref"], "obs4ref", tmp_path, symlink=True, verify=True
        )

    def test_fetch_force_cleanup(self, mock_obs4ref, invoke_cli, tmp_path):
        assert tmp_path.exists()

        invoke_cli(
            [
                "datasets",
                "fetch-data",
                "--registry",
                "obs4ref",
                "--output-directory",
                str(tmp_path),
                "--force-cleanup",
            ]
        )

        assert not tmp_path.exists()

    def test_fetch_force_cleanup_missing(self, mock_obs4ref, invoke_cli, tmp_path):
        invoke_cli(
            [
                "datasets",
                "fetch-data",
                "--registry",
                "obs4ref",
                "--output-directory",
                str(tmp_path / "missing"),
                "--force-cleanup",
            ]
        )
