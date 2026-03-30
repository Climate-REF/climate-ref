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

    def test_list_limit(self, db_seeded, invoke_cli):
        result = invoke_cli(["datasets", "list", "--limit", "1", "--column", "variable_id"])
        assert len(result.stdout.strip().split("\n")) == 3  # header + spacer + 1 row

    def test_list_column(self, db_seeded, invoke_cli):
        result = invoke_cli(["datasets", "list", "--column", "variable_id"])
        assert "variable_id" in result.stdout
        assert "grid" not in result.stdout

    def test_list_column_invalid(self, db_seeded, invoke_cli):
        invoke_cli(["datasets", "list", "--column", "wrong"], expected_exit_code=1)

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
        )

        # Continues past the missing directory
        assert f"File or directory {sample_data_dir / 'missing'} does not exist" in result.stderr

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
