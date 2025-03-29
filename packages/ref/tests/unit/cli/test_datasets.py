from pathlib import Path

import pytest

from cmip_ref.datasets.cmip6 import CMIP6DatasetAdapter
from cmip_ref.models import Dataset
from cmip_ref.models.dataset import CMIP6Dataset, CMIP6File
from cmip_ref.testing import SAMPLE_DATA_VERSION


def test_ingest_help(invoke_cli):
    result = invoke_cli(["datasets", "ingest", "--help"])

    assert "Ingest a dataset" in result.stdout


class TestDatasetsList:
    def test_list(self, db_seeded, invoke_cli):
        result = invoke_cli(["datasets", "list"])
        assert "experi…" in result.stdout

    def test_list_limit(self, db_seeded, invoke_cli):
        result = invoke_cli(["datasets", "list", "--limit", "1", "--column", "instance_id"])
        assert len(result.stdout.strip().split("\n")) == 3  # header + spacer + 1 row

    def test_list_column(self, db_seeded, invoke_cli):
        result = invoke_cli(["datasets", "list", "--column", "variable_id"])
        assert "variable_id" in result.stdout
        assert "grid" not in result.stdout

    def test_list_column_invalid(self, db_seeded, invoke_cli):
        invoke_cli(["datasets", "list", "--column", "wrong"], expected_exit_code=1)


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

        assert db.session.query(Dataset).count() == 5
        assert db.session.query(CMIP6Dataset).count() == 5
        assert db.session.query(CMIP6File).count() == 5

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
        assert "Solving for metrics that require recalculation." in result.stderr

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
        assert db.session.query(CMIP6File).count() == 1

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
        assert isinstance(result.exception, FileNotFoundError)
        assert result.exception.filename == sample_data_dir / "missing"

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
        mock_fetch = mocker.patch("cmip_ref.cli.datasets.fetch_sample_data")
        invoke_cli(
            [
                "datasets",
                "fetch-sample-data",
            ]
        )

        mock_fetch.assert_called_once_with(version=SAMPLE_DATA_VERSION, force_cleanup=False, symlink=False)

    def test_fetch(self, mocker, invoke_cli):
        mock_fetch = mocker.patch("cmip_ref.cli.datasets.fetch_sample_data")
        invoke_cli(
            [
                "datasets",
                "fetch-sample-data",
                "--version",
                "v0.1.0",
                "--force-cleanup",
                "--symlink",
            ]
        )

        mock_fetch.assert_called_once_with(version="v0.1.0", force_cleanup=True, symlink=True)


@pytest.fixture(scope="function")
def mock_obs4ref(mocker):
    mock_build_registry = mocker.patch("cmip_ref.cli.datasets.build_reference_data_registry")
    mock_fetch = mocker.patch("cmip_ref.cli.datasets.fetch_all_files")

    return mock_build_registry, mock_fetch


class TestFetchObs4REFData:
    def test_fetch_defaults(self, mock_obs4ref, invoke_cli, tmp_path):
        mock_build_registry, mock_fetch = mock_obs4ref

        invoke_cli(["datasets", "fetch-obs4ref-data", "--output-directory", str(tmp_path)])

        mock_fetch.assert_called_once_with(mock_build_registry(), tmp_path, symlink=False)

    def test_fetch_symlink(self, mock_obs4ref, invoke_cli, tmp_path):
        mock_build_registry, mock_fetch = mock_obs4ref
        invoke_cli(["datasets", "fetch-obs4ref-data", "--output-directory", str(tmp_path), "--symlink"])

        mock_fetch.assert_called_once_with(mock_build_registry(), tmp_path, symlink=True)

    def test_fetch_force_cleanup(self, mock_obs4ref, invoke_cli, tmp_path):
        assert tmp_path.exists()

        invoke_cli(["datasets", "fetch-obs4ref-data", "--output-directory", str(tmp_path), "--force-cleanup"])

        assert not tmp_path.exists()

    def test_fetch_force_cleanup_missing(self, mock_obs4ref, invoke_cli, tmp_path):
        invoke_cli(
            [
                "datasets",
                "fetch-obs4ref-data",
                "--output-directory",
                str(tmp_path / "missing"),
                "--force-cleanup",
            ]
        )
