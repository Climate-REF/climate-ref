"""
Re-useable fixtures etc. for tests that are shared across the whole project

See https://docs.pytest.org/en/7.1.x/reference/fixtures.html#conftest-py-sharing-fixtures-across-multiple-files
"""

import os
import re
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from click.testing import Result
from typer.testing import CliRunner

from cmip_ref import cli
from cmip_ref.config import Config, MetricsProviderConfig
from cmip_ref.datasets.cmip6 import CMIP6DatasetAdapter
from cmip_ref.datasets.obs4mips import Obs4MIPsDatasetAdapter
from cmip_ref.testing import TEST_DATA_DIR, fetch_sample_data
from cmip_ref_core.datasets import DatasetCollection, MetricDataset, SourceDatasetType
from cmip_ref_core.metrics import DataRequirement, Metric, MetricExecutionDefinition, MetricExecutionResult
from cmip_ref_core.providers import MetricsProvider

pytest_plugins = ("celery.contrib.pytest",)


@pytest.fixture(scope="session")
def tmp_path_session():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    if TEST_DATA_DIR is None:
        raise ValueError("Test data should exist when running tests")
    return TEST_DATA_DIR


@pytest.fixture(scope="session")
def sample_data_dir(test_data_dir) -> Path:
    return test_data_dir / "sample-data"


@pytest.fixture(autouse=True, scope="session")
def sample_data() -> None:
    # Downloads the sample data if it doesn't exist
    fetch_sample_data(force_cleanup=False, symlink=False)


@pytest.fixture(scope="session")
def cmip6_data_catalog(sample_data_dir) -> pd.DataFrame:
    adapter = CMIP6DatasetAdapter()
    return adapter.find_local_datasets(sample_data_dir / "CMIP6")


@pytest.fixture(scope="session")
def obs4mips_data_catalog(sample_data_dir) -> pd.DataFrame:
    adapter = Obs4MIPsDatasetAdapter()
    return pd.concat(
        [
            adapter.find_local_datasets(sample_data_dir / "obs4MIPs"),
            adapter.find_local_datasets(sample_data_dir / "obs4REF"),
        ]
    )


@pytest.fixture(autouse=True)
def config(tmp_path, monkeypatch, request) -> Config:
    # Optionally use the `REF_TEST_OUTPUT` env variable as the root output directory
    # This is useful in the CI to capture any results for later analysis
    root_output_dir = Path(os.environ.get("REF_TEST_OUTPUT", tmp_path / "cmip_ref"))
    # Each test gets its own directory (based on the test filename and the test name)
    # Sanitize the directory name to remove invalid characters
    dir_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", request.node.name)
    ref_config_dir = root_output_dir / request.module.__name__ / dir_name

    monkeypatch.setenv("REF_CONFIGURATION", str(ref_config_dir))

    # Uses the default configuration
    cfg = Config.load(ref_config_dir / "ref.toml")

    # Put the conda environments in a shared location
    # ROOT / .ref / software
    cfg.paths.software = Path(__file__).parent / ".ref" / "software"

    # Allow adding datasets from outside the tree for testing
    cfg.metric_providers = [MetricsProviderConfig(provider="cmip_ref_metrics_example")]

    cfg.save()

    return cfg


@pytest.fixture
def invoke_cli():
    """
    Invoke the CLI with the given arguments and verify the exit code
    """

    # We want to split stderr and stdout
    # stderr == logging
    # stdout == output from commands
    runner = CliRunner(mix_stderr=False)

    def _invoke_cli(args: list[str], expected_exit_code: int = 0, always_log: bool = False) -> Result:
        result = runner.invoke(
            app=cli.app,
            args=args,
        )

        if always_log or result.exit_code != expected_exit_code:
            print("## Command: ", " ".join(args))
            print("Exit code: ", result.exit_code)
            print("Command stdout")
            print(result.stdout)
            print("Command stderr")
            print(result.stderr)
            print("## Command end")

        if result.exit_code != expected_exit_code:
            if result.exception:
                raise result.exception
            raise ValueError(f"Expected exit code {expected_exit_code}, got {result.exit_code}")
        return result

    return _invoke_cli


class MockMetric(Metric):
    name = "mock"
    slug = "mock"

    # This runs on every dataset
    data_requirements = (DataRequirement(source_type=SourceDatasetType.CMIP6, filters=(), group_by=None),)

    def run(self, definition: MetricExecutionDefinition) -> MetricExecutionResult:
        # TODO: This doesn't write output.json, use build function?
        return MetricExecutionResult(
            output_bundle_filename=definition.output_directory / "output.json",
            metric_bundle_filename=definition.output_directory / "metric.json",
            successful=True,
            definition=definition,
        )


class FailedMetric(Metric):
    name = "failed"
    slug = "failed"

    data_requirements = (DataRequirement(source_type=SourceDatasetType.CMIP6, filters=(), group_by=None),)

    def run(self, definition: MetricExecutionDefinition) -> MetricExecutionResult:
        return MetricExecutionResult.build_from_failure(definition)


@pytest.fixture
def provider(tmp_path) -> MetricsProvider:
    provider = MetricsProvider("mock_provider", "v0.1.0")
    provider.register(MockMetric())
    provider.register(FailedMetric())

    return provider


@pytest.fixture
def mock_metric() -> MockMetric:
    return MockMetric()


@pytest.fixture
def definition_factory(tmp_path: Path, config):
    def _create_definition(
        *,
        metric_dataset: MetricDataset | None = None,
        cmip6: DatasetCollection | None = None,
        obs4mips: DatasetCollection | None = None,
    ) -> MetricExecutionDefinition:
        if metric_dataset is None:
            datasets = {}
            if cmip6:
                datasets[SourceDatasetType.CMIP6] = cmip6
            if obs4mips:
                datasets[SourceDatasetType.obs4MIPs] = obs4mips
            metric_dataset = MetricDataset(datasets)

        return MetricExecutionDefinition(
            dataset_key="key",
            metric_dataset=metric_dataset,
            root_directory=config.paths.scratch,
            output_directory=config.paths.scratch / "output_fragment",
        )

    return _create_definition


@pytest.fixture
def metric_definition(definition_factory, cmip6_data_catalog) -> MetricExecutionDefinition:
    selected_dataset = cmip6_data_catalog[
        cmip6_data_catalog["instance_id"].isin(
            {
                "CMIP6.ScenarioMIP.CSIRO.ACCESS-ESM1-5.ssp126.r1i1p1f1.Amon.tas.gn.v20210318",
                "CMIP6.ScenarioMIP.CSIRO.ACCESS-ESM1-5.ssp126.r1i1p1f1.fx.areacella.gn.v20210318",
            }
        )
    ]
    metric_dataset = MetricDataset(
        {
            SourceDatasetType.CMIP6: DatasetCollection(
                selected_dataset,
                "instance_id",
            )
        }
    )
    return definition_factory(metric_dataset=metric_dataset)
