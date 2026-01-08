"""
Re-useable fixtures etc. for tests that are shared across the whole project

See https://docs.pytest.org/en/7.1.x/reference/fixtures.html#conftest-py-sharing-fixtures-across-multiple-files
"""

import os
import re
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pandas as pd
import pytest
from _pytest.logging import LogCaptureFixture
from attrs import define
from click.testing import Result
from loguru import logger
from typer.testing import CliRunner

from climate_ref import cli
from climate_ref.config import Config, DiagnosticProviderConfig
from climate_ref.datasets.cmip6 import CMIP6DatasetAdapter
from climate_ref.datasets.obs4mips import Obs4MIPsDatasetAdapter
from climate_ref.models import Execution
from climate_ref.solver import solve_executions
from climate_ref.testing import (
    ESGF_DATA_DIR,
    TEST_DATA_DIR,
    TestCaseRunner,
    fetch_sample_data,
    validate_result,
)
from climate_ref_core.datasets import DatasetCollection, ExecutionDatasetCollection, SourceDatasetType
from climate_ref_core.diagnostics import DataRequirement, Diagnostic, ExecutionDefinition, ExecutionResult
from climate_ref_core.exceptions import TestCaseError
from climate_ref_core.logging import add_log_handler, remove_log_handler
from climate_ref_core.providers import DiagnosticProvider

pytest_plugins = ("celery.contrib.pytest",)


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "docker: mark test requires docker to run")
    config.addinivalue_line("markers", "requires_esgf_data: mark test requires ESGF test data")


def pytest_addoption(parser):
    parser.addoption("--slow", action="store_true", help="include tests marked slow")
    parser.addoption("--no-docker", action="store_true", help="skip docker tests")


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--slow"):
        skip_slow = pytest.mark.skip(reason="need --slow option to run")
        for item in items:
            if item.get_closest_marker("slow"):
                item.add_marker(skip_slow)
    if config.getoption("--no-docker"):
        skip_docker = pytest.mark.skip(reason="--no-docker option provided")
        for item in items:
            if item.get_closest_marker("docker"):
                item.add_marker(skip_docker)


@pytest.fixture(scope="session")
def tmp_path_session():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def caplog(caplog: LogCaptureFixture) -> Iterator[LogCaptureFixture]:
    """
    Capture logs from the default logger
    """

    def filter_(record):
        return record["level"].no >= caplog.handler.level

    add_log_handler(sink=caplog.handler, level=0, format="{message}", filter=filter_)
    yield caplog
    remove_log_handler()


@pytest.fixture(autouse=True)
def cleanup_log_handlers(request: pytest.FixtureRequest) -> Iterator[None]:
    yield
    if hasattr(logger, "default_handler_id"):
        logger.warning("Logger handler not removed, removing it now")
        remove_log_handler()


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    if TEST_DATA_DIR is None:
        raise ValueError("Test data should exist when running tests")
    return TEST_DATA_DIR


@pytest.fixture(scope="session")
def sample_data_dir(test_data_dir) -> Path:
    return test_data_dir / "sample-data"


@pytest.fixture(scope="session")
def regression_data_dir(test_data_dir) -> Path:
    return test_data_dir / "regression"


@pytest.fixture(scope="session")
def esgf_test_data_dir() -> Path | None:
    """
    Path to ESGF test data directory.

    Returns None if the directory doesn't exist (data not yet fetched).
    """
    if ESGF_DATA_DIR is None or not ESGF_DATA_DIR.exists():
        return None
    return ESGF_DATA_DIR


@pytest.fixture(scope="session")
def esgf_solve_catalog(test_data_dir) -> dict[SourceDatasetType, pd.DataFrame] | None:
    """
    Load ESGF metadata catalog for solve tests.

    These catalogs contain ESGF dataset metadata without actual data files,
    used to test the solver without downloading data.

    Returns None if the catalog doesn't exist yet.
    """
    catalog_dir = test_data_dir / "esgf-catalog"
    if not catalog_dir.exists():
        return None

    result: dict[SourceDatasetType, pd.DataFrame] = {}

    cmip6_path = catalog_dir / "cmip6_catalog.parquet"
    if cmip6_path.exists():
        result[SourceDatasetType.CMIP6] = pd.read_parquet(cmip6_path)

    obs4mips_path = catalog_dir / "obs4mips_catalog.parquet"
    if obs4mips_path.exists():
        result[SourceDatasetType.obs4MIPs] = pd.read_parquet(obs4mips_path)

    return result if result else None


@pytest.fixture
def run_test_case(
    config: Config,
):
    """
    Fixture for running diagnostic test cases.

    Example
    -------
    ```python
    def test_diagnostic(run_test_case, my_diagnostic):
        result = run_test_case.run(my_diagnostic, "default")
        assert result.successful
    ```
    """

    runner = TestCaseRunner(config=config, datasets=None)

    # Wrap the runner to convert exceptions to pytest.skip
    class PytestTestCaseRunner:
        def run(self, diagnostic, test_case_name="default", output_dir=None):
            try:
                return runner.run(diagnostic, test_case_name, output_dir)
            except TestCaseError as e:
                pytest.skip(str(e))

                # Explicitly indicate that this code path does not return
                raise

    return PytestTestCaseRunner()


@pytest.fixture(autouse=True, scope="session")
def sample_data() -> None:
    if os.environ.get("REF_TEST_DATA_DIR"):
        logger.warning("Not fetching sample data. Using custom test data directory")
        return
    # Downloads the sample data if it doesn't exist
    logger.disable("climate_ref_core.dataset_registry")
    fetch_sample_data(force_cleanup=False, symlink=False)
    logger.enable("climate_ref_core.dataset_registry")


@pytest.fixture(scope="session")
def cmip6_data_catalog(sample_data_dir) -> pd.DataFrame:
    adapter = CMIP6DatasetAdapter()
    return adapter.find_local_datasets(sample_data_dir / "CMIP6")


@pytest.fixture(scope="session")
def obs4mips_data_catalog(sample_data_dir) -> pd.DataFrame:
    adapter = Obs4MIPsDatasetAdapter()
    obs4ref = adapter.find_local_datasets(sample_data_dir / "obs4REF")
    obs4mips = adapter.find_local_datasets(sample_data_dir / "obs4MIPs")
    return pd.concat([obs4ref, obs4mips], ignore_index=True)


@pytest.fixture(scope="session")
def data_catalog(cmip6_data_catalog, obs4mips_data_catalog):
    return {
        SourceDatasetType.CMIP6: cmip6_data_catalog,
        SourceDatasetType.obs4MIPs: obs4mips_data_catalog,
    }


@pytest.fixture(autouse=True)
def config(tmp_path, monkeypatch, request) -> Config:
    # Optionally use the `REF_TEST_OUTPUT` env variable as the root output directory
    # This is useful in the CI to capture any executions for later analysis
    root_output_dir = Path(os.environ.get("REF_TEST_OUTPUT", tmp_path / "climate_ref"))
    # Each test gets its own directory (based on the test filename and the test name)
    # Sanitize the directory name to remove invalid characters
    dir_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", request.node.name)
    ref_config_dir = root_output_dir / request.module.__name__ / dir_name

    # Read the configured location for the conda environments.
    software_path = Config.default().paths.software

    # Use the default configuration with a test-specific directory.
    monkeypatch.setenv("REF_CONFIGURATION", str(ref_config_dir))
    cfg = Config.default()

    # Use the configured location for the conda environments.
    cfg.paths.software = software_path

    # Allow adding datasets from outside the tree for testing
    cfg.diagnostic_providers = [DiagnosticProviderConfig(provider="climate_ref_example")]

    # Use the synchronous executor for testing to avoid spinning up multiple processes on every test
    cfg.executor.executor = "climate_ref.executor.SynchronousExecutor"

    cfg.save()

    return cfg


@pytest.fixture
def invoke_cli(monkeypatch):
    """
    Invoke the CLI with the given arguments and verify the exit code
    """

    # We want to split stderr and stdout
    # stderr == logging
    # stdout == output from commands
    runner = CliRunner(mix_stderr=False)

    def _invoke_cli(args: list[str], expected_exit_code: int = 0, always_log: bool = False) -> Result:
        # Disable color output for testing
        monkeypatch.setenv("NO_COLOR", "1")
        monkeypatch.setenv("COLUMNS", "200")

        result = runner.invoke(
            app=cli.app,
            args=args,
        )

        # Clean up the log handler the is added by the CLI
        if hasattr(logger, "default_handler_id"):
            remove_log_handler()

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


class MockDiagnostic(Diagnostic):
    name = "mock"
    slug = "mock"

    # This runs on every dataset
    data_requirements = (DataRequirement(source_type=SourceDatasetType.CMIP6, filters=(), group_by=None),)

    def run(self, definition: ExecutionDefinition) -> ExecutionResult:
        # TODO: This doesn't write output.json, use build function?
        return ExecutionResult(
            output_bundle_filename=definition.output_directory / "output.json",
            metric_bundle_filename=definition.output_directory / "diagnostic.json",
            successful=True,
            definition=definition,
        )


class FailedDiagnostic(Diagnostic):
    name = "failed"
    slug = "failed"

    data_requirements = (DataRequirement(source_type=SourceDatasetType.CMIP6, filters=(), group_by=None),)

    def run(self, definition: ExecutionDefinition) -> ExecutionResult:
        return ExecutionResult.build_from_failure(definition)


@pytest.fixture
def provider(tmp_path, config) -> DiagnosticProvider:
    provider = DiagnosticProvider("mock_provider", "v0.1.0")
    provider.register(MockDiagnostic())  # type: ignore
    provider.register(FailedDiagnostic())  # type: ignore
    provider.configure(config)

    return provider


@pytest.fixture
def mock_diagnostic(provider) -> MockDiagnostic:
    return cast(MockDiagnostic, provider.get("mock"))


@pytest.fixture
def definition_factory(tmp_path: Path, config):
    def _create_definition(
        *,
        diagnostic: Diagnostic,
        execution_dataset_collection: ExecutionDatasetCollection | None = None,
        cmip6: DatasetCollection | None = None,
        obs4mips: DatasetCollection | None = None,
        pmp_climatology: DatasetCollection | None = None,
    ) -> ExecutionDefinition:
        if execution_dataset_collection is None:
            datasets = {}
            if cmip6:
                datasets[SourceDatasetType.CMIP6] = cmip6
            if obs4mips:
                datasets[SourceDatasetType.obs4MIPs] = obs4mips
            if pmp_climatology:
                datasets[SourceDatasetType.PMPClimatology] = pmp_climatology
            execution_dataset_collection = ExecutionDatasetCollection(datasets)

        return ExecutionDefinition(
            diagnostic=diagnostic,
            key="key",
            datasets=execution_dataset_collection,
            root_directory=config.paths.scratch,  # type: ignore
            output_directory=config.paths.scratch / "output_fragment",
        )

    return _create_definition


@pytest.fixture
def metric_definition(definition_factory, cmip6_data_catalog, mock_diagnostic) -> ExecutionDefinition:
    selected_dataset = cmip6_data_catalog[
        cmip6_data_catalog["instance_id"].isin(
            {
                "CMIP6.ScenarioMIP.CSIRO.ACCESS-ESM1-5.ssp126.r1i1p1f1.Amon.tas.gn.v20210318",
                "CMIP6.ScenarioMIP.CSIRO.ACCESS-ESM1-5.ssp126.r1i1p1f1.fx.areacella.gn.v20210318",
            }
        )
    ]
    collection = ExecutionDatasetCollection(
        {
            SourceDatasetType.CMIP6: DatasetCollection(
                selected_dataset,
                "instance_id",
            )
        }
    )
    return definition_factory(diagnostic=mock_diagnostic, execution_dataset_collection=collection)


@define
class ExecutionRegression:
    """
    Copy the execution output from a diagnostic to the test-data directory

    These data can then be used to test the parsing of the CMEC bundles without
    having to run the entire diagnostic.

    The data are only copied if the `--force-regen` pytest option is set.
    """

    diagnostic: Diagnostic
    regression_data_dir: Path
    request: pytest.FixtureRequest
    replacements: dict[str, str]

    sanitised_file_globs: tuple[str, ...] = (
        "*.json",
        "*.txt",
        "*.yaml",
        "*.yml",
        "*.html",
        "*.xml",
    )

    def _replace_file(self, file: Path, replacements: dict[str, str]):
        with open(file, encoding="utf-8") as f:
            content = f.read()

            for key, value in replacements.items():
                content = content.replace(key, value)
        with open(file, "w") as f:
            f.write(content)

    def path(self, key: str) -> Path:
        return self.regression_data_dir / self.diagnostic.provider.slug / self.diagnostic.slug / key

    def replace_references(self, output_dir: Path, replacements: dict[str, str]):
        """
        Replace any references to local directories with a placeholder
        """
        for glob in self.sanitised_file_globs:
            for file in output_dir.rglob(glob):
                self._replace_file(file, replacements)

    def hydrate_output_directory(self, output_dir: Path, replacements):
        """
        Replace any references to the placeholder with the actual output directory
        """
        for glob in self.sanitised_file_globs:
            for file in output_dir.rglob(glob):
                self._replace_file(file, replacements)

    def check(self, key: str, output_directory: Path) -> None:
        if not self.request.config.getoption("force_regen"):
            logger.info("Not regenerating regression results")
            return
        # Replace any references to the output directory with a placeholder
        # This should make the regression data more portable and not tied to a specific machine
        self.replace_references(
            output_directory,
            {str(output_directory): "<OUTPUT_DIR>", **self.replacements},
        )

        logger.info(f"Regenerating regression output for {self.diagnostic.full_slug()}")
        output_dir = self.path(key)
        if output_dir.exists():
            shutil.rmtree(output_dir)

        shutil.copytree(output_directory, output_dir)


@pytest.fixture
def execution_regression(
    request,
    regression_data_dir,
    test_data_dir,
):
    def _regression(diagnostic: Diagnostic) -> ExecutionRegression:
        return ExecutionRegression(
            diagnostic=diagnostic,
            regression_data_dir=regression_data_dir,
            request=request,
            replacements={
                str(test_data_dir): "<TEST_DATA_DIR>",
            },
        )

    return _regression


@define
class DiagnosticValidator:
    config: Config
    diagnostic: Diagnostic
    data_catalog: dict[SourceDatasetType, pd.DataFrame]
    execution_regression: ExecutionRegression

    def get_definition(self) -> ExecutionDefinition:
        execution = next(
            solve_executions(
                data_catalog=self.data_catalog,
                diagnostic=self.diagnostic,
                provider=self.diagnostic.provider,
            )
        )
        return execution.build_execution_definition(output_root=self.config.paths.scratch)

    def get_regression_definition(self) -> ExecutionDefinition:
        """
        Get the regression definition for the diagnostic
        """
        definition = self.get_definition()

        # TODO: Should this be a helper function on ExecutionRegression?
        regression_output_dir = self.execution_regression.path(definition.key)

        # Copy the regression output directory to the execution output directory
        definition.output_directory.mkdir(parents=True, exist_ok=True)
        shutil.copytree(regression_output_dir, definition.output_directory, dirs_exist_ok=True)

        self.execution_regression.replace_references(
            definition.output_directory,
            {
                "<OUTPUT_DIR>": str(definition.output_directory),
                **{value: key for key, value in self.execution_regression.replacements.items()},
            },
        )

        return definition

    def execute(self, definition: ExecutionDefinition) -> None:
        """
        Validate the diagnostic by running it and checking the result
        """
        # Execute the diagnostic
        definition.output_directory.mkdir(parents=True, exist_ok=True)
        try:
            self.diagnostic.run(definition)
        finally:
            # Potentially save the result for regression testing
            self.execution_regression.check(key=definition.key, output_directory=definition.output_directory)

    def validate(self, definition: ExecutionDefinition) -> None:
        result = self.diagnostic.build_execution_result(definition)

        # Ensure the output log file exist
        # This isn't tracked in the regression data
        result.to_output_path("out.log").touch()
        validate_result(self.diagnostic, self.config, result)


@pytest.fixture
def diagnostic_validation(config, mocker, provider, data_catalog, execution_regression):
    mocker.patch.object(Execution, "execution_group")

    def _create_validator(diagnostic: Diagnostic) -> DiagnosticValidator:
        diagnostic.provider.configure(config)

        return DiagnosticValidator(
            config=config,
            diagnostic=diagnostic,
            data_catalog=data_catalog,
            execution_regression=execution_regression(diagnostic=diagnostic),  # type: ignore
        )

    return _create_validator
