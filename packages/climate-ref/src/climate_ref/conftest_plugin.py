"""
Pytest plugin providing shared fixtures for diagnostic provider testing.

This plugin is registered as ``climate_ref`` in the ``pytest11`` entry point group.
Provider packages can use these fixtures by adding ``climate-ref[test]`` to their
test dependencies.

Usage in a provider's conftest.py::

    pytest_plugins = ("climate_ref.conftest_plugin",)

Or install ``climate-ref[test]`` and the plugin is auto-discovered.

Provided fixtures
-----------------
- ``config`` -- per-test ``Config`` with isolated directories
- ``caplog`` -- loguru-compatible log capture
- ``test_data_dir`` / ``sample_data_dir`` / ``regression_data_dir`` -- data paths
- ``sample_data`` -- session-scoped sample data fetch
- ``cmip6_data_catalog`` / ``obs4mips_data_catalog`` / ``data_catalog`` -- data catalogs
- ``run_test_case`` -- ``TestCaseRunner`` wrapper that converts errors to ``pytest.skip``
- ``definition_factory`` -- create ``ExecutionDefinition`` instances
- ``provider`` / ``mock_diagnostic`` -- mock diagnostic provider
- ``execution_regression`` -- regression output management
- ``diagnostic_validation`` -- legacy validation helper
- ``invoke_cli`` -- CLI test runner
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, cast

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
from climate_ref.solve_helpers import load_solve_catalog
from climate_ref.solver import solve_executions
from climate_ref.testing import (
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


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: mark test as slow to run")
    config.addinivalue_line("markers", "docker: mark test requires docker to run")
    config.addinivalue_line("markers", "requires_esgf_data: mark test requires ESGF test data")


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add custom CLI options."""
    parser.addoption("--slow", action="store_true", help="include tests marked slow")
    parser.addoption("--no-docker", action="store_true", help="skip docker tests")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip slow/docker tests unless opted in."""
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
def tmp_path_session() -> Iterator[Path]:
    """Session-scoped temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def caplog(caplog: LogCaptureFixture) -> Iterator[LogCaptureFixture]:
    """Capture logs from the loguru default logger."""

    def filter_(record: dict[str, Any]) -> bool:
        return bool(record["level"].no >= caplog.handler.level)

    add_log_handler(sink=caplog.handler, level=0, format="{message}", filter=filter_)
    yield caplog
    remove_log_handler()


@pytest.fixture(autouse=True)
def cleanup_log_handlers(request: pytest.FixtureRequest) -> Iterator[None]:
    """Remove any dangling loguru handlers after each test."""
    yield
    if hasattr(logger, "default_handler_id"):
        logger.warning("Logger handler not removed, removing it now")
        remove_log_handler()


@pytest.fixture(scope="session")
def test_data_dir() -> Path:
    """Path to the centralised test data directory."""
    if TEST_DATA_DIR is None:
        raise ValueError("Test data should exist when running tests")
    return TEST_DATA_DIR


@pytest.fixture(scope="session")
def sample_data_dir(test_data_dir: Path) -> Path:
    """Path to the sample data directory."""
    return test_data_dir / "sample-data"


@pytest.fixture(scope="session")
def regression_data_dir(test_data_dir: Path) -> Path:
    """Path to the regression data directory."""
    return test_data_dir / "regression"


@pytest.fixture(scope="session")
def esgf_solve_catalog(test_data_dir: Path) -> dict[SourceDatasetType, pd.DataFrame] | None:
    """Load ESGF metadata catalog for solve tests, if available."""
    return load_solve_catalog(test_data_dir / "esgf-catalog")


@pytest.fixture(scope="session")
def esgf_data_catalog(
    esgf_solve_catalog: dict[SourceDatasetType, pd.DataFrame] | None,
    test_data_dir: Path,
) -> dict[SourceDatasetType, pd.DataFrame]:
    """
    ESGF metadata catalog for tests that only need DataFrames, not actual files.

    Uses pre-generated parquet catalogs from ``tests/test-data/esgf-catalog/``.
    Fails if the catalog is not available (run ``scripts/generate_esgf_catalog.py``).
    """
    if esgf_solve_catalog is None:
        expected_path = test_data_dir / "esgf-catalog"
        pytest.fail(
            f"ESGF parquet catalog not found in {expected_path}. "
            "Run scripts/generate_esgf_catalog.py to generate it."
        )
    return esgf_solve_catalog


@pytest.fixture
def run_test_case(config: Config) -> object:
    """
    Fixture for running diagnostic test cases.

    Wraps ``TestCaseRunner`` to convert ``TestCaseError`` into ``pytest.skip``.
    """
    runner = TestCaseRunner(config=config, datasets=None)

    class PytestTestCaseRunner:
        def run(
            self,
            diagnostic: Diagnostic,
            test_case_name: str = "default",
            output_dir: Path | None = None,
        ) -> ExecutionResult:
            try:
                return runner.run(diagnostic, test_case_name, output_dir)
            except TestCaseError as e:
                pytest.skip(str(e))
                raise  # unreachable, but keeps type checkers happy

    return PytestTestCaseRunner()


@pytest.fixture(scope="session")
def sample_data() -> None:
    """Download sample data if not already present."""
    if os.environ.get("REF_TEST_DATA_DIR"):
        logger.warning("Not fetching sample data. Using custom test data directory")
        return
    logger.disable("climate_ref_core.dataset_registry")
    fetch_sample_data(force_cleanup=False, symlink=False)
    logger.enable("climate_ref_core.dataset_registry")


@pytest.fixture(scope="session")
def cmip6_data_catalog(sample_data: None, sample_data_dir: Path) -> pd.DataFrame:
    """CMIP6 sample data catalog."""
    adapter = CMIP6DatasetAdapter()
    return adapter.find_local_datasets(sample_data_dir / "CMIP6")


@pytest.fixture(scope="session")
def obs4mips_data_catalog(sample_data: None, sample_data_dir: Path) -> pd.DataFrame:
    """obs4MIPs sample data catalog."""
    adapter = Obs4MIPsDatasetAdapter()
    obs4ref = adapter.find_local_datasets(sample_data_dir / "obs4REF")
    obs4mips = adapter.find_local_datasets(sample_data_dir / "obs4MIPs")
    return pd.concat([obs4ref, obs4mips], ignore_index=True)


@pytest.fixture(scope="session")
def data_catalog(
    cmip6_data_catalog: pd.DataFrame, obs4mips_data_catalog: pd.DataFrame
) -> dict[SourceDatasetType, pd.DataFrame]:
    """Provide combined data catalog with CMIP6 and obs4MIPs sources."""
    return {
        SourceDatasetType.CMIP6: cmip6_data_catalog,
        SourceDatasetType.obs4MIPs: obs4mips_data_catalog,
    }


@pytest.fixture
def config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest) -> Config:
    """Per-test Config with isolated directories."""
    root_output_dir = Path(os.environ.get("REF_TEST_OUTPUT", tmp_path / "climate_ref"))
    dir_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", request.node.name)
    ref_config_dir = root_output_dir / request.module.__name__ / dir_name

    software_path = Config.default().paths.software

    monkeypatch.setenv("REF_CONFIGURATION", str(ref_config_dir))
    cfg = Config.default()
    cfg.paths.software = software_path
    cfg.diagnostic_providers = [DiagnosticProviderConfig(provider="climate_ref_example")]
    cfg.executor.executor = "climate_ref.executor.SynchronousExecutor"
    cfg.save()

    return cfg


@pytest.fixture
def invoke_cli(monkeypatch: pytest.MonkeyPatch) -> Callable[..., Result]:
    """Invoke the REF CLI and verify exit code."""
    runner = CliRunner()
    runner.mix_stderr = False

    def _invoke_cli(args: list[str], expected_exit_code: int = 0, always_log: bool = False) -> Result:
        monkeypatch.setenv("NO_COLOR", "1")
        monkeypatch.setenv("COLUMNS", "200")

        result = runner.invoke(app=cli.app, args=args)

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
    """A no-op diagnostic for testing."""

    name = "mock"
    slug = "mock"
    data_requirements = (DataRequirement(source_type=SourceDatasetType.CMIP6, filters=(), group_by=None),)

    def run(self, definition: ExecutionDefinition) -> ExecutionResult:
        """Run a no-op diagnostic that always succeeds."""
        return ExecutionResult(
            output_bundle_filename=definition.output_directory / "output.json",
            metric_bundle_filename=definition.output_directory / "diagnostic.json",
            successful=True,
            definition=definition,
        )


class FailedDiagnostic(Diagnostic):
    """A diagnostic that always fails, for testing."""

    name = "failed"
    slug = "failed"
    data_requirements = (DataRequirement(source_type=SourceDatasetType.CMIP6, filters=(), group_by=None),)

    def run(self, definition: ExecutionDefinition) -> ExecutionResult:
        """Run a diagnostic that always returns a failure result."""
        return ExecutionResult.build_from_failure(definition)


@pytest.fixture
def provider(tmp_path: Path, config: Config) -> DiagnosticProvider:
    """Create a mock provider with mock and failed diagnostics registered."""
    provider = DiagnosticProvider("mock_provider", "v0.1.0")
    provider.register(MockDiagnostic())  # type: ignore
    provider.register(FailedDiagnostic())  # type: ignore
    provider.configure(config)
    return provider


@pytest.fixture
def mock_diagnostic(provider: DiagnosticProvider) -> MockDiagnostic:
    """Return the mock diagnostic from the mock provider."""
    return cast(MockDiagnostic, provider.get("mock"))


@pytest.fixture
def definition_factory(tmp_path: Path, config: Config) -> Callable[..., ExecutionDefinition]:
    """Create ExecutionDefinition instances for testing."""

    def _create_definition(
        *,
        diagnostic: Diagnostic,
        execution_dataset_collection: ExecutionDatasetCollection | None = None,
        cmip6: DatasetCollection | None = None,
        obs4mips: DatasetCollection | None = None,
        pmp_climatology: DatasetCollection | None = None,
    ) -> ExecutionDefinition:
        if execution_dataset_collection is None:
            datasets: dict[SourceDatasetType | str, DatasetCollection] = {}
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
            root_directory=config.paths.scratch,
            output_directory=config.paths.scratch / "output_fragment",
        )

    return _create_definition


@pytest.fixture
def metric_definition(
    definition_factory: Callable[..., ExecutionDefinition],
    cmip6_data_catalog: pd.DataFrame,
    mock_diagnostic: MockDiagnostic,
) -> ExecutionDefinition:
    """Create an ExecutionDefinition with a selected CMIP6 dataset for metric tests."""
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
    """Copy execution output to the test-data directory for regression testing."""

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

    def _replace_file(self, file: Path, replacements: dict[str, str]) -> None:
        with open(file, encoding="utf-8") as f:
            content = f.read()
            for key, value in replacements.items():
                content = content.replace(key, value)
        with open(file, "w") as f:
            f.write(content)

    def path(self, key: str) -> Path:
        """Return the regression data path for the given key."""
        return self.regression_data_dir / self.diagnostic.provider.slug / self.diagnostic.slug / key

    def replace_references(self, output_dir: Path, replacements: dict[str, str]) -> None:
        """Replace any references to local directories with a placeholder."""
        for glob in self.sanitised_file_globs:
            for file in output_dir.rglob(glob):
                self._replace_file(file, replacements)

    def hydrate_output_directory(self, output_dir: Path, replacements: dict[str, str]) -> None:
        """Replace any references to the placeholder with the actual output directory."""
        for glob in self.sanitised_file_globs:
            for file in output_dir.rglob(glob):
                self._replace_file(file, replacements)

    def check(self, key: str, output_directory: Path) -> None:
        """Check and optionally regenerate regression data."""
        if not self.request.config.getoption("force_regen"):
            logger.info("Not regenerating regression results")
            return
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
    request: pytest.FixtureRequest, regression_data_dir: Path, test_data_dir: Path
) -> Callable[[Diagnostic], ExecutionRegression]:
    """Create ExecutionRegression instances for a diagnostic."""

    def _regression(diagnostic: Diagnostic) -> ExecutionRegression:
        return ExecutionRegression(
            diagnostic=diagnostic,
            regression_data_dir=regression_data_dir,
            request=request,
            replacements={str(test_data_dir): "<TEST_DATA_DIR>"},
        )

    return _regression


@define
class DiagnosticValidator:
    """
    Validator for running diagnostics with sample data.

    .. deprecated::
        Use ``RegressionValidator`` from ``climate_ref_core.testing`` instead.
    """

    config: Config
    diagnostic: Diagnostic
    data_catalog: dict[SourceDatasetType, pd.DataFrame]
    execution_regression: ExecutionRegression

    def get_definition(self) -> ExecutionDefinition:
        """Build an execution definition from the data catalog."""
        execution = next(
            solve_executions(
                data_catalog=self.data_catalog,
                diagnostic=self.diagnostic,
                provider=self.diagnostic.provider,
            )
        )
        return execution.build_execution_definition(output_root=self.config.paths.scratch)

    def get_regression_definition(self) -> ExecutionDefinition:
        """Load regression data and build an execution definition."""
        definition = self.get_definition()
        regression_output_dir = self.execution_regression.path(definition.key)
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
        """Run the diagnostic and optionally save regression data."""
        definition.output_directory.mkdir(parents=True, exist_ok=True)
        try:
            self.diagnostic.run(definition)
        finally:
            self.execution_regression.check(key=definition.key, output_directory=definition.output_directory)

    def validate(self, definition: ExecutionDefinition) -> None:
        """Validate CMEC bundles and store the execution result."""
        result = self.diagnostic.build_execution_result(definition)
        result.to_output_path("out.log").touch()
        validate_result(self.diagnostic, self.config, result)


@pytest.fixture
def diagnostic_validation(
    config: Config,
    mocker: Any,
    provider: DiagnosticProvider,
    data_catalog: dict[SourceDatasetType, pd.DataFrame],
    execution_regression: Callable[[Diagnostic], ExecutionRegression],
) -> Callable[[Diagnostic], DiagnosticValidator]:
    """Create DiagnosticValidator instances for testing."""
    mocker.patch.object(Execution, "execution_group")

    def _create_validator(diagnostic: Diagnostic) -> DiagnosticValidator:
        diagnostic.provider.configure(config)
        return DiagnosticValidator(
            config=config,
            diagnostic=diagnostic,
            data_catalog=data_catalog,
            execution_regression=execution_regression(diagnostic),
        )

    return _create_validator
