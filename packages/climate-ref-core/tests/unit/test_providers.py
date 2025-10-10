import datetime
import logging
import subprocess
import textwrap
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

import climate_ref_core.providers
from climate_ref_core.constraints import IgnoreFacets
from climate_ref_core.diagnostics import CommandLineDiagnostic, Diagnostic
from climate_ref_core.exceptions import InvalidDiagnosticException, InvalidProviderException
from climate_ref_core.providers import CondaDiagnosticProvider, DiagnosticProvider, import_provider


@pytest.fixture
def mock_config(tmp_path, mocker):
    """Use a mock config to avoid depending on `climate_ref.config.Config`."""
    config = mocker.Mock()
    config.paths.software = tmp_path / "software"
    config.ignore_datasets_file = tmp_path / "ignore_datasets.yaml"
    config.ignore_datasets_file.touch()
    return config


class TestDiagnosticProvider:
    def test_provider(self):
        provider = DiagnosticProvider("provider_name", "v0.23")

        assert provider.name == "provider_name"
        assert provider.version == "v0.23"
        assert len(provider) == 0
        assert repr(provider) == "DiagnosticProvider(name='provider_name', version='v0.23')"

    def test_provider_register(self, mock_diagnostic):
        provider = DiagnosticProvider("provider_name", "v0.23")
        provider.register(mock_diagnostic)

        assert len(provider) == 1
        assert "mock" in provider._diagnostics
        assert isinstance(provider.get("mock"), Diagnostic)

        assert len(provider.diagnostics()) == 1
        assert provider.diagnostics()[0].name == "mock"

    def test_provider_register_invalid(self):
        class InvalidMetric:
            pass

        provider = DiagnosticProvider("provider_name", "v0.23")
        with pytest.raises(InvalidDiagnosticException):
            provider.register(InvalidMetric())

    def test_provider_fixture(self, provider):
        assert provider.name == "mock_provider"
        assert provider.version == "v0.1.0"
        assert len(provider) == 2
        assert "mock" in provider._diagnostics
        assert "failed" in provider._diagnostics

        result = provider.get("mock")
        assert isinstance(result, Diagnostic)

    def test_configure(self, provider, mock_config):
        mock_config.ignore_datasets_file.write_text(
            textwrap.dedent(
                """
                mock_provider:
                  mock:
                    cmip6:
                      - source_id: A
                """
            ),
            encoding="utf-8",
        )
        provider.configure(mock_config)
        expected_constraint = IgnoreFacets(facets={"source_id": ("A",)})
        assert provider.diagnostics()[0].data_requirements[0][0].constraints[0] == expected_constraint

    def test_configure_unknown_diagnostic(self, provider, mock_config, caplog):
        mock_config.ignore_datasets_file.write_text(
            textwrap.dedent(
                """
                mock_provider:
                  invalid_diagnostic:
                    cmip6:
                      - source_id: A
                """
            ),
            encoding="utf-8",
        )
        with caplog.at_level(logging.WARNING):
            provider.configure(mock_config)
        expected_msg = (
            f"Unknown diagnostics found in {mock_config.ignore_datasets_file} "
            "for provider mock_provider: invalid_diagnostic"
        )
        assert expected_msg in caplog.text

    def test_configure_unknown_source_type(self, provider, mock_config, caplog):
        mock_config.ignore_datasets_file.write_text(
            textwrap.dedent(
                """
                mock_provider:
                  mock:
                    invalid_source_type:
                      - source_id: A
                """
            ),
            encoding="utf-8",
        )
        with caplog.at_level(logging.WARNING):
            provider.configure(mock_config)
        expected_msg = (
            f"Unknown source types found in {mock_config.ignore_datasets_file} "
            "for diagnostic 'mock' by provider mock_provider: invalid_source_type"
        )
        assert expected_msg in caplog.text


@pytest.mark.parametrize("fqn", ["climate_ref_esmvaltool:provider", "climate_ref_esmvaltool"])
def test_import_provider(fqn):
    provider = import_provider(fqn)

    assert provider.name == "ESMValTool"
    assert provider.slug == "esmvaltool"
    assert isinstance(provider, DiagnosticProvider)


def test_import_provider_missing():
    fqn = "climate_ref"
    match = f"Invalid provider: '{fqn}.provider'\n Provider not found in module"
    with pytest.raises(InvalidProviderException, match=match):
        import_provider(fqn)

    fqn = "climate_ref.datasets:WrongProvider"
    match = f"Invalid provider: '{fqn}'\n Provider not found in module"
    with pytest.raises(InvalidProviderException, match=match):
        import_provider(fqn)

    fqn = "missing.local:WrongProvider"
    match = f"Invalid provider: '{fqn}'\n Module not found"
    with pytest.raises(InvalidProviderException, match=match):
        import_provider(fqn)

    fqn = "climate_ref:__version__"
    match = f"Invalid provider: '{fqn}'\n Expected DiagnosticProvider, got <class 'str'>"
    with pytest.raises(InvalidProviderException, match=match):
        import_provider(fqn)


@pytest.mark.parametrize(
    "sysname,machine",
    [
        ("Linux", "x86_64"),
        ("Darwin", "x86_64"),
        ("Darwin", "arm64"),
        ("Unknown", "x86_64"),
    ],
)
def test_get_micromamba_url(mocker, sysname, machine):
    uname = mocker.patch.object(climate_ref_core.providers.os, "uname", create_autospec=True)
    uname.return_value.sysname = sysname
    uname.return_value.machine = machine
    if sysname == "Unknown":
        with pytest.raises(ValueError):
            climate_ref_core.providers._get_micromamba_url()
    else:
        result = climate_ref_core.providers._get_micromamba_url()
        assert "{" not in result


class TestCondaDiagnosticProvider:
    @pytest.fixture
    def provider(self, tmp_path):
        provider = CondaDiagnosticProvider("provider_name", "v0.23")
        provider.prefix = tmp_path / "conda"
        return provider

    def test_no_prefix(self):
        provider = CondaDiagnosticProvider("provider_name", "v0.23")

        with pytest.raises(ValueError, match=r"No prefix for conda environments configured.*"):
            provider.prefix

    def test_configure(self, mock_config):
        provider = CondaDiagnosticProvider("provider_name", "v0.23")
        provider.configure(mock_config)

        assert isinstance(provider.prefix, Path)

    @pytest.mark.parametrize("update", [True, False])
    def test_get_conda_exe(self, mocker, provider, update):
        if update:
            conda_exe = provider.prefix / "micromamba"
            provider.prefix.mkdir()
            conda_exe.touch()
            mocker.patch.object(
                climate_ref_core.providers,
                "MICROMAMBA_MAX_AGE",
                datetime.timedelta(microseconds=1),
            )
            time.sleep(0.01)  # wait for the executable to expire.

        get = mocker.patch.object(
            climate_ref_core.providers.requests,
            "get",
            create_autospec=True,
        )
        response = get.return_value
        response.content = b"test"

        result = provider.get_conda_exe(update=update)

        response.raise_for_status.assert_called_with()
        assert result.read_bytes() == b"test"

    def test_get_conda_exe_repeat(self, mocker, tmp_path, provider):
        conda_exe = tmp_path / "micromamba"
        provider._conda_exe = conda_exe
        mocker.patch.object(
            CondaDiagnosticProvider,
            "_install_conda",
            create_autospec=True,
        )

        result = provider.get_conda_exe(update=True)

        assert result == conda_exe
        provider._install_conda.assert_not_called()

    def test_no_module(self, provider):
        with pytest.raises(ValueError, match=r"Unable to determine the provider module.*"):
            provider.get_environment_file()

    def test_env_path(self, mocker, tmp_path, provider):
        metric = mocker.create_autospec(CommandLineDiagnostic)
        metric.slug = "mock-diagnostic"
        metric.__module__ = "mock_metric_provider.diagnostics.mock_metric"
        provider.register(metric)

        resources = mocker.patch.object(
            climate_ref_core.providers.importlib,
            "resources",
            create_autospec=True,
        )
        lockfile = tmp_path / "conda-lock.yml"
        lockfile.touch()

        @contextmanager
        def lockfile_context():
            yield lockfile

        resources.as_file.return_value = lockfile_context()

        env_path = provider.env_path
        assert isinstance(env_path, Path)
        assert env_path.is_relative_to(provider.prefix)
        assert env_path.name.startswith("provider_name")

    def test_create_env(self, mocker, tmp_path, provider):
        lockfile = tmp_path / "conda-lock.yml"
        conda_exe = tmp_path / "conda" / "micromamba"
        env_path = provider.prefix / "mock-env"

        @contextmanager
        def lockfile_context():
            yield lockfile

        mocker.patch.object(
            CondaDiagnosticProvider,
            "get_environment_file",
            create_autospec=True,
            return_value=lockfile_context(),
        )
        mocker.patch.object(
            CondaDiagnosticProvider,
            "get_conda_exe",
            create_autospec=True,
            return_value=conda_exe,
        )
        mocker.patch.object(
            CondaDiagnosticProvider,
            "env_path",
            new_callable=mocker.PropertyMock,
            return_value=env_path,
        )

        run = mocker.patch.object(
            climate_ref_core.providers.subprocess,
            "run",
            create_autospec=True,
        )

        provider.create_env()

        run.assert_called_with(
            [
                f"{conda_exe}",
                "create",
                "--yes",
                "--file",
                f"{lockfile}",
                "--prefix",
                f"{env_path}",
            ],
            check=True,
        )

    def test_skip_create_env(self, mocker, caplog, provider):
        env_path = provider.prefix / "mock-env"
        env_path.mkdir(parents=True)
        mocker.patch.object(
            CondaDiagnosticProvider,
            "env_path",
            new_callable=mocker.PropertyMock,
            return_value=env_path,
        )
        caplog.set_level(logging.INFO)

        provider.create_env()

        assert f"Environment at {env_path} already exists, skipping." in caplog.text

    @pytest.mark.parametrize("env_exists", [True, False])
    def test_run(self, mocker, tmp_path, provider, env_exists):
        conda_exe = tmp_path / "conda" / "micromamba"
        env_path = provider.prefix / "mock-env"
        if env_exists:
            env_path.mkdir(parents=True)

        mocker.patch.object(
            CondaDiagnosticProvider,
            "create_env",
            create_autospec=True,
        )
        mocker.patch.object(
            CondaDiagnosticProvider,
            "get_conda_exe",
            create_autospec=True,
            return_value=conda_exe,
        )
        mocker.patch.object(
            CondaDiagnosticProvider,
            "env_path",
            new_callable=mocker.PropertyMock,
            return_value=env_path,
        )

        run = mocker.patch.object(
            climate_ref_core.providers.subprocess,
            "run",
            create_autospec=True,
        )

        if not env_exists:
            with pytest.raises(
                RuntimeError,
                match=(f"Conda environment for provider `{provider.slug}` not available at {env_path}."),
            ):
                provider.run(["mock-command"])
        else:
            mocker.patch.object(
                climate_ref_core.providers.os.environ,
                "copy",
                return_value={"existing_var": "existing_value"},
            )
            provider.env_vars = {"test_var": "test_value"}
            provider.run(["mock-command"])

            run.assert_called_with(
                [
                    f"{conda_exe}",
                    "run",
                    "--prefix",
                    f"{env_path}",
                    "mock-command",
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env={"existing_var": "existing_value", "test_var": "test_value"},
            )
