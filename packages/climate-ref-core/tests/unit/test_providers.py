import io
import logging
import subprocess
from contextlib import contextmanager
from contextlib import nullcontext as does_not_raise
from pathlib import Path

import pytest
import pytest_mock
from requests import Response

import climate_ref_core.providers
from climate_ref.config import Config
from climate_ref_core.diagnostics import CommandLineDiagnostic, Diagnostic
from climate_ref_core.exceptions import InvalidDiagnosticException, InvalidProviderException
from climate_ref_core.providers import CondaDiagnosticProvider, DiagnosticProvider, import_provider


class TestMetricsProvider:
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


class TestCondaMetricsProvider:
    @pytest.fixture
    def provider(self, tmp_path, mocker):
        mocker.patch.object(
            climate_ref_core.providers.os.environ,
            "copy",
            return_value={"existing_var": "existing_value"},
        )
        provider = CondaDiagnosticProvider("provider_name", "v0.23")
        provider.prefix = tmp_path / "conda"
        return provider

    def test_no_prefix(self):
        provider = CondaDiagnosticProvider("provider_name", "v0.23")

        with pytest.raises(ValueError, match=r"No prefix for conda environments configured.*"):
            provider.prefix

    def test_configure(self, config: Config, provider: CondaDiagnosticProvider) -> None:
        provider.configure(config)

        assert isinstance(provider.prefix, Path)

        # Ensure configure() sets HOME to contain mamba writes
        assert "HOME" in provider.env_vars

    def test_preserves_env_vars(self, config: Config, mocker: pytest_mock.MockFixture) -> None:
        mock_env = mocker.patch.object(
            climate_ref_core.providers.os.environ,
            "copy",
            return_value={"preserved_var": "untouched", "overridden_var": "untouched"},
        )
        provider = CondaDiagnosticProvider("provider_name", "v0.23")
        provider.configure(config)
        provider.env_vars["overridden_var"] = "overridden"
        provider.env_vars["new_var"] = "added"

        # Ensure os.environ.copy was used vs manipulating the whole execution environ
        mock_env.assert_called_once()

        # Ensure existing env vars are preserved and new ones are added
        assert provider.env_vars == {
            "preserved_var": "untouched",
            "overridden_var": "overridden",
            "new_var": "added",
            "HOME": str(provider.prefix),
        }

    @pytest.mark.parametrize(
        "exists, update, is_stale, should_have_downloaded",
        [
            (True, True, True, True),
            (True, True, False, False),
            (True, False, True, False),
            (True, False, False, False),
            (False, True, True, True),
            (False, True, False, True),
            (False, False, True, True),
            (False, False, False, True),
        ],
    )
    def test_get_conda_exe(
        self, mocker: pytest_mock.MockFixture, provider, exists, update, is_stale, should_have_downloaded
    ):
        fake_file = io.BytesIO()

        mock_conda_exe = mocker.MagicMock(spec=Path, exists=lambda: exists)
        mock_conda_exe.open.return_value.__enter__.return_value.write = fake_file.write
        mock_conda_exe.read_bytes = lambda: fake_file.getvalue()
        mocker.patch.object(Path, "__truediv__", return_value=mock_conda_exe)

        mocker.patch.object(provider, "_is_stale", return_value=is_stale)
        mocker.patch("climate_ref_core.providers.MICROMAMBA_MAX_AGE", 0)

        mock_response = mocker.MagicMock(spec=Response)
        mock_response.iter_content.return_value.__iter__.return_value = iter([b"test"])
        mock_get = mocker.patch.object(climate_ref_core.providers.requests, "get", return_value=mock_response)

        if should_have_downloaded:
            assert provider.get_conda_exe(update=update).read_bytes() == b"test"
            mock_response.raise_for_status.assert_called_once()
        else:
            mock_get.assert_not_called()

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
            env={"existing_var": "existing_value"},
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

    @pytest.mark.parametrize(
        ("env_exists", "raised"),
        [
            (True, does_not_raise()),
            (
                False,
                pytest.raises(
                    RuntimeError,
                    match=r"Conda environment for provider `provider_name` not available at .*",
                ),
            ),
        ],
    )
    def test_run(self, mocker: pytest_mock.MockerFixture, tmp_path, provider, env_exists, raised):
        conda_exe = tmp_path / "conda" / "micromamba"
        mock_env_path = mocker.Mock(
            spec=Path,
            new_callable=mocker.PropertyMock,
            exists=lambda: env_exists,
            __str__=lambda _: str(provider.prefix / "mock-env"),
        )

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
            return_value=mock_env_path,
        )

        run = mocker.patch.object(
            climate_ref_core.providers.subprocess,
            "run",
            create_autospec=True,
        )

        provider.env_vars["test_var"] = "test_value"

        with raised:
            provider.run(["mock-command"])

            run.assert_called_with(
                [
                    f"{conda_exe}",
                    "run",
                    "--prefix",
                    f"{mock_env_path}",
                    "mock-command",
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                env={"existing_var": "existing_value", "test_var": "test_value"},
            )
