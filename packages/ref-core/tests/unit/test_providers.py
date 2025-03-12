import datetime
import logging
import time
from contextlib import contextmanager
from pathlib import Path

import pytest

import cmip_ref_core.providers
from cmip_ref_core.exceptions import InvalidMetricException, InvalidProviderException
from cmip_ref_core.metrics import CommandLineMetric, Metric
from cmip_ref_core.providers import CondaMetricsProvider, MetricsProvider, import_provider


class TestMetricsProvider:
    def test_provider(self):
        provider = MetricsProvider("provider_name", "v0.23")

        assert provider.name == "provider_name"
        assert provider.version == "v0.23"
        assert len(provider) == 0

    def test_provider_register(self, mock_metric):
        provider = MetricsProvider("provider_name", "v0.23")
        provider.register(mock_metric)

        assert len(provider) == 1
        assert "mock" in provider._metrics
        assert isinstance(provider.get("mock"), Metric)

        assert len(provider.metrics()) == 1
        assert provider.metrics()[0].name == "mock"

    def test_provider_register_invalid(self):
        class InvalidMetric:
            pass

        provider = MetricsProvider("provider_name", "v0.23")
        with pytest.raises(InvalidMetricException):
            provider.register(InvalidMetric())

    def test_provider_fixture(self, provider):
        assert provider.name == "mock_provider"
        assert provider.version == "v0.1.0"
        assert len(provider) == 2
        assert "mock" in provider._metrics
        assert "failed" in provider._metrics

        result = provider.get("mock")
        assert isinstance(result, Metric)


@pytest.mark.parametrize("fqn", ["cmip_ref_metrics_esmvaltool.provider", "cmip_ref_metrics_esmvaltool"])
def test_import_provider(fqn):
    provider = import_provider(fqn)

    assert provider.name == "ESMValTool"
    assert provider.slug == "esmvaltool"
    assert isinstance(provider, MetricsProvider)


def test_import_provider_missing():
    fqn = "cmip_ref"
    match = f"Invalid provider: '{fqn}'\n Provider 'provider' not found in cmip_ref"
    with pytest.raises(InvalidProviderException, match=match):
        import_provider(fqn)

    fqn = "cmip_ref.datasets.WrongProvider"
    match = f"Invalid provider: '{fqn}'\n Provider 'WrongProvider' not found in cmip_ref.datasets"
    with pytest.raises(InvalidProviderException, match=match):
        import_provider(fqn)

    fqn = "missing.local.WrongProvider"
    match = f"Invalid provider: '{fqn}'\n Module 'missing.local' not found"
    with pytest.raises(InvalidProviderException, match=match):
        import_provider(fqn)

    fqn = "cmip_ref.constants.config_filename"
    match = f"Invalid provider: '{fqn}'\n Expected MetricsProvider, got <class 'str'>"
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
    uname = mocker.patch.object(cmip_ref_core.providers.os, "uname", create_autospec=True)
    uname.return_value.sysname = sysname
    uname.return_value.machine = machine
    if sysname == "Unknown":
        with pytest.raises(ValueError):
            cmip_ref_core.providers._get_micromamba_url()
    else:
        result = cmip_ref_core.providers._get_micromamba_url()
        assert "{" not in result


class TestCondaMetricsProvider:
    @pytest.fixture
    def provider(self, tmp_path):
        provider = CondaMetricsProvider("provider_name", "v0.23")
        provider.prefix = tmp_path / "conda"
        return provider

    def test_no_prefix(self):
        provider = CondaMetricsProvider("provider_name", "v0.23")

        with pytest.raises(ValueError, match="No prefix for conda environments configured.*"):
            provider.prefix

    def test_configure(self, config):
        provider = CondaMetricsProvider("provider_name", "v0.23")
        provider.configure(config)

        assert isinstance(provider.prefix, Path)

    @pytest.mark.parametrize("update", [True, False])
    def test_get_conda_exe(self, mocker, provider, update):
        if update:
            conda_exe = provider.prefix / "micromamba"
            provider.prefix.mkdir()
            conda_exe.touch()
            mocker.patch.object(
                cmip_ref_core.providers,
                "MICROMAMBA_MAX_AGE",
                datetime.timedelta(microseconds=1),
            )
            time.sleep(0.01)  # wait for the executable to expire.

        get = mocker.patch.object(
            cmip_ref_core.providers.requests,
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
            CondaMetricsProvider,
            "_install_conda",
            create_autospec=True,
        )

        result = provider.get_conda_exe(update=True)

        assert result == conda_exe
        provider._install_conda.assert_not_called()

    def test_no_module(self, provider):
        with pytest.raises(ValueError, match="Unable to determine the provider module.*"):
            provider.get_environment_file()

    def test_env_path(self, mocker, tmp_path, provider):
        metric = mocker.create_autospec(CommandLineMetric)
        metric.slug = "mock-metric"
        metric.__module__ = "mock_metric_provider.metrics.mock_metric"
        provider.register(metric)

        resources = mocker.patch.object(
            cmip_ref_core.providers.importlib,
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
            CondaMetricsProvider,
            "get_environment_file",
            create_autospec=True,
            return_value=lockfile_context(),
        )
        mocker.patch.object(
            CondaMetricsProvider,
            "get_conda_exe",
            create_autospec=True,
            return_value=conda_exe,
        )
        mocker.patch.object(
            CondaMetricsProvider,
            "env_path",
            new_callable=mocker.PropertyMock,
            return_value=env_path,
        )

        run = mocker.patch.object(
            cmip_ref_core.providers.subprocess,
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
            CondaMetricsProvider,
            "env_path",
            new_callable=mocker.PropertyMock,
            return_value=env_path,
        )
        caplog.set_level(logging.INFO)

        provider.create_env()

        assert f"Environment at {env_path} already exists, skipping." in caplog.text

    def test_run(self, mocker, tmp_path, provider):
        conda_exe = tmp_path / "conda" / "micromamba"
        env_path = provider.prefix / "mock-env"

        mocker.patch.object(
            CondaMetricsProvider,
            "create_env",
            create_autospec=True,
        )
        mocker.patch.object(
            CondaMetricsProvider,
            "get_conda_exe",
            create_autospec=True,
            return_value=conda_exe,
        )
        mocker.patch.object(
            CondaMetricsProvider,
            "env_path",
            new_callable=mocker.PropertyMock,
            return_value=env_path,
        )

        run = mocker.patch.object(
            cmip_ref_core.providers.subprocess,
            "run",
            create_autospec=True,
        )

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
        )
