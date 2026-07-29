import importlib.metadata
import io
import logging
import subprocess
import textwrap
from contextlib import contextmanager
from contextlib import nullcontext as does_not_raise
from pathlib import Path

import pytest
import pytest_mock
from requests import Response

import climate_ref_core.providers
from climate_ref_core.constraints import IgnoreFacets
from climate_ref_core.data import DataResourceError, LayeredResource, PackagedResource
from climate_ref_core.diagnostics import CommandLineDiagnostic, Diagnostic
from climate_ref_core.exceptions import (
    CondaCommandError,
    InvalidDiagnosticException,
    InvalidProviderException,
)
from climate_ref_core.providers import (
    CondaDiagnosticProvider,
    DiagnosticProvider,
    import_provider,
    provider_by_slug,
    resolve_diagnostic,
)


@pytest.fixture
def mock_config(tmp_path, mocker):
    """Use a mock config to avoid depending on `climate_ref.config.Config`."""
    config = mocker.Mock()
    config.paths.software = tmp_path / "software"
    config.ignore_datasets_file = tmp_path / "ignore_datasets.yaml"
    config.ignore_datasets_file.touch()
    # The grey list is resolved through a real LayeredResource so the tests exercise
    # the same resolution the application uses.
    config.ignore_datasets_resource = LayeredResource(
        packaged=PackagedResource("climate_ref_core.pycmec", "cv_cmip7_aft.yaml"),
        override=config.ignore_datasets_file,
    )
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

    def test_configure_missing_ignore_file(self, provider, mock_config, caplog):
        # A missing ignore datasets file (e.g. the cache could not be written) must not raise;
        # it is treated as "ignore nothing".
        mock_config.ignore_datasets_file.unlink()
        with caplog.at_level(logging.WARNING):
            provider.configure(mock_config)
        assert "Could not read the grey list" in caplog.text
        assert str(mock_config.ignore_datasets_file) in caplog.text

    def test_configure_missing_override_falls_back_to_the_packaged_grey_list(
        self, provider, mock_config, caplog
    ):
        # A mistyped ignore_datasets_file must not silently drop the grey list protections.
        mock_config.ignore_datasets_file.unlink()
        mock_config.ignore_datasets_resource = LayeredResource(
            packaged=PackagedResource("climate_ref", "default_ignore_datasets.yaml"),
            override=mock_config.ignore_datasets_file,
        )

        with caplog.at_level(logging.DEBUG):
            provider.configure(mock_config)

        # The override is reported as unreadable, and the packaged copy is used instead.
        assert f"Could not read the grey list from {mock_config.ignore_datasets_file}" in caplog.text
        assert "using the grey list from climate_ref/default_ignore_datasets.yaml" in caplog.text

    def test_configure_warns_when_no_grey_list_can_be_read(self, provider, mock_config, caplog, mocker):
        # Both the override and the packaged copy failing must be said out loud.
        mock_config.ignore_datasets_file.unlink()
        packaged = mocker.Mock()
        packaged.read_text.side_effect = DataResourceError("packaged copy is missing")
        packaged.describe.return_value = "packaged"
        packaged.__str__ = mocker.Mock(return_value="packaged")
        mock_config.ignore_datasets_resource = LayeredResource(
            packaged=PackagedResource("climate_ref", "default_ignore_datasets.yaml"),
            override=mock_config.ignore_datasets_file,
        )
        mocker.patch.object(
            type(mock_config.ignore_datasets_resource.packaged),
            "read_text",
            side_effect=DataResourceError("packaged copy is missing"),
        )

        with caplog.at_level(logging.WARNING):
            provider.configure(mock_config)

        assert "No grey list could be read" in caplog.text

    def test_configure_ignores_a_grey_list_that_is_not_a_mapping(self, provider, mock_config, caplog):
        # A hand-edited list at the top level must not crash the provider.
        mock_config.ignore_datasets_file.write_text("- not: a mapping\n", encoding="utf-8")

        with caplog.at_level(logging.WARNING):
            provider.configure(mock_config)

        assert "is not a mapping" in caplog.text

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
            f"Unknown diagnostics found in {mock_config.ignore_datasets_file} (override) "
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
            f"Unknown source types found in {mock_config.ignore_datasets_file} (override) "
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
    def provider(self, tmp_path, mocker):
        mocker.patch.object(
            climate_ref_core.providers.os,
            "environ",
            {"existing_var": "existing_value"},
        )
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

        # HOME is defaulted at launch so micromamba has a writable directory
        assert "HOME" in provider._launch_env()

    def test_launch_env_merges_the_live_environment(self, config, mocker: pytest_mock.MockFixture) -> None:
        mocker.patch.object(
            climate_ref_core.providers.os,
            "environ",
            {"preserved_var": "untouched", "overridden_var": "untouched"},
        )
        provider = CondaDiagnosticProvider("provider_name", "v0.23")
        provider.configure(config)
        provider.env_overrides["overridden_var"] = "overridden"
        provider.env_overrides["new_var"] = "added"

        # The base is read at launch, so a variable set after construction still applies
        climate_ref_core.providers.os.environ["late_var"] = "set-later"

        assert provider._launch_env() == {
            "preserved_var": "untouched",
            "overridden_var": "overridden",
            "new_var": "added",
            "late_var": "set-later",
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
        mock_conda_exe.read_bytes = fake_file.getvalue
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
            env=mocker.ANY,
        )
        env = run.call_args.kwargs["env"]
        assert env["existing_var"] == "existing_value"
        assert env["HOME"] == str(provider.prefix)

    def test_create_env_with_pip_packages(self, mocker, tmp_path, provider):
        lockfile = tmp_path / "conda-lock.yml"
        conda_exe = tmp_path / "conda" / "micromamba"
        env_path = provider.prefix / "mock-env"

        provider.pip_packages = [
            "git+https://example.com/tool.git@abc123",
            "git+https://example.com/core.git@def456",
        ]

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

        pip_install_calls = [c for c in run.call_args_list if "pip" in c.args[0]]
        assert len(pip_install_calls) == 2
        assert pip_install_calls[0].args[0][-1] == "git+https://example.com/tool.git@abc123"
        assert pip_install_calls[1].args[0][-1] == "git+https://example.com/core.git@def456"

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

        provider.env_overrides["test_var"] = "test_value"

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
                env=mocker.ANY,
            )
            env = run.call_args.kwargs["env"]
            assert env["existing_var"] == "existing_value"
            assert env["test_var"] == "test_value"
            assert env["HOME"] == str(provider.prefix)

    def test_run_command_fails(self, mocker: pytest_mock.MockerFixture, tmp_path, provider):
        """Test that run() re-raises CalledProcessError when command fails."""
        conda_exe = tmp_path / "conda" / "micromamba"
        mock_env_path = mocker.Mock(
            spec=Path,
            new_callable=mocker.PropertyMock,
            exists=lambda: True,
            __str__=lambda _: str(provider.prefix / "mock-env"),
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

        # Mock subprocess.run to raise CalledProcessError
        error = subprocess.CalledProcessError(1, "mock-command", output="error output")
        error.stdout = "error output"
        mocker.patch.object(
            climate_ref_core.providers.subprocess,
            "run",
            create_autospec=True,
            side_effect=error,
        )

        with pytest.raises(CondaCommandError) as exc_info:
            provider.run(["mock-command"])
        assert "error output" in exc_info.value.stdout


class TestLifecycleHooks:
    """Tests for provider lifecycle hooks."""

    def test_setup_calls_hooks_in_order(self, mocker):
        """Test that setup() calls hooks in the correct order."""
        provider = DiagnosticProvider("test", "1.0")
        mock_config = mocker.Mock()

        # Mock all the individual hooks
        setup_env = mocker.patch.object(provider, "setup_environment")
        fetch_data = mocker.patch.object(provider, "fetch_data")

        provider.setup(mock_config)

        # Verify called in order
        setup_env.assert_called_once_with(mock_config)
        fetch_data.assert_called_once_with(mock_config)

    def test_setup_skip_env(self, mocker):
        """Test that setup() skips setup_environment when skip_env=True."""
        provider = DiagnosticProvider("test", "1.0")
        mock_config = mocker.Mock()

        setup_env = mocker.patch.object(provider, "setup_environment")
        fetch_data = mocker.patch.object(provider, "fetch_data")

        provider.setup(mock_config, skip_env=True)

        setup_env.assert_not_called()
        fetch_data.assert_called_once_with(mock_config)

    def test_setup_skip_data(self, mocker):
        """Test that setup() skips fetch_data when skip_data=True."""
        provider = DiagnosticProvider("test", "1.0")
        mock_config = mocker.Mock()

        setup_env = mocker.patch.object(provider, "setup_environment")
        fetch_data = mocker.patch.object(provider, "fetch_data")

        provider.setup(mock_config, skip_data=True)

        setup_env.assert_called_once_with(mock_config)
        fetch_data.assert_not_called()

    def test_setup_skip_both(self, mocker):
        """Test that setup() skips both when both skip flags are True."""
        provider = DiagnosticProvider("test", "1.0")
        mock_config = mocker.Mock()

        setup_env = mocker.patch.object(provider, "setup_environment")
        fetch_data = mocker.patch.object(provider, "fetch_data")

        provider.setup(mock_config, skip_env=True, skip_data=True)

        setup_env.assert_not_called()
        fetch_data.assert_not_called()

    def test_default_hooks_are_noop(self, mocker):
        """Test that default hook implementations do nothing."""
        provider = DiagnosticProvider("test", "1.0")
        mock_config = mocker.Mock()

        # These should not raise
        provider.setup_environment(mock_config)
        provider.fetch_data(mock_config)

    def test_validate_setup_default_returns_true(self, mocker):
        """Test that default validate_setup returns True."""
        provider = DiagnosticProvider("test", "1.0")
        mock_config = mocker.Mock()

        assert provider.validate_setup(mock_config) is True

    def test_get_data_path_default_returns_none(self):
        """Test that default get_data_path returns None."""
        provider = DiagnosticProvider("test", "1.0")

        assert provider.get_data_path() is None

    def test_conda_setup_environment_calls_create_env(self, mocker, tmp_path):
        """Test that CondaDiagnosticProvider.setup_environment calls create_env."""
        mocker.patch.object(
            climate_ref_core.providers.os.environ,
            "copy",
            return_value={},
        )
        provider = CondaDiagnosticProvider("test", "1.0")
        provider.prefix = tmp_path / "conda"
        mock_config = mocker.Mock()

        create_env = mocker.patch.object(provider, "create_env")

        provider.setup_environment(mock_config)

        create_env.assert_called_once()

    def test_conda_validate_setup_checks_env_path(self, mocker, tmp_path):
        """Test that CondaDiagnosticProvider.validate_setup checks env_path exists."""
        mocker.patch.object(
            climate_ref_core.providers.os.environ,
            "copy",
            return_value={},
        )
        provider = CondaDiagnosticProvider("test", "1.0")
        provider.prefix = tmp_path / "conda"
        mock_config = mocker.Mock()

        env_path = tmp_path / "conda" / "test-env"
        mocker.patch.object(
            CondaDiagnosticProvider,
            "env_path",
            new_callable=mocker.PropertyMock,
            return_value=env_path,
        )

        # Should return False when env_path doesn't exist
        assert provider.validate_setup(mock_config) is False

        # Create the path
        env_path.mkdir(parents=True)

        # Should return True when env_path exists
        assert provider.validate_setup(mock_config) is True


class TestProviderLookup:
    @pytest.fixture(autouse=True)
    def _clear_the_lookup_cache(self):
        # provider_by_slug is cached for the life of the process,
        # so a mocked set of entry points would otherwise be answered from a previous test.
        provider_by_slug.cache_clear()
        yield
        provider_by_slug.cache_clear()

    @staticmethod
    def _fake_entry_points(mocker, names):
        """Register entry points named `names`, each pointing at a value of the same name."""
        entry_points = [
            importlib.metadata.EntryPoint(name=name, value=name, group="climate-ref.providers")
            for name in names
        ]
        mocker.patch.object(
            climate_ref_core.providers.importlib.metadata,
            "entry_points",
            return_value=entry_points,
        )
        return entry_points

    @staticmethod
    def _fake_imports(mocker, providers):
        """
        Resolve an entry point value through `providers`, recording the order of the attempts.

        A value mapped to an exception raises it, standing in for a provider that fails to import.
        """
        attempted = []

        def _import(value):
            attempted.append(value)
            result = providers[value]
            if isinstance(result, Exception):
                raise result
            return result

        mocker.patch.object(climate_ref_core.providers, "import_provider", side_effect=_import)
        return attempted

    def test_provider_by_slug(self):
        # The same singleton the entry point exposes, so it carries any configuration
        # the current process has already applied to it
        assert provider_by_slug("example") is import_provider("climate_ref_example:provider")

    def test_provider_by_slug_unknown(self):
        with pytest.raises(InvalidProviderException, match="No provider with slug 'nope'"):
            provider_by_slug("nope")

    def test_the_named_entry_point_is_tried_first(self, mocker):
        """The name is the slug by convention, so nothing else should need importing."""
        self._fake_entry_points(mocker, ["aaa", "wanted", "zzz"])
        wanted = mocker.Mock(slug="wanted")
        attempted = self._fake_imports(
            mocker,
            {"aaa": mocker.Mock(slug="aaa"), "wanted": wanted, "zzz": mocker.Mock(slug="zzz")},
        )

        assert provider_by_slug("wanted") is wanted
        assert attempted == ["wanted"]

    def test_an_entry_point_whose_name_differs_from_its_slug_is_still_found(self, mocker):
        self._fake_entry_points(mocker, ["misnamed"])
        provider = mocker.Mock(slug="wanted")
        self._fake_imports(mocker, {"misnamed": provider})

        assert provider_by_slug("wanted") is provider

    def test_a_broken_provider_does_not_mask_the_one_being_looked_for(self, mocker):
        """A provider we were not asked for is skipped when it fails to import."""
        self._fake_entry_points(mocker, ["broken", "misnamed"])
        wanted = mocker.Mock(slug="wanted")
        attempted = self._fake_imports(
            mocker,
            {"broken": InvalidProviderException("broken", "boom"), "misnamed": wanted},
        )

        assert provider_by_slug("wanted") is wanted
        assert attempted == ["broken", "misnamed"]

    def test_a_broken_named_provider_propagates(self, mocker):
        """Failing to import the provider actually asked for is an error, not a miss."""
        self._fake_entry_points(mocker, ["wanted", "other"])
        self._fake_imports(
            mocker,
            {
                "wanted": InvalidProviderException("wanted", "boom"),
                "other": mocker.Mock(slug="other"),
            },
        )

        with pytest.raises(InvalidProviderException, match="boom"):
            provider_by_slug("wanted")

    def test_an_unmatched_slug_reports_what_is_available(self, mocker):
        self._fake_entry_points(mocker, ["zzz", "aaa"])
        self._fake_imports(mocker, {"zzz": mocker.Mock(slug="zzz"), "aaa": mocker.Mock(slug="aaa")})

        with pytest.raises(InvalidProviderException, match="Available: aaa, zzz"):
            provider_by_slug("wanted")

    def test_resolve_diagnostic(self):
        provider = import_provider("climate_ref_example:provider")

        assert resolve_diagnostic("example/global-mean-timeseries") is provider.get("global-mean-timeseries")

    def test_resolve_diagnostic_requires_a_provider_prefix(self):
        with pytest.raises(InvalidProviderException, match="provider/diagnostic"):
            resolve_diagnostic("global-mean-timeseries")

    def test_resolve_diagnostic_unknown_diagnostic(self):
        with pytest.raises(KeyError):
            resolve_diagnostic("example/not-a-diagnostic")
