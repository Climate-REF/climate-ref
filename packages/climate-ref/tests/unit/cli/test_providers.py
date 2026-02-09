from climate_ref.provider_registry import ProviderRegistry
from climate_ref_core.providers import CondaDiagnosticProvider, DiagnosticProvider


class TestProvidersList:
    def test_list(self, config, invoke_cli):
        result = invoke_cli(["providers", "list"])
        assert result.exit_code == 0
        assert "provider" in result.stdout
        assert "example" in result.stdout

    def test_list_with_conda_provider(self, config, invoke_cli, mocker, tmp_path):
        """Test list command shows conda environment info."""
        # Create a mock conda provider with env_path that doesn't exist
        mock_provider = mocker.MagicMock(spec=CondaDiagnosticProvider)
        mock_provider.slug = "conda-test"
        mock_provider.version = "1.0.0"
        mock_provider.env_path = tmp_path / "nonexistent_env"
        mock_provider.get_data_path.return_value = None

        mock_registry = mocker.MagicMock(spec=ProviderRegistry)
        mock_registry.providers = [mock_provider]
        mocker.patch.object(ProviderRegistry, "build_from_config", return_value=mock_registry)

        result = invoke_cli(["providers", "list"])
        assert result.exit_code == 0
        assert "conda-test" in result.stdout
        assert "(not installed)" in result.stdout

    def test_list_with_data_path_not_fetched(self, config, invoke_cli, mocker, tmp_path):
        """Test list command shows data path not fetched info."""
        # Create a mock provider with data_path that doesn't exist
        mock_provider = mocker.MagicMock(spec=DiagnosticProvider)
        mock_provider.slug = "data-test"
        mock_provider.version = "1.0.0"
        mock_provider.get_data_path.return_value = tmp_path / "nonexistent_data"

        mock_registry = mocker.MagicMock(spec=ProviderRegistry)
        mock_registry.providers = [mock_provider]
        mocker.patch.object(ProviderRegistry, "build_from_config", return_value=mock_registry)

        result = invoke_cli(["providers", "list"])
        assert result.exit_code == 0
        assert "data-test" in result.stdout
        assert "(not fetched)" in result.stdout

    def test_list_with_conda_env_installed(self, config, invoke_cli, mocker, tmp_path):
        """Test list command shows installed conda environment without suffix."""
        # Create a mock conda provider with env_path that exists
        env_path = tmp_path / "installed_env"
        env_path.mkdir()

        mock_provider = mocker.MagicMock(spec=CondaDiagnosticProvider)
        mock_provider.slug = "installed-conda"
        mock_provider.version = "1.0.0"
        mock_provider.env_path = env_path
        mock_provider.get_data_path.return_value = None

        mock_registry = mocker.MagicMock(spec=ProviderRegistry)
        mock_registry.providers = [mock_provider]
        mocker.patch.object(ProviderRegistry, "build_from_config", return_value=mock_registry)

        result = invoke_cli(["providers", "list"])
        assert result.exit_code == 0
        assert "installed-conda" in result.stdout
        assert "(not installed)" not in result.stdout

    def test_list_with_data_path_fetched(self, config, invoke_cli, mocker, tmp_path):
        """Test list command shows data path without suffix when fetched."""
        # Create a mock provider with data_path that exists
        data_path = tmp_path / "fetched_data"
        data_path.mkdir()

        mock_provider = mocker.MagicMock(spec=DiagnosticProvider)
        mock_provider.slug = "data-fetched"
        mock_provider.version = "1.0.0"
        mock_provider.get_data_path.return_value = data_path

        mock_registry = mocker.MagicMock(spec=ProviderRegistry)
        mock_registry.providers = [mock_provider]
        mocker.patch.object(ProviderRegistry, "build_from_config", return_value=mock_registry)

        result = invoke_cli(["providers", "list"])
        assert result.exit_code == 0
        assert "data-fetched" in result.stdout
        assert "(not fetched)" not in result.stdout


class TestProvidersCreateEnv:
    def test_create_env(self, config, invoke_cli):
        result = invoke_cli(["providers", "create-env"])
        assert result.exit_code == 0

    def test_create_env_invalid_provider(self, config, invoke_cli):
        invoke_cli(
            [
                "providers",
                "create-env",
                "--provider",
                "nonexistent",
            ],
            expected_exit_code=1,
        )

    def test_create_env_conda_provider(self, config, invoke_cli, mocker, tmp_path):
        """Test create-env with a conda provider."""
        # Create a mock conda provider with all required attributes
        mock_provider = mocker.MagicMock(spec=CondaDiagnosticProvider)
        mock_provider.slug = "conda-test"
        mock_provider.version = "1.0.0"
        mock_provider.env_path = tmp_path / "env"
        mock_provider.get_data_path.return_value = None

        mock_registry = mocker.MagicMock(spec=ProviderRegistry)
        mock_registry.providers = [mock_provider]
        mocker.patch.object(ProviderRegistry, "build_from_config", return_value=mock_registry)

        result = invoke_cli(["providers", "create-env"])
        assert result.exit_code == 0
        mock_provider.create_env.assert_called_once()
        assert "Creating" in result.stderr
        assert "conda-test" in result.stderr


class TestProvidersSetup:
    def test_setup(self, config, invoke_cli):
        """Test that setup command runs without error."""
        result = invoke_cli(["providers", "setup"])
        assert result.exit_code == 0

    def test_setup_provider_filter(self, config, invoke_cli):
        """Test setup with --provider filter."""
        result = invoke_cli(["providers", "setup", "--provider", "example"])
        assert result.exit_code == 0

    def test_setup_invalid_provider(self, config, invoke_cli):
        """Test setup with invalid provider."""
        invoke_cli(
            ["providers", "setup", "--provider", "nonexistent"],
            expected_exit_code=1,
        )

    def test_setup_validate_only(self, config, invoke_cli):
        """Test setup with --validate-only."""
        result = invoke_cli(["providers", "setup", "--validate-only"])
        assert result.exit_code == 0
        assert "valid" in result.stdout or "invalid" in result.stdout

    def test_setup_skip_env(self, config, invoke_cli):
        """Test setup with --skip-env."""
        result = invoke_cli(["providers", "setup", "--skip-env"])
        assert result.exit_code == 0

    def test_setup_skip_data(self, config, invoke_cli):
        """Test setup with --skip-data."""
        result = invoke_cli(["providers", "setup", "--skip-data"])
        assert result.exit_code == 0

    def test_setup_skip_both(self, config, invoke_cli):
        """Test setup with both --skip-env and --skip-data."""
        result = invoke_cli(["providers", "setup", "--skip-env", "--skip-data"])
        assert result.exit_code == 0

    def test_setup_validate_only_provider_filter(self, config, invoke_cli):
        """Test setup with --validate-only and --provider filter."""
        result = invoke_cli(["providers", "setup", "--validate-only", "--provider", "example"])
        assert result.exit_code == 0
        assert "valid" in result.stdout

    def test_setup_skip_validate(self, config, invoke_cli):
        """Test setup with --skip-validate skips validation."""
        result = invoke_cli(["providers", "setup", "--skip-validate"])
        assert result.exit_code == 0
        assert "Skipped validation" in result.stderr

    def test_setup_skip_validate_with_provider(self, config, invoke_cli):
        """Test setup with --skip-validate and --provider."""
        result = invoke_cli(["providers", "setup", "--skip-validate", "--provider", "example"])
        assert result.exit_code == 0
        assert "Skipped validation for provider example" in result.stderr

    def test_setup_validation_failed(self, config, invoke_cli, mocker):
        """Test setup reports validation failure."""
        # Create a mock provider that passes setup but fails validation
        mock_provider = mocker.MagicMock()
        mock_provider.slug = "failing-provider"
        mock_provider.validate_setup.return_value = False

        mock_registry = mocker.MagicMock(spec=ProviderRegistry)
        mock_registry.providers = [mock_provider]
        mocker.patch.object(ProviderRegistry, "build_from_config", return_value=mock_registry)

        result = invoke_cli(["providers", "setup"], expected_exit_code=1)
        assert "validation failed" in result.stderr
        assert "Setup failed for providers" in result.stderr

    def test_setup_exception_handling(self, config, invoke_cli, mocker):
        """Test setup handles exceptions from provider.setup()."""
        # Create a mock provider that raises an exception during setup
        mock_provider = mocker.MagicMock()
        mock_provider.slug = "error-provider"
        mock_provider.setup.side_effect = RuntimeError("Setup exploded")

        mock_registry = mocker.MagicMock(spec=ProviderRegistry)
        mock_registry.providers = [mock_provider]
        mocker.patch.object(ProviderRegistry, "build_from_config", return_value=mock_registry)

        result = invoke_cli(["providers", "setup"], expected_exit_code=1)
        assert "Failed to setup provider error-provider" in result.stderr
        assert "Setup failed for providers" in result.stderr

    def test_setup_validate_only_invalid_provider_fails(self, config, invoke_cli, mocker):
        """Test --validate-only exits with code 1 when provider is invalid."""
        mock_provider = mocker.MagicMock()
        mock_provider.slug = "invalid-provider"
        mock_provider.validate_setup.return_value = False

        mock_registry = mocker.MagicMock(spec=ProviderRegistry)
        mock_registry.providers = [mock_provider]
        mocker.patch.object(ProviderRegistry, "build_from_config", return_value=mock_registry)

        result = invoke_cli(["providers", "setup", "--validate-only"], expected_exit_code=1)
        assert "invalid" in result.stdout


class TestProvidersShow:
    def test_show_example_provider(self, config, invoke_cli):
        """Test show command displays diagnostic info (default list format)."""
        result = invoke_cli(["providers", "show", "--provider", "example"])
        assert result.exit_code == 0
        assert "Global" in result.stdout
        assert "global-mean-timeseries" in result.stdout
        # Default is list format, shows detailed fields
        assert "Facets:" in result.stdout
        assert "Source type:" in result.stdout
        # Should show variables
        assert "tas" in result.stdout

    def test_show_nonexistent_provider(self, config, invoke_cli):
        """Test show command exits with code 1 for unknown provider."""
        invoke_cli(
            ["providers", "show", "--provider", "nonexistent"],
            expected_exit_code=1,
        )

    def test_show_displays_or_options(self, config, invoke_cli):
        """Test show command displays OR-logic options for the example provider."""
        result = invoke_cli(["providers", "show", "--provider", "example"])
        assert result.exit_code == 0
        # Example provider has OR-logic (CMIP6 and CMIP7 options)
        assert "Option 1" in result.stdout
        assert "Option 2" in result.stdout

    def test_show_empty_provider(self, config, invoke_cli, mocker):
        """Test show command handles provider with no diagnostics."""
        mock_provider = mocker.MagicMock(spec=DiagnosticProvider)
        mock_provider.slug = "empty"
        mock_provider.name = "Empty"
        mock_provider.version = "0.0.0"
        mock_provider.diagnostics.return_value = []

        mock_registry = mocker.MagicMock(spec=ProviderRegistry)
        mock_registry.providers = [mock_provider]
        mock_registry.get.return_value = mock_provider
        mocker.patch.object(ProviderRegistry, "build_from_config", return_value=mock_registry)

        result = invoke_cli(["providers", "show", "--provider", "empty"])
        assert result.exit_code == 0
        assert "no registered diagnostics" in result.stdout

    def test_show_list_format(self, config, invoke_cli):
        """Test show command with --format list displays detailed output."""
        result = invoke_cli(["providers", "show", "--provider", "example", "--format", "list"])
        assert result.exit_code == 0
        # List format shows provider name and version
        assert "Example" in result.stdout
        # List format shows slug, facets, group by
        assert "global-mean-timeseries" in result.stdout
        assert "Facets:" in result.stdout
        assert "Group by:" in result.stdout
        assert "Source type:" in result.stdout
        assert "Variables:" in result.stdout

    def test_show_table_format_explicit(self, config, invoke_cli):
        """Test show command with --format table displays table output."""
        result = invoke_cli(["providers", "show", "--provider", "example", "--format", "table"])
        assert result.exit_code == 0
        assert "global-mean-timeseries" in result.stdout
        assert "cmip6" in result.stdout

    def test_show_table_columns_filter(self, config, invoke_cli):
        """Test show command with --columns filters table output."""
        result = invoke_cli(
            [
                "providers",
                "show",
                "--provider",
                "example",
                "--format",
                "table",
                "--columns",
                "diagnostic",
                "--columns",
                "variables",
            ]
        )
        assert result.exit_code == 0
        assert "tas" in result.stdout
        # Filtered columns should not include slug
        assert "slug" not in result.stdout.lower().split("\n")[0]

    def test_show_table_columns_invalid(self, config, invoke_cli):
        """Test show command with invalid --columns exits with error."""
        invoke_cli(
            [
                "providers",
                "show",
                "--provider",
                "example",
                "--format",
                "table",
                "--columns",
                "nonexistent",
            ],
            expected_exit_code=1,
        )
