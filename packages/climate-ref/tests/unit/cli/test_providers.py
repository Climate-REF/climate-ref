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
