class TestProvidersList:
    def test_list(self, config, invoke_cli):
        result = invoke_cli(["providers", "list"])
        assert result.exit_code == 0
        assert "provider" in result.stdout
        assert "example" in result.stdout


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
