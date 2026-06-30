import json
from pathlib import Path

import pytest

from climate_ref.cli._config_access import coerce_value, resolve_key
from climate_ref.config import Config
from climate_ref.constants import CONFIG_FILENAME


def test_without_subcommand(invoke_cli):
    # exit code 2 denotes a user error
    result = invoke_cli(["config"], expected_exit_code=2)
    assert "Missing command." in result.stderr


def test_config_help(invoke_cli):
    result = invoke_cli(["config", "--help"])

    assert "View and update the REF configuration" in result.stdout


class TestConfigList:
    def test_config_list(self, config, invoke_cli):
        result = invoke_cli(["config", "list"])

        assert 'database_url = "sqlite://' in result.output

    def test_config_list_custom_missing(self, config, invoke_cli):
        result = invoke_cli(
            [
                "--configuration-directory",
                "missing",
                "config",
                "list",
            ],
            expected_exit_code=1,
        )

        assert "Configuration file not found" in result.stdout

    def test_config_list_json(self, config, invoke_cli):
        result = invoke_cli(["config", "list", "--format", "json"])

        # Output must be valid JSON (the TOML structure rendered as nested objects).
        payload = json.loads(result.stdout)
        assert isinstance(payload, dict)
        assert "sqlite://" in json.dumps(payload)


class TestConfigInit:
    def test_init_create_from_scratch(self, tmp_path, invoke_cli):
        config_dir = tmp_path / "new-config"

        result = invoke_cli(["--configuration-directory", str(config_dir), "config", "init"])

        assert (config_dir / CONFIG_FILENAME).is_file()
        assert "Wrote configuration file" in result.stdout
        assert "ref config validate" in result.stdout

    def test_init_refuses_existing_file(self, config, invoke_cli):
        result = invoke_cli(["config", "init"], expected_exit_code=1)

        assert "Configuration already exists" in result.stdout
        assert "--force" in result.stdout

    def test_init_force_overwrites(self, config, invoke_cli):
        assert config._config_file is not None
        config._config_file.write_text("not valid toml")

        result = invoke_cli(["config", "init", "--force"])

        assert "Overwrote configuration file" in result.stdout
        assert "not valid toml" not in config._config_file.read_text()

    def test_init_no_defaults_writes_smaller_file(self, tmp_path, invoke_cli):
        full_dir = tmp_path / "full"
        minimal_dir = tmp_path / "minimal"

        invoke_cli(["--configuration-directory", str(full_dir), "config", "init"])
        invoke_cli(["--configuration-directory", str(minimal_dir), "config", "init", "--no-defaults"])

        assert (minimal_dir / CONFIG_FILENAME).stat().st_size < (full_dir / CONFIG_FILENAME).stat().st_size

    def test_init_output_validates(self, tmp_path, invoke_cli):
        config_dir = tmp_path / "new-config"

        invoke_cli(["--configuration-directory", str(config_dir), "config", "init"])
        result = invoke_cli(["--configuration-directory", str(config_dir), "config", "validate"])

        assert "is valid" in result.stdout


class TestConfigGet:
    def test_get_scalar(self, invoke_cli):
        result = invoke_cli(["config", "get", "db.database_url"])

        assert "sqlite://" in result.stdout
        assert result.stdout.strip().startswith("sqlite://")

    def test_get_top_level_scalar(self, invoke_cli):
        result = invoke_cli(["config", "get", "log_level"])

        assert result.stdout.strip() == "INFO"

    def test_get_subtable_as_toml(self, invoke_cli):
        result = invoke_cli(["config", "get", "paths"])

        assert "[paths]" in result.stdout
        assert "scratch" in result.stdout

    def test_get_unknown_key(self, invoke_cli):
        result = invoke_cli(["config", "get", "not.real"], expected_exit_code=1)

        assert "Unknown configuration key: not.real" in result.stderr

    def test_get_warns_when_env_overrides(self, monkeypatch, invoke_cli):
        monkeypatch.setenv("REF_DATABASE_URL", "sqlite:///env.db")

        result = invoke_cli(["config", "get", "db.database_url"])

        assert result.stdout.strip() == "sqlite:///env.db"
        assert "REF_DATABASE_URL" in result.stderr


class TestConfigSet:
    def test_set_scalar_and_persist(self, config, invoke_cli):
        assert config._config_file is not None

        invoke_cli(["config", "set", "log_level", "DEBUG"])
        result = invoke_cli(["config", "get", "log_level"])

        assert result.stdout.strip() == "DEBUG"
        assert 'log_level = "DEBUG"' in config._config_file.read_text()

    def test_set_invalid_literal(self, invoke_cli):
        result = invoke_cli(["config", "set", "cmip6_parser", "bogus"], expected_exit_code=1)

        assert "Invalid value for cmip6_parser" in result.stderr

    def test_set_unknown_key(self, invoke_cli):
        result = invoke_cli(["config", "set", "not.real", "x"], expected_exit_code=1)

        assert "Unknown configuration key: not.real" in result.stderr

    def test_set_rejects_structured_key(self, invoke_cli):
        result = invoke_cli(["config", "set", "diagnostic_providers", "x"], expected_exit_code=1)

        assert "Editing structured configuration values" in result.stderr

    def test_set_warns_when_env_overrides(self, monkeypatch, invoke_cli):
        monkeypatch.setenv("REF_DATABASE_URL", "sqlite:///env.db")

        result = invoke_cli(["config", "set", "db.database_url", "sqlite:///file.db"])

        assert "REF_DATABASE_URL" in result.stderr
        assert "db.database_url = sqlite:///file.db" in result.stdout


class TestConfigUnset:
    def test_unset_resets_to_default(self, invoke_cli):
        invoke_cli(["config", "set", "log_level", "DEBUG"])
        invoke_cli(["config", "unset", "log_level"])
        result = invoke_cli(["config", "get", "log_level"])

        assert result.stdout.strip() == "INFO"

    def test_unset_unknown_key(self, invoke_cli):
        result = invoke_cli(["config", "unset", "not.real"], expected_exit_code=1)

        assert "Unknown configuration key: not.real" in result.stderr

    def test_unset_rejects_structured_key(self, invoke_cli):
        result = invoke_cli(["config", "unset", "diagnostic_providers"], expected_exit_code=1)

        assert "Editing structured configuration values" in result.stderr


class TestConfigEdit:
    def test_edit_runs_editor_and_validates(self, monkeypatch, config, invoke_cli):
        assert config._config_file is not None
        edited: list[str] = []

        def fake_edit(*, filename: str) -> None:
            edited.append(filename)

        monkeypatch.setattr("climate_ref.cli.config.click.edit", fake_edit)

        result = invoke_cli(["config", "edit"])

        assert edited == [str(config._config_file)]
        assert "Edited" in result.stdout

    def test_edit_warns_when_result_invalid(self, monkeypatch, config, invoke_cli):
        assert config._config_file is not None

        def fake_edit(*, filename: str) -> None:
            Path(filename).write_text("unknown = true\n")

        monkeypatch.setattr("climate_ref.cli.config.click.edit", fake_edit)

        result = invoke_cli(["config", "edit"])

        assert result.exit_code == 0
        assert "invalid after editing" in result.stderr
        assert "unknown" in result.stderr


class TestConfigValidate:
    def test_validate_valid_config(self, invoke_cli):
        result = invoke_cli(["config", "validate"])

        assert "is valid" in result.stdout

    def test_validate_invalid_config_runs_despite_bad_file(self, tmp_path, invoke_cli):
        config_dir = tmp_path / "bad-config"
        config_dir.mkdir()
        (config_dir / CONFIG_FILENAME).write_text("unexpected = true\n")

        result = invoke_cli(
            ["--configuration-directory", str(config_dir), "config", "validate"],
            expected_exit_code=1,
        )

        assert "is invalid" in result.stdout
        assert "unexpected" in result.stdout

    def test_validate_missing_config(self, tmp_path, invoke_cli):
        result = invoke_cli(
            ["--configuration-directory", str(tmp_path / "missing"), "config", "validate"],
            expected_exit_code=1,
        )

        assert "No configuration file found" in result.stdout

    def test_validate_valid_json(self, invoke_cli):
        result = invoke_cli(["config", "validate", "--format", "json"])

        payload = json.loads(result.stdout)
        assert payload == {"valid": True, "error_count": 0, "diagnostics": []}

    def test_validate_invalid_json(self, tmp_path, invoke_cli):
        config_dir = tmp_path / "bad-config"
        config_dir.mkdir()
        (config_dir / CONFIG_FILENAME).write_text("unexpected = true\n")

        result = invoke_cli(
            ["--configuration-directory", str(config_dir), "config", "validate", "--format", "json"],
            expected_exit_code=1,
        )

        payload = json.loads(result.stdout)
        assert payload["valid"] is False
        assert payload["error_count"] >= 1
        assert "unexpected" in payload["diagnostics"][0]["message"]


class TestConfigAccessHelpers:
    def test_resolve_key(self):
        config = Config()

        parent, field, value = resolve_key(config, "paths.scratch")

        assert field.name == "scratch"
        assert getattr(parent, field.name) == value

    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("false", False),
            ("0", False),
            ("yes", True),
        ],
    )
    def test_coerce_bool(self, raw, expected):
        config = Config()
        _, field, _ = resolve_key(config, "db.run_migrations")

        assert coerce_value(field, raw) is expected

    def test_coerce_path(self):
        config = Config()
        _, field, _ = resolve_key(config, "paths.scratch")

        assert coerce_value(field, "relative") == Path("relative").resolve()

    def test_coerce_literal_rejects_unknown_value(self):
        config = Config()
        _, field, _ = resolve_key(config, "cmip6_parser")

        with pytest.raises(Exception):
            coerce_value(field, "bogus")
