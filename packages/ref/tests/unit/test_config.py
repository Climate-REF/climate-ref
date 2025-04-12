import logging
import sys
from pathlib import Path

import pytest
from attr import evolve
from cattrs import IterableValidationError

from cmip_ref.config import Config, PathConfig, transform_error
from cmip_ref_core.exceptions import InvalidExecutorException
from cmip_ref_core.executor import Executor


class TestConfig:
    def test_load_missing(self, tmp_path, monkeypatch):
        ref_configuration_value = str(tmp_path / "cmip_ref")
        monkeypatch.setenv("REF_CONFIGURATION", ref_configuration_value)

        # The configuration file doesn't exist
        # so it should default to some sane defaults
        assert not (tmp_path / "ref.toml").exists()

        loaded = Config.load(Path("ref.toml"))

        assert loaded.paths.log == tmp_path / "cmip_ref" / "log"
        assert loaded.paths.scratch == tmp_path / "cmip_ref" / "scratch"
        assert loaded.paths.results == tmp_path / "cmip_ref" / "results"
        assert loaded.db.database_url == f"sqlite:///{ref_configuration_value}/db/cmip_ref.db"

        # The results aren't serialised back to disk
        assert not (tmp_path / "ref.toml").exists()
        assert loaded._raw is None
        assert loaded._config_file == Path("ref.toml")

    def test_default(self, config):
        config.paths.scratch = Path("data")
        config.save()

        # The default location is overridden in the config fixture
        loaded = Config.default()
        assert loaded.paths.scratch == Path("data").resolve()

    def test_load(self, config, tmp_path):
        res = config.dump(defaults=True)

        with open(tmp_path / "ref.toml", "w") as fh:
            fh.write(res.as_string())

        loaded = Config.load(tmp_path / "ref.toml")

        assert config.dumps() == loaded.dumps()

    def test_load_extra_keys(self, tmp_path, caplog):
        content = """[paths]
data = "data"
extra_key = "extra"
another_key = "extra"

[db]
filename = "sqlite://cmip_ref.db"
"""

        with open(tmp_path / "ref.toml", "w") as fh:
            fh.write(content)

        with caplog.at_level(logging.WARNING):
            Config.load(tmp_path / "ref.toml")

        assert len(caplog.records) == 2
        # The order for multiple keys isn't stable
        assert "@ $.paths" in caplog.records[0].message
        assert "extra_key" in caplog.records[0].message
        assert "another_key" in caplog.records[0].message
        assert "extra fields found (filename) @ $.db" in caplog.records[1].message

        for record in caplog.records:
            assert record.levelname == "WARNING"

    def test_invalid(self, tmp_path, caplog):
        content = """[paths]
    scratch = 1

    [db]
    filename = "sqlite://cmip_ref.db"
    """

        with open(tmp_path / "ref.toml", "w") as fh:
            fh.write(content)

        with caplog.at_level(logging.WARNING):
            with pytest.raises(ValueError, match=f"Error loading configuration from {tmp_path / 'ref.toml'}"):
                Config.load(tmp_path / "ref.toml")

        assert len(caplog.records) == 2
        assert "extra fields found (filename) @ $.db" in caplog.records[0].message
        assert caplog.records[0].levelname == "WARNING"

        if sys.version_info >= (3, 12):
            expected_msg = (
                "argument should be a str or an os.PathLike object where __fspath__ returns a str, "
                "not 'Integer'"
            )
        else:
            expected_msg = "expected str, bytes or os.PathLike object, not Integer"
        assert f"invalid type ({expected_msg}) @ $.paths.scratch" in caplog.records[1].message
        assert caplog.records[1].levelname == "ERROR"

    def test_save(self, tmp_path):
        config = Config(paths=PathConfig(scratch=Path("scratch")))

        with pytest.raises(ValueError):
            # The configuration file hasn't been set as it was created directly
            config.save()

        config.save(tmp_path / "ref.toml")

        assert (tmp_path / "ref.toml").exists()

    def test_defaults(self, monkeypatch):
        monkeypatch.setenv("REF_CONFIGURATION", "test")

        cfg = Config.load(Path("test.toml"))
        default_path = Path("test").resolve()

        with_defaults = cfg.dump(defaults=True)

        without_defaults = cfg.dump(defaults=False)

        assert without_defaults == {
            "log_level": "INFO",
            "metric_providers": [
                {"provider": "cmip_ref_metrics_esmvaltool.provider"},
                {"provider": "cmip_ref_metrics_ilamb.provider"},
                {"provider": "cmip_ref_metrics_pmp.provider"},
            ],
        }
        assert with_defaults == {
            "log_level": "INFO",
            "metric_providers": [
                {
                    "provider": "cmip_ref_metrics_esmvaltool.provider",
                    "config": {},
                },
                {
                    "provider": "cmip_ref_metrics_ilamb.provider",
                    "config": {},
                },
                {
                    "provider": "cmip_ref_metrics_pmp.provider",
                    "config": {},
                },
            ],
            "executor": {"executor": "cmip_ref.executor.local.LocalExecutor", "config": {}},
            "paths": {
                "log": f"{default_path}/log",
                "results": f"{default_path}/results",
                "scratch": f"{default_path}/scratch",
                "software": f"{default_path}/software",
            },
            "db": {"database_url": "sqlite:///test/db/cmip_ref.db", "run_migrations": True},
        }

    def test_from_env_variables(self, monkeypatch, config):
        monkeypatch.setenv("REF_DATABASE_URL", "test-database")
        monkeypatch.setenv("REF_EXECUTOR", "new-executor")
        monkeypatch.setenv("REF_SCRATCH_ROOT", "/my/test/scratch")
        monkeypatch.setenv("REF_LOG_ROOT", "/my/test/logs")
        monkeypatch.setenv("REF_RESULTS_ROOT", "/my/test/results")

        config_new = config.refresh()

        assert config_new.db.database_url == "test-database"
        assert config_new.executor.executor == "new-executor"
        assert config_new.paths.scratch == Path("/my/test/scratch")
        assert config_new.paths.log == Path("/my/test/logs")
        assert config_new.paths.results == Path("/my/test/results")

    def test_executor_build(self, config, db):
        executor = config.executor.build(config, db)
        assert executor.name == "local"
        assert isinstance(executor, Executor)

    @pytest.mark.skipif(
        sys.version_info > (3, 11),
        reason="isinstance check on mock executor fails with Python 3.12+",
    )
    def test_executor_build_config(self, mocker, config, db):
        mock_executor = mocker.MagicMock(spec=Executor)
        mocker.patch("cmip_ref.config.import_executor_cls", return_value=mock_executor)

        executor = config.executor.build(config, db)
        assert executor == mock_executor.return_value
        mock_executor.assert_called_once_with(config=config, database=db)

    @pytest.mark.skipif(
        sys.version_info > (3, 11),
        reason="isinstance check on mock executor fails with Python 3.12+",
    )
    def test_executor_build_extra_config(self, mocker, config, db):
        mock_executor = mocker.MagicMock(spec=Executor)
        mocker.patch("cmip_ref.config.import_executor_cls", return_value=mock_executor)

        config.executor = evolve(config.executor, config={"extra": 1})

        executor = config.executor.build(config, db)
        assert executor == mock_executor.return_value
        mock_executor.assert_called_once_with(config=config, database=db, extra=1)

    def test_executor_build_invalid(self, config, db, mocker):
        config.executor = evolve(config.executor, executor="cmip_ref.config.DbConfig")

        class NotAnExecutor:
            def __init__(self, **kwargs): ...

        mocker.patch("cmip_ref.config.import_executor_cls", return_value=NotAnExecutor)

        match = r"Expected an Executor, got <class '.*\.NotAnExecutor'>"
        with pytest.raises(InvalidExecutorException, match=match):
            config.executor.build(config, db)


def test_transform_error():
    assert transform_error(ValueError("Test error"), "test") == ["invalid value @ test"]

    err = IterableValidationError("Validation error", [ValueError("Test error"), KeyError()], Config)
    assert transform_error(err, "test") == ["invalid value @ test", "required field missing @ test"]
