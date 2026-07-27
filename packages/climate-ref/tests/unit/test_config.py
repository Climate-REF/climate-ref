import datetime
import importlib.metadata
import logging
import os
import stat
import sys
from datetime import timedelta
from pathlib import Path

import pytest
import requests
from attr import evolve
from cattrs import IterableValidationError

import climate_ref.config
from climate_ref.config import (
    DEFAULT_IGNORE_DATASETS_URL,
    DEFAULT_LOG_FORMAT,
    Config,
    PathConfig,
    _ignore_datasets_cache_file,
    _legacy_ignore_datasets_file,
    refresh_ignore_datasets_file,
    transform_error,
)
from climate_ref_core.data import ResourceOrigin, resolve_cache_dir
from climate_ref_core.exceptions import InvalidExecutorException
from climate_ref_core.executor import Executor


class TestConfig:
    def test_load_missing(self, tmp_path, monkeypatch):
        ref_configuration_value = str(tmp_path / "climate_ref")
        monkeypatch.setenv("REF_CONFIGURATION", ref_configuration_value)

        # The configuration file doesn't exist
        # so it should default to some sane defaults
        assert not (tmp_path / "ref.toml").exists()

        loaded = Config.load(Path("ref.toml"))

        assert loaded.paths.log == tmp_path / "climate_ref" / "log"
        assert loaded.paths.scratch == tmp_path / "climate_ref" / "scratch"
        assert loaded.paths.results == tmp_path / "climate_ref" / "results"
        assert loaded.db.database_url == f"sqlite:///{ref_configuration_value}/db/climate_ref.db"

        # The executions aren't serialised back to disk
        assert not (tmp_path / "ref.toml").exists()
        assert loaded._raw is None
        assert loaded._config_file == Path("ref.toml")

    def test_default(self, config):
        config.paths.scratch = Path("data")
        config.save()

        # The default location is overridden in the config fixture
        loaded = Config.default()
        assert loaded.paths.scratch == Path("data").resolve()

    def test_default_no_network(self, tmp_path, monkeypatch, mocker):
        # Config.default() must not perform any network I/O or filesystem writes when
        # resolving the ignore datasets file; the fetch is deferred to solve time.
        monkeypatch.setenv("REF_CONFIGURATION", str(tmp_path / "climate_ref"))
        assert not (tmp_path / "climate_ref" / "ref.toml").exists()

        cache_dir = tmp_path / "cache" / "climate_ref"
        monkeypatch.setenv("REF_DATASET_CACHE_DIR", str(cache_dir))
        get_mock = mocker.patch.object(climate_ref.config.requests, "get")

        loaded = Config.default()

        get_mock.assert_not_called()
        assert not cache_dir.exists()
        # Nothing is configured, so the copy shipped in the package is used.
        assert loaded.ignore_datasets_file is None
        assert loaded.ignore_datasets_resource.origin == ResourceOrigin.package
        assert loaded.paths.dimensions_cv is None
        assert loaded.paths.dimensions_cv_resource.origin == ResourceOrigin.package
        assert loaded.paths.scratch == tmp_path / "climate_ref" / "scratch"

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
filename = "sqlite://climate_ref.db"
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
    filename = "sqlite://climate_ref.db"
    """

        with open(tmp_path / "ref.toml", "w") as fh:
            fh.write(content)

        with caplog.at_level(logging.WARNING):
            with pytest.raises(ValueError, match=f"Error loading configuration from {tmp_path / 'ref.toml'}"):
                Config.load(tmp_path / "ref.toml")

        assert len(caplog.records) == 2
        assert "extra fields found (filename) @ $.db" in caplog.records[0].message
        assert caplog.records[0].levelname == "WARNING"

        expected_msg = (
            "argument should be a str or an os.PathLike object where __fspath__ returns a str, not 'Integer'"
        )
        assert f"invalid type ({expected_msg}) @ $.paths.scratch" in caplog.records[1].message
        assert caplog.records[1].levelname == "ERROR"

    @pytest.mark.parametrize("n_jobs", [0, -2])
    def test_invalid_n_jobs_rejected(self, n_jobs):
        """An unusable n_jobs must fail loudly rather than silently parse serially."""
        with pytest.raises(ValueError, match="n_jobs must be -1"):
            Config(n_jobs=n_jobs)

    @pytest.mark.parametrize("n_jobs", [-1, 1, 32])
    def test_valid_n_jobs_accepted(self, n_jobs):
        assert Config(n_jobs=n_jobs).n_jobs == n_jobs

    def test_invalid_n_jobs_from_environment_rejected(self, monkeypatch):
        monkeypatch.setenv("REF_N_JOBS", "0")

        with pytest.raises(ValueError, match="n_jobs must be -1"):
            Config()

    def test_save(self, tmp_path):
        config = Config(paths=PathConfig(scratch=Path("scratch")))

        with pytest.raises(ValueError):
            # The configuration file hasn't been set as it was created directly
            config.save()

        config.save(tmp_path / "ref.toml")

        assert (tmp_path / "ref.toml").exists()

    def test_defaults(self, monkeypatch, mocker):
        monkeypatch.setenv("REF_CONFIGURATION", "test")
        # Clear any externally set env vars that would affect the test
        monkeypatch.delenv("REF_SOFTWARE_ROOT", raising=False)
        mocker.patch(
            "climate_ref.config.importlib.metadata.entry_points",
            return_value=importlib.metadata.EntryPoints(
                [
                    importlib.metadata.EntryPoint(
                        name="example",
                        value="climate_ref_example:provider",
                        group="climate-ref.providers",
                    ),
                ]
            ),
        )

        cfg = Config.load(Path("test.toml"))
        default_path = Path("test").resolve()

        with_defaults = cfg.dump(defaults=True)

        without_defaults = cfg.dump(defaults=False)

        assert without_defaults == {
            "ignore_datasets_url": DEFAULT_IGNORE_DATASETS_URL,
            "log_level": "INFO",
            "log_format": DEFAULT_LOG_FORMAT,
            "cmip6_parser": "complete",
            "cmip7_parser": "complete",
            "n_jobs": 1,
            "diagnostic_providers": [
                {"provider": "climate_ref_example:provider"},
            ],
        }
        assert with_defaults == {
            "ignore_datasets_url": DEFAULT_IGNORE_DATASETS_URL,
            "log_level": "INFO",
            "log_format": DEFAULT_LOG_FORMAT,
            "cmip6_parser": "complete",
            "cmip7_parser": "complete",
            "n_jobs": 1,
            "diagnostic_providers": [
                {
                    "provider": "climate_ref_example:provider",
                    "config": {},
                },
            ],
            "executor": {"executor": "climate_ref.executor.LocalExecutor", "config": {}},
            "native_store": {
                "url": "https://baselines.climate-ref.org",
                "s3_endpoint_url": "https://2aa5172b2bba093c516027d6fa13cdc8.r2.cloudflarestorage.com",
                "bucket": "ref-baselines-public",
                "cache_dir": str(resolve_cache_dir("native-baselines")),
            },
            "paths": {
                "log": f"{default_path}/log",
                "results": f"{default_path}/results",
                "scratch": f"{default_path}/scratch",
                "software": f"{default_path}/software",
            },
            "db": {
                "database_url": "sqlite:///test/db/climate_ref.db",
                "max_backups": 5,
                "run_migrations": True,
            },
        }

    def test_from_env_variables(self, monkeypatch, config):
        monkeypatch.setenv("REF_DATABASE_URL", "test-database")
        monkeypatch.setenv("REF_EXECUTOR", "new-executor")
        monkeypatch.setenv("REF_SCRATCH_ROOT", "/my/test/scratch")
        monkeypatch.setenv("REF_LOG_ROOT", "/my/test/logs")
        monkeypatch.setenv("REF_RESULTS_ROOT", "/my/test/executions")
        monkeypatch.setenv("REF_CMIP6_PARSER", "drs")

        config_new = config.refresh()

        assert config_new.db.database_url == "test-database"
        assert config_new.executor.executor == "new-executor"
        assert config_new.paths.scratch == Path("/my/test/scratch")
        assert config_new.paths.log == Path("/my/test/logs")
        assert config_new.paths.results == Path("/my/test/executions")
        assert config_new.cmip6_parser == "drs"

    def test_ignore_datasets_env_variables(self, monkeypatch, config):
        monkeypatch.setenv("REF_IGNORE_DATASETS_FILE", "/my/test/ignore_datasets.yaml")
        monkeypatch.setenv("REF_IGNORE_DATASETS_URL", "")

        config_new = config.refresh()

        # Env overrides for ignore_datasets_file must be coerced to Path so callers
        # can rely on `.is_file()` / `.read_text()` rather than getting a bare str.
        assert config_new.ignore_datasets_file == Path("/my/test/ignore_datasets.yaml")
        assert isinstance(config_new.ignore_datasets_file, Path)
        assert config_new.ignore_datasets_url == ""

    def test_ignore_datasets_url_env_override(self, monkeypatch, config):
        monkeypatch.setenv("REF_IGNORE_DATASETS_URL", "https://example.invalid/fork.yaml")

        config_new = config.refresh()

        assert config_new.ignore_datasets_url == "https://example.invalid/fork.yaml"

    def test_custom_env_variable(self, monkeypatch, tmp_path, config):
        monkeypatch.setenv("ABC", "/my")
        config.paths.results = "${ABC}/test/executions"
        # Environment variables are only expanded when loading from file.
        config.save(tmp_path / "ref.toml")
        config_new = Config.load(tmp_path / "ref.toml")
        assert config_new.paths.results == Path("/my/test/executions")

    def test_executor_build(self, config, db):
        executor = config.executor.build(config, db)
        assert executor.name == "synchronous"
        assert isinstance(executor, Executor)

    @pytest.mark.skipif(
        sys.version_info > (3, 11),
        reason="isinstance check on mock executor fails with Python 3.12+",
    )
    def test_executor_build_config(self, mocker, config, db):
        mock_executor = mocker.MagicMock(spec=Executor)
        mocker.patch("climate_ref_core.executor.import_executor_cls", return_value=mock_executor)

        executor = config.executor.build(config, db)
        assert executor == mock_executor.return_value
        mock_executor.assert_called_once_with(config=config, database=db)

    @pytest.mark.skipif(
        sys.version_info > (3, 11),
        reason="isinstance check on mock executor fails with Python 3.12+",
    )
    def test_executor_build_extra_config(self, mocker, config, db):
        mock_executor = mocker.MagicMock(spec=Executor)
        mocker.patch("climate_ref_core.executor.import_executor_cls", return_value=mock_executor)

        config.executor = evolve(config.executor, config={"extra": 1})

        executor = config.executor.build(config, db)
        assert executor == mock_executor.return_value
        mock_executor.assert_called_once_with(config=config, database=db, extra=1)

    def test_executor_build_invalid(self, config, db, mocker):
        config.executor = evolve(config.executor, executor="climate_ref.config.DbConfig")

        class NotAnExecutor:
            def __init__(self, **kwargs): ...

        mocker.patch("climate_ref_core.executor.import_executor_cls", return_value=NotAnExecutor)

        match = r"Expected an Executor, got <class '.*\.NotAnExecutor'>"
        with pytest.raises(InvalidExecutorException, match=match):
            config.executor.build(config, db)


def test_transform_error():
    assert transform_error(ValueError("Test error"), "test") == ["invalid value @ test"]

    err = IterableValidationError("Validation error", [ValueError("Test error"), KeyError()], Config)
    assert transform_error(err, "test") == ["invalid value @ test", "required field missing @ test"]


def test_ignore_datasets_cache_file(monkeypatch, tmp_path):
    """The cache location is a pure path computation with no filesystem or network access."""
    monkeypatch.setenv("REF_DATASET_CACHE_DIR", str(tmp_path))

    path = _ignore_datasets_cache_file()

    assert path == tmp_path / "grey_list" / "default_ignore_datasets.yaml"
    assert not path.exists()


def _refresh_config(monkeypatch, tmp_path, url=DEFAULT_IGNORE_DATASETS_URL):
    monkeypatch.setenv("REF_DATASET_CACHE_DIR", str(tmp_path))
    config = Config()
    config.ignore_datasets_url = url
    return config


@pytest.mark.parametrize("status", ["fresh", "stale", "missing"])
def test_refresh_ignore_datasets_file(mocker, monkeypatch, tmp_path, status):
    mocker.patch.object(
        climate_ref.config.requests,
        "get",
        return_value=mocker.MagicMock(status_code=200, content=b"downloaded"),
    )
    config = _refresh_config(monkeypatch, tmp_path / "nested")
    target = _ignore_datasets_cache_file()
    if status != "missing":
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("existing", encoding="utf-8")
    if status == "stale":
        mocker.patch.object(
            climate_ref.config, "DEFAULT_IGNORE_DATASETS_REFRESH_INTERVAL", timedelta(seconds=-1)
        )

    refresh_ignore_datasets_file(config)

    assert target.parent.exists()
    if status == "fresh":
        assert target.read_text(encoding="utf-8") == "existing"
    else:
        assert target.read_text(encoding="utf-8") == "downloaded"
    assert config.ignore_datasets_resource.origin == ResourceOrigin.cache


def test_refresh_ignore_datasets_file_disabled_by_empty_url(mocker, monkeypatch, tmp_path):
    """An empty URL skips the fetch entirely, and the packaged copy is used instead."""
    get_mock = mocker.patch.object(climate_ref.config.requests, "get")
    config = _refresh_config(monkeypatch, tmp_path, url="")

    refresh_ignore_datasets_file(config)

    get_mock.assert_not_called()
    assert not _ignore_datasets_cache_file().exists()
    assert config.ignore_datasets_resource.origin == ResourceOrigin.package


def test_refresh_ignore_datasets_file_skipped_when_file_configured(mocker, monkeypatch, tmp_path):
    """An explicit file is the operator's to manage, so no fetch is attempted."""
    get_mock = mocker.patch.object(climate_ref.config.requests, "get")
    config = _refresh_config(monkeypatch, tmp_path)
    pinned = tmp_path / "pinned.yaml"
    pinned.write_text("{}", encoding="utf-8")
    config.ignore_datasets_file = pinned

    refresh_ignore_datasets_file(config)

    get_mock.assert_not_called()
    assert config.ignore_datasets_resource.origin == ResourceOrigin.override


@pytest.mark.parametrize(
    "failure",
    [
        pytest.param("http", id="http-error"),
        pytest.param("network", id="network-error"),
    ],
)
def test_refresh_ignore_datasets_file_fail_no_cache(mocker, monkeypatch, tmp_path, caplog, failure):
    """A failed download falls back to the packaged copy rather than failing the solve."""
    if failure == "http":
        result = mocker.MagicMock(status_code=404, content=b"{}")
        result.raise_for_status.side_effect = requests.RequestException
        mocker.patch.object(climate_ref.config.requests, "get", return_value=result)
    else:
        mocker.patch.object(
            climate_ref.config.requests,
            "get",
            side_effect=requests.exceptions.ConnectionError("Network unreachable"),
        )
    config = _refresh_config(monkeypatch, tmp_path)

    with caplog.at_level(logging.WARNING):
        refresh_ignore_datasets_file(config)

    assert not _ignore_datasets_cache_file().exists()
    assert config.ignore_datasets_resource.origin == ResourceOrigin.package
    assert any("Could not refresh the grey list" in r.message for r in caplog.records)


def test_refresh_ignore_datasets_file_unwritable_cache(mocker, monkeypatch, tmp_path, caplog):
    """An unwritable cache directory is not fatal, since the packaged copy is always readable."""
    mocker.patch.object(
        climate_ref.config.requests,
        "get",
        return_value=mocker.MagicMock(status_code=200, content=b"downloaded"),
    )
    config = _refresh_config(monkeypatch, tmp_path / "readonly" / "cache")
    (tmp_path / "readonly").mkdir()
    (tmp_path / "readonly").chmod(0o500)

    try:
        with caplog.at_level(logging.WARNING):
            refresh_ignore_datasets_file(config)
    finally:
        (tmp_path / "readonly").chmod(0o700)

    assert config.ignore_datasets_resource.origin == ResourceOrigin.package
    assert any("Could not refresh the grey list" in r.message for r in caplog.records)


def test_refresh_ignore_datasets_file_writes_atomically(mocker, monkeypatch, tmp_path):
    """A failed write must not leave a truncated file behind with a fresh modification time."""
    mocker.patch.object(
        climate_ref.config.requests,
        "get",
        return_value=mocker.MagicMock(status_code=200, content=b"downloaded"),
    )
    mocker.patch.object(climate_ref.config.Path, "replace", side_effect=OSError("disk full"))
    config = _refresh_config(monkeypatch, tmp_path)

    refresh_ignore_datasets_file(config)

    target = _ignore_datasets_cache_file()
    assert not target.exists()
    # The temporary file is cleaned up rather than left in the cache directory.
    assert list(target.parent.iterdir()) == []
    assert config.ignore_datasets_resource.origin == ResourceOrigin.package


def test_refresh_ignore_datasets_file_is_group_readable(mocker, monkeypatch, tmp_path):
    """The cache is shared between users, so it must not inherit the 0600 mkstemp mode."""
    mocker.patch.object(
        climate_ref.config.requests,
        "get",
        return_value=mocker.MagicMock(status_code=200, content=b"downloaded"),
    )
    config = _refresh_config(monkeypatch, tmp_path)

    refresh_ignore_datasets_file(config)

    target = _ignore_datasets_cache_file()
    assert stat.S_IMODE(target.stat().st_mode) == 0o644


def test_stale_cache_does_not_shadow_the_packaged_copy(monkeypatch, tmp_path):
    """A cache left by an older release must not shadow the newer packaged copy forever."""
    monkeypatch.setenv("REF_DATASET_CACHE_DIR", str(tmp_path))
    config = Config()
    target = _ignore_datasets_cache_file()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("ancient", encoding="utf-8")

    assert config.ignore_datasets_resource.origin == ResourceOrigin.cache

    ancient = (
        datetime.datetime.now() - climate_ref.config.DEFAULT_IGNORE_DATASETS_MAX_STALE - timedelta(days=1)
    ).timestamp()
    os.utime(target, (ancient, ancient))

    assert config.ignore_datasets_resource.origin == ResourceOrigin.package


def test_inherited_legacy_ignore_datasets_default_is_not_an_override(monkeypatch, tmp_path, mocker):
    """Releases up to 0.16 baked the cache path into ref.toml, which must not pin the grey list."""
    monkeypatch.setenv("REF_DATASET_CACHE_DIR", str(tmp_path))
    get_mock = mocker.patch.object(
        climate_ref.config.requests,
        "get",
        return_value=mocker.MagicMock(status_code=200, content=b"downloaded"),
    )
    legacy = _legacy_ignore_datasets_file()
    legacy.parent.mkdir(parents=True, exist_ok=True)
    legacy.write_text("stale", encoding="utf-8")

    config_file = tmp_path / "ref.toml"
    config_file.write_text(f'ignore_datasets_file = "{legacy}"\n', encoding="utf-8")

    config = Config.load(config_file)

    # Cleared at load, so it is not reported, not saved, and does not disable refreshing.
    assert config.ignore_datasets_file is None

    refresh_ignore_datasets_file(config)

    get_mock.assert_called_once()
    assert config.ignore_datasets_resource.origin == ResourceOrigin.cache
    assert config.ignore_datasets_resource.read_text() == "downloaded"

    config.save(config_file)
    assert "ignore_datasets_file" not in config_file.read_text()


def test_refresh_ignore_datasets_file_fail_uses_stale_cache(mocker, monkeypatch, tmp_path, caplog):
    """A download failure must preserve and reuse an existing cached copy without touching it."""
    result = mocker.MagicMock(status_code=500, content=b"{}")
    result.raise_for_status.side_effect = requests.RequestException
    mocker.patch.object(climate_ref.config.requests, "get", return_value=result)
    config = _refresh_config(monkeypatch, tmp_path)
    target = _ignore_datasets_cache_file()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("cached", encoding="utf-8")
    original_mtime = target.stat().st_mtime
    mocker.patch.object(climate_ref.config, "DEFAULT_IGNORE_DATASETS_REFRESH_INTERVAL", timedelta(seconds=-1))

    with caplog.at_level(logging.WARNING):
        refresh_ignore_datasets_file(config)

    assert target.read_text(encoding="utf-8") == "cached"
    assert target.stat().st_mtime == original_mtime
    assert config.ignore_datasets_resource.origin == ResourceOrigin.cache
    assert any("Could not refresh the grey list" in r.message for r in caplog.records)
