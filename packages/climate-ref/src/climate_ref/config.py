"""
Configuration management

The REF uses a tiered configuration model,
where configuration is sourced from a hierarchy of different places.

Each configuration value has a default which is used if not other configuration is available.
Then configuration is loaded from a `.toml` file which overrides any default values.
Finally, some configuration can be overridden at runtime using environment variables,
which always take precedence over any other configuration values.
"""

# The basics of the configuration management takes a lot of inspiration from the
# `esgpull` configuration management system with some of the extra complexity removed.
# https://github.com/ESGF/esgf-download/blob/main/esgpull/config.py

import datetime
import importlib.metadata
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import platformdirs
import requests
import tomlkit
from attr import Factory
from attrs import define, field
from cattrs import Converter
from cattrs.gen import make_dict_unstructure_fn, override
from loguru import logger
from tomlkit import TOMLDocument

from climate_ref._config_helpers import (
    _environ_post_init,
    _format_exception,
    _format_key_exception,
    _pop_empty,
    _pop_none,
    config,
    env_field,
    transform_error,
)
from climate_ref.constants import CONFIG_FILENAME
from climate_ref_core.data import LayeredResource, PackagedResource, resolve_cache_dir
from climate_ref_core.env import env
from climate_ref_core.exceptions import InvalidExecutorException
from climate_ref_core.logging import DEFAULT_LOG_FORMAT
from climate_ref_core.pycmec.controlled_vocabulary import BUNDLED_CV

if TYPE_CHECKING:
    from climate_ref.database import Database
    from climate_ref_core.executor import Executor

env_prefix = "REF"
"""
Prefix for the environment variables used by the REF
"""


def ensure_absolute_path(path: str | Path) -> Path:
    """
    Ensure that the path is absolute

    Parameters
    ----------
    path
        Path to check

    Returns
    -------
        Absolute path
    """
    if isinstance(path, str):
        path = Path(path)
    path = Path(*[os.path.expandvars(p) for p in path.parts])
    return path.resolve()


def _optional_path(path: str | Path | None) -> Path | None:
    """
    Convert a value to a path, treating an unset or empty value as "not configured"

    An empty string is treated as unset so that an environment variable can clear
    a value that is set in the configuration file.

    Parameters
    ----------
    path
        Path to convert

    Returns
    -------
        The path, or None if no path was configured
    """
    if path is None:
        return None
    if isinstance(path, str):
        if not path.strip():
            return None
        path = Path(path)
    return Path(os.path.expandvars(str(path))).expanduser()


@config(prefix=env_prefix)
class PathConfig:
    """
    Common paths used by the REF application

    /// admonition | Warning
        type: warning

    These paths must be common across all systems that the REF is being run.
    Generally, this means that they should be mounted in the same location on all systems.
    ///

    If any of these paths are specified as relative paths,
    they will be resolved to absolute paths.
    These absolute paths will be used for all operations in the REF.
    """

    log: Path = env_field(name="LOG_ROOT", converter=ensure_absolute_path)
    """
    Directory to store log files from the compute engine

    This is not currently used by the REF, but is included for future use.
    """

    scratch: Path = env_field(name="SCRATCH_ROOT", converter=ensure_absolute_path)
    """
    Shared scratch space for the REF.

    This directory is used to write the intermediate executions of a diagnostic execution.
    After the diagnostic has been run, the executions will be copied to the executions directory.

    This directory must be accessible by all the diagnostic services that are used to run the diagnostics,
    but does not need to be mounted in the same location on all the diagnostic services.
    """

    software: Path = env_field(name="SOFTWARE_ROOT", converter=ensure_absolute_path)
    """
    Shared software space for the REF.

    This directory is used to store software environments.

    This directory must be accessible by all the diagnostic services that are used to run the diagnostics,
    and should be mounted in the same location on all the diagnostic services.
    """

    # TODO: This could be another data source option
    results: Path = env_field(name="RESULTS_ROOT", converter=ensure_absolute_path)
    """
    Path to store the executions
    """

    dimensions_cv: Path | None = env_field(name="DIMENSIONS_CV_PATH", converter=_optional_path, default=None)
    """
    Path to a file containing the controlled vocabulary for the dimensions in a CMEC diagnostics bundle

    Leave this unset to use the controlled vocabulary for the CMIP7 Assessment Fast Track
    diagnostics, which is shipped inside the `climate_ref_core` package.
    That copy is read straight out of the installed package,
    so it needs no network access and no writable filesystem.

    This controlled vocabulary is used to validate the dimensions in the diagnostics bundle.
    If custom diagnostics are implemented,
    point this at a copy that has been extended with any new dimensions.
    """

    @log.default
    def _log_factory(self) -> Path:
        return env.path("REF_CONFIGURATION").resolve() / "log"

    @scratch.default
    def _scratch_factory(self) -> Path:
        return env.path("REF_CONFIGURATION").resolve() / "scratch"

    @software.default
    def _software_factory(self) -> Path:
        return env.path("REF_CONFIGURATION").resolve() / "software"

    @results.default
    def _results_factory(self) -> Path:
        return env.path("REF_CONFIGURATION").resolve() / "results"

    @property
    def dimensions_cv_resource(self) -> LayeredResource:
        """
        The controlled vocabulary to validate diagnostic bundles against

        Resolves to `dimensions_cv` if it is set, and to the copy shipped
        inside `climate_ref_core` otherwise.

        Returns
        -------
        :
            The resolved controlled vocabulary resource.
        """
        return LayeredResource(packaged=BUNDLED_CV, override=self.dimensions_cv)


@config(prefix=env_prefix)
class NativeStoreConfig:
    """
    Configuration for the content-addressed native-bundle object store.

    The native store holds the curated native outputs (NetCDF, PNG, ...) produced by each
    test case, keyed by their sha256 digest.
    Read operations (``has``, ``fetch``) are always anonymous and credential-free.
    Write operations are gated to the ``mint`` verb only.
    """

    url: str = env_field(name="NATIVE_STORE_URL", default="https://baselines.climate-ref.org")
    """
    Base URL of the native-bundle object store.

    Blobs are served at ``{url}/{digest}``.
    Defaults to the production Climate-REF baselines endpoint.

    Set ``REF_NATIVE_STORE_URL`` to a local ``file:///path/to/dir`` (or a plain filesystem path)
    for offline development and testing.
    """

    s3_endpoint_url: str = env_field(
        name="NATIVE_STORE_S3_ENDPOINT_URL",
        default="https://2aa5172b2bba093c516027d6fa13cdc8.r2.cloudflarestorage.com",
    )
    """
    S3 API endpoint for the writable (Cloudflare R2) backend, without the bucket.

    Non-secret routing config, consumed only by the ``mint`` verb. Defaults to the
    production Climate-REF R2 account endpoint (default jurisdiction — note there is no
    ``.eu`` in the host). Anonymous read (``fetch`` / ``has``) uses :attr:`url` instead and
    never touches this.
    Set ``REF_NATIVE_STORE_S3_ENDPOINT_URL`` to override (e.g. a staging account).
    """

    bucket: str = env_field(name="NATIVE_STORE_BUCKET", default="ref-baselines-public")
    """
    Name of the writable (Cloudflare R2) bucket.

    Non-secret routing config, consumed only by the ``mint`` verb.
    Set ``REF_NATIVE_STORE_BUCKET`` to override.

    Write credentials are **not** stored here: the access-key id and secret-access-key are
    read from ``REF_NATIVE_STORE_ACCESS_KEY_ID`` / ``REF_NATIVE_STORE_SECRET_ACCESS_KEY``
    (falling back to boto3's default credential chain) at upload time only, so secrets never
    land in a serialised config.
    """

    cache_dir: Path = env_field(name="NATIVE_STORE_CACHE_DIR", converter=Path)
    """
    Local pooch cache directory for downloaded native blobs.

    Defaults via :func:`~climate_ref_core.dataset_registry.resolve_cache_dir`,
    so the ``REF_DATASET_CACHE_DIR`` environment variable applies here too.
    """

    @cache_dir.default
    def _cache_dir_factory(self) -> Path:
        return resolve_cache_dir("native-baselines")


@config(prefix=env_prefix)
class ExecutorConfig:
    """
    Configuration to define the executor to use for running diagnostics
    """

    executor: str = env_field(name="EXECUTOR", default="climate_ref.executor.LocalExecutor")
    """
    Executor class to use for running diagnostics

    This should be the fully qualified name of the executor class
    (e.g. `climate_ref.executor.LocalExecutor`).
    The default is to use the local executor which runs the executions locally, in-parallel
    using a process pool.

    This class will be used for all executions of diagnostics.
    """

    config: dict[str, Any] = field(factory=dict)
    """
    Additional configuration for the executor.

    See the documentation for the executor for the available configuration options.
    These options will be passed to the executor class when it is created.
    """

    def build(self, config: "Config", database: "Database") -> "Executor":
        """
        Create an instance of the executor

        Returns
        -------
        :
            An executor that can be used to run diagnostics
        """
        # Import lazily to avoid loading heavy dependencies (pandas, xarray)
        # at module load time - these are only needed when actually running diagnostics
        from climate_ref_core.executor import Executor, import_executor_cls  # noqa: PLC0415

        ExecutorCls = import_executor_cls(self.executor)
        kwargs = {
            "config": config,
            "database": database,
            **self.config,
        }
        executor = ExecutorCls(**kwargs)

        if not isinstance(executor, Executor):
            raise InvalidExecutorException(executor, f"Expected an Executor, got {type(executor)}")
        return executor


@define
class DiagnosticProviderConfig:
    """
    Defining the diagnostic providers used by the REF.

    Each diagnostic provider is a package that contains the logic for running a specific
    set of diagnostics.
    This configuration determines which diagnostic providers are loaded and used when solving.

    Multiple diagnostic providers can be specified as shown in the example below.

    ```toml
    [[diagnostic_providers]]
    provider = "climate_ref_esmvaltool:provider"

    [diagnostic_providers.config]

    [[diagnostic_providers]]
    provider = "climate_ref_ilamb:provider"

    [diagnostic_providers.config]

    [[diagnostic_providers]]
    provider = "climate_ref_pmp:provider"

    [diagnostic_providers.config]
    ```
    """

    provider: str
    """
    Package that contains the diagnostic provider

    This should be the fully qualified name of the diagnostic provider.
    """

    config: dict[str, Any] = field(factory=dict)
    """
    Additional configuration for the diagnostic provider.

    See the documentation for the diagnostic package for the available configuration options.
    """

    # TODO: Additional configuration for narrowing down the diagnostics to run


@config(prefix=env_prefix)
class DbConfig:
    """
    Database configuration

    We support SQLite and PostgreSQL databases.
    The default is to use SQLite, which is a file-based database that is stored in the
    `REF_CONFIGURATION` directory.
    This is a good option for testing and development, but not recommended for production use.

    For production use, we recommend using PostgreSQL.
    """

    database_url: str = env_field(name="DATABASE_URL")
    """
    Database URL that describes the connection to the database.

    Defaults to `sqlite:///{config.paths.db}/climate_ref.db`.
    This configuration value will be overridden by the `REF_DATABASE_URL` environment variable.

    **Schemas**

    The following schemas are supported:
    ```
    postgresql://USER:PASSWORD@HOST:PORT/NAME

    sqlite:///RELATIVE_PATH or sqlite:////ABS_PATH or sqlite:///:memory:
    ```
    """
    run_migrations: bool = field(default=True)

    max_backups: int = env_field(name="MAX_BACKUPS", default=5)
    """
    Maximum number of database backups to keep.


    When running migrations for on-disk SQLite databases, a backup of the database is created.
    This setting controls how many of these backups are retained.
    The oldest backups are automatically removed when this limit is exceeded.
    """

    @database_url.default
    def _connection_url_factory(self) -> str:
        filename = env.path("REF_CONFIGURATION") / "db" / "climate_ref.db"
        sqlite_url = f"sqlite:///{filename}"
        return sqlite_url


def default_providers() -> list[DiagnosticProviderConfig]:
    """
    Return default diagnostic providers.

    Used if no diagnostic providers are specified in the configuration

    Returns
    -------
    :
        List of default diagnostic providers
    """
    env_providers = env.list("REF_DIAGNOSTIC_PROVIDERS", default=None)
    if env_providers:
        return [DiagnosticProviderConfig(provider=provider) for provider in env_providers]

    # Refer to https://setuptools.pypa.io/en/latest/userguide/entry_point.html#entry-points-for-plugins
    # and https://packaging.python.org/en/latest/specifications/entry-points/
    # to learn more about entry points.
    return [
        DiagnosticProviderConfig(provider=entry_point.value, config={})
        for entry_point in importlib.metadata.entry_points(group="climate-ref.providers")
    ]


def _load_config(config_file: str | Path, doc: dict[str, Any]) -> "Config":
    # Try loading the configuration with strict validation
    try:
        return _converter_defaults.structure(doc, Config)
    except Exception as exc:
        # Find the extra key errors which are displayed as warnings
        key_validation_errors = transform_error(exc, format_exception=_format_key_exception)
        for key_error in key_validation_errors:
            logger.warning(f"Error loading configuration from {config_file}: {key_error}")

    # Try again with relaxed validation
    return _converter_defaults_relaxed.structure(doc, Config)


DEFAULT_IGNORE_DATASETS_MAX_AGE = datetime.timedelta(hours=6)
DEFAULT_IGNORE_DATASETS_FILENAME = "default_ignore_datasets.yaml"
DEFAULT_IGNORE_DATASETS_URL = f"https://raw.githubusercontent.com/Climate-REF/climate-ref/refs/heads/main/{DEFAULT_IGNORE_DATASETS_FILENAME}"

BUNDLED_IGNORE_DATASETS = PackagedResource("climate_ref", DEFAULT_IGNORE_DATASETS_FILENAME)
"""
The grey list shipped inside the `climate_ref` package.

This is the copy that was current when the installed version was released.
It is always readable, so a solve never depends on the network or on a writable cache.
"""


DEFAULT_IGNORE_DATASETS_MAX_STALE = datetime.timedelta(days=30)


def _ignore_datasets_cache_file() -> Path:
    """
    Return the location the fetched copy of the grey list is cached at

    This is a pure path computation with no filesystem or network access.

    Returns
    -------
    :
        Path to the cached grey list, under the shared REF dataset cache.
    """
    return resolve_cache_dir("grey_list") / DEFAULT_IGNORE_DATASETS_FILENAME


def _legacy_ignore_datasets_file() -> Path:
    """
    Return the cache location used before the grey list moved under the REF dataset cache

    Releases up to 0.16 wrote this path into every saved configuration file as a default,
    so an upgraded installation has it recorded as though the operator had chosen it.

    Returns
    -------
    :
        The pre-0.17 cache location.
    """
    return platformdirs.user_cache_path("climate_ref") / DEFAULT_IGNORE_DATASETS_FILENAME


def _configured_ignore_datasets_file(config: "Config") -> Path | None:
    """
    Return the operator's chosen grey list, ignoring an inherited legacy default

    Before the grey list gained a packaged fallback its cache path was a config *default*,
    so `Config.save()` baked it into every `ref.toml`.
    Treating that value as a deliberate override would pin those installations to a stale
    cache and silently disable refreshing, so it is treated as unset.

    Parameters
    ----------
    config
        The configuration to read `ignore_datasets_file` from.

    Returns
    -------
    :
        The configured path, or None when nothing was deliberately chosen.
    """
    configured = config.ignore_datasets_file
    if configured is not None and configured == _legacy_ignore_datasets_file():
        logger.debug(
            f"Ignoring inherited grey list default {configured}. "
            "Remove `ignore_datasets_file` from your configuration to silence this."
        )
        return None
    return configured


def refresh_ignore_datasets_file(config: "Config") -> None:
    """
    Refresh the cached grey list from `config.ignore_datasets_url`.

    This is called at solve time so that configuration loading never performs network I/O.
    The download happens at most once every `DEFAULT_IGNORE_DATASETS_MAX_AGE`.
    A fresh cached file is reused untouched.

    Refreshing is best effort.
    An unset URL, an unreachable network, and an unwritable cache directory are all
    non-fatal: the solve falls back to the cached copy if there is one,
    and to the copy shipped in the package otherwise.
    This keeps read-only and air-gapped deployments working without any configuration.

    Refreshing is skipped entirely when `ignore_datasets_file` is set,
    since an explicit file is the operator's to manage.

    Parameters
    ----------
    config
        The configuration providing `ignore_datasets_file` and `ignore_datasets_url`.
    """
    url = config.ignore_datasets_url

    if not url or _configured_ignore_datasets_file(config) is not None:
        return

    try:
        path = _ignore_datasets_cache_file()

        if path.is_dir():
            logger.warning(f"The grey list cache path {path} is a directory, not a file. Skipping refresh.")
            return

        if _cache_age(path) < DEFAULT_IGNORE_DATASETS_MAX_AGE:
            return

        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Downloading the grey list from {url} to {path}")
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        _write_atomically(path, response.content)
    except Exception as exc:
        # The packaged copy is always available, so a failed refresh is never fatal.
        logger.warning(f"Could not refresh the grey list from {url}: {exc}")


def _cache_age(path: Path) -> datetime.timedelta:
    """
    Return how old a cached file is, treating an unreadable one as infinitely old

    Parameters
    ----------
    path
        The cached file.

    Returns
    -------
    :
        The age of the file, or `datetime.timedelta.max` if it cannot be read.
    """
    try:
        modification_time = datetime.datetime.fromtimestamp(path.stat().st_mtime)
    except OSError:
        return datetime.timedelta.max
    return datetime.datetime.now() - modification_time


def _write_atomically(path: Path, content: bytes) -> None:
    """
    Write a file via a temporary file and a rename

    A partial write would otherwise leave a truncated file with a fresh modification time,
    which would then shadow the packaged copy until the cache window expired.

    Parameters
    ----------
    path
        Destination file.
    content
        Bytes to write.
    """
    handle, temporary_name = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    temporary = Path(temporary_name)
    try:
        with os.fdopen(handle, "wb") as file:
            file.write(content)
        temporary.replace(path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise


@define(auto_attribs=True)
class Config:
    """
    Configuration that is used by the REF
    """

    _prefix = env_prefix

    log_level: str = field(default="INFO")
    """
    Log level of messages that are displayed by the REF via the CLI

    This value is overridden if a value is specified via the CLI.
    """
    log_format: str = env_field("LOG_FORMAT", default=DEFAULT_LOG_FORMAT)
    """
    Format of the log messages that are displayed by the REF via the CLI

    Examples of the formatting options are available in the
    [loguru documentation](https://loguru.readthedocs.io/en/stable/api/logger.html#module-loguru._logger).
    """

    cmip6_parser: Literal["drs", "complete"] = env_field("CMIP6_PARSER", default="complete")
    """
    Parser to use for CMIP6 datasets

    This can be either `drs` or `complete`.

    - `drs`: Use the DRS parser, which parses the dataset based on the DRS naming conventions.
    - `complete`: Use the complete parser, which parses the dataset based on all available metadata.
    """

    cmip7_parser: Literal["drs", "complete"] = env_field("CMIP7_PARSER", default="complete")
    """
    Parser to use for CMIP7 datasets

    This can be either `drs` or `complete`.

    - `drs`: Use the DRS parser, which parses the dataset based on the DRS naming conventions.
    - `complete`: Use the complete parser, which parses the dataset based on all available metadata.
    """

    ignore_datasets_file: Path | None = env_field(  # noqa: RUF009
        "IGNORE_DATASETS_FILE",
        converter=_optional_path,
        default=None,
    )
    """
    Path to a file containing the grey list

    This file is a YAML file that contains a list of facets to ignore per diagnostic.

    The format is:
    ```yaml
    provider:
      diagnostic:
        source_type:
          - facet: value
          - another_facet: [another_value1, another_value2]
    ```

    Leave this unset to use the grey list shipped inside the `climate_ref` package,
    refreshed from `ignore_datasets_url` when that is possible.
    Setting it pins the grey list to a file you manage, and disables fetching.
    """

    ignore_datasets_url: str = env_field("IGNORE_DATASETS_URL", default=DEFAULT_IGNORE_DATASETS_URL)
    """
    URL to refresh the grey list from at solve time.

    The download happens during solving only, at most once every 6 hours,
    and never during configuration loading.
    A failed download is not an error, since the copy shipped in the package is used instead.

    Set to an empty string (e.g. `REF_IGNORE_DATASETS_URL=`) to skip the attempt entirely,
    which avoids the request timeout on a host with no route to the internet.
    """

    paths: PathConfig = Factory(PathConfig)
    native_store: NativeStoreConfig = Factory(NativeStoreConfig)
    db: DbConfig = Factory(DbConfig)
    executor: ExecutorConfig = Factory(ExecutorConfig)
    diagnostic_providers: list[DiagnosticProviderConfig] = Factory(default_providers)  # noqa: RUF009, RUF100
    _raw: TOMLDocument | None = field(init=False, default=None, repr=False)
    _config_file: Path | None = field(init=False, default=None, repr=False)

    @property
    def ignore_datasets_resource(self) -> LayeredResource:
        """
        The grey list to apply when solving

        Resolves to `ignore_datasets_file` if it is set,
        then to a copy refreshed from `ignore_datasets_url`,
        and finally to the copy shipped inside `climate_ref`.

        A cache that has not been refreshed for `DEFAULT_IGNORE_DATASETS_MAX_STALE` is skipped.
        Without that bound a copy left behind by an older release would shadow the newer
        packaged copy indefinitely on a host that can never reach the network.

        Returns
        -------
        :
            The resolved grey list resource.
        """
        cache: Path | None = _ignore_datasets_cache_file()
        if cache is not None and _cache_age(cache) > DEFAULT_IGNORE_DATASETS_MAX_STALE:
            logger.debug(f"Ignoring the grey list cache at {cache} because it is too old to trust.")
            cache = None

        return LayeredResource(
            packaged=BUNDLED_IGNORE_DATASETS,
            override=_configured_ignore_datasets_file(self),
            cache=cache,
        )

    @classmethod
    def load(cls, config_file: Path, allow_missing: bool = True) -> "Config":
        """
        Load the configuration from a file

        Parameters
        ----------
        config_file
            Path to the configuration file.
            This should be a TOML file.

        Returns
        -------
        :
            The configuration loaded from the file
        """
        if config_file.is_file():
            with config_file.open() as fh:
                doc = tomlkit.load(fh)
                raw = doc
        else:
            if not allow_missing:
                raise FileNotFoundError(f"Configuration file not found: {config_file}")

            doc = TOMLDocument()
            raw = None

        try:
            config = _load_config(config_file, doc)
        except Exception as exc:
            # If that still fails, error out
            key_validation_errors = transform_error(exc, format_exception=_format_exception)
            for key_error in key_validation_errors:
                logger.error(f"Error loading configuration from {config_file}: {key_error}")

            # Deliberately not raising "from exc" to avoid long tracebacks from cattrs
            # The transformed error messages are sufficient for debugging
            raise ValueError(f"Error loading configuration from {config_file}") from None

        config._raw = raw
        config._config_file = config_file
        return config

    @classmethod
    def collect_validation_errors(cls, config_file: Path) -> list[str]:
        """
        Collect strict validation errors for a configuration file.

        Parameters
        ----------
        config_file
            Path to the configuration file to validate.

        Returns
        -------
        :
            A list of validation errors. An empty list means the file is valid.
        """
        try:
            with config_file.open() as fh:
                doc = tomlkit.load(fh)
            _converter_defaults.structure(doc, cls)
        except Exception as exc:
            return transform_error(exc, format_exception=_format_exception)
        return []

    def refresh(self) -> "Config":
        """
        Refresh the configuration values

        This returns a new instance of the configuration based on the same configuration file and
        any current environment variables.
        """
        if self._config_file is None:
            raise ValueError("No configuration file specified")
        return self.load(self._config_file)

    def save(self, config_file: Path | None = None) -> None:
        """
        Save the configuration as a TOML file

        The configuration will be saved to the specified file.
        If no file is specified, the configuration will be saved to the file
        that was used to load the configuration.

        Parameters
        ----------
        config_file
            The file to save the configuration to

        Raises
        ------
        ValueError
            If no configuration file is specified and the configuration was not loaded from a file
        """
        if config_file is None:
            if self._config_file is None:  # pragma: no cover
                # I'm not sure if this is possible
                raise ValueError("No configuration file specified")
            config_file = self._config_file

        config_file.parent.mkdir(parents=True, exist_ok=True)

        with open(config_file, "w") as fh:
            fh.write(self.dumps())

    @classmethod
    def default(cls) -> "Config":
        """
        Load the default configuration

        This will load the configuration from the default configuration location,
        which is typically the user's configuration directory.
        This location can be overridden by setting the `REF_CONFIGURATION` environment variable.

        Returns
        -------
        :
            The default configuration
        """
        root = env.path("REF_CONFIGURATION")
        path_to_load = root / CONFIG_FILENAME

        logger.debug(f"Loading default configuration from {path_to_load}")
        return cls.load(path_to_load)

    def dumps(self, defaults: bool = True) -> str:
        """
        Dump the configuration to a TOML string

        Parameters
        ----------
        defaults
            If True, include default values in the output

        Returns
        -------
        :
            The configuration as a TOML string
        """
        return self.dump(defaults).as_string()

    def dump(
        self,
        defaults: bool = True,
    ) -> TOMLDocument:
        """
        Dump the configuration to a TOML document

        Parameters
        ----------
        defaults
            If True, include default values in the output

        Returns
        -------
        :
            The configuration as a TOML document
        """
        if defaults:
            converter = _converter_defaults
        else:
            converter = _converter_no_defaults
        dump = converter.unstructure(self)
        # TOML cannot express null, so an unset value is written as an absent key.
        _pop_none(dump)
        if not defaults:
            _pop_empty(dump)
        doc = TOMLDocument()
        doc.update(dump)
        return doc

    def __attrs_post_init__(self) -> None:
        # This is needed to apply the environment variable overrides on initialization
        _environ_post_init(self)


def _make_converter(omit_default: bool, forbid_extra_keys: bool) -> Converter:
    conv = Converter(omit_if_default=omit_default, forbid_extra_keys=forbid_extra_keys)
    conv.register_unstructure_hook(Path, str)
    conv.register_unstructure_hook(
        Config,
        make_dict_unstructure_fn(
            Config,
            conv,
            _raw=override(omit=True),
            _config_file=override(omit=True),
        ),
    )
    return conv


_converter_defaults = _make_converter(omit_default=False, forbid_extra_keys=True)
_converter_defaults_relaxed = _make_converter(omit_default=False, forbid_extra_keys=False)
_converter_no_defaults = _make_converter(omit_default=True, forbid_extra_keys=True)
