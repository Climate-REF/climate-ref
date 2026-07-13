# Configuration

The REF uses a tiered configuration model, where options can be sourced from different places.
Then configuration is loaded from a `.toml` file which overrides any default values.
However, some configuration variables can be overridden at runtime using environment variables,
which always take precedence over any other configuration values set by default or found in a `.toml` file.

The default values for these environment variables are generally suitable,
but if you require updating these values we recommend the use of a `.env` file
to make the changes easier to reproduce in future.

## Configuration File Discovery

The REF will look for a configuration file in the following locations, taking the first one it finds:

* `${REF_CONFIGURATION}/ref.toml`
* `~/.config/climate_ref/ref.toml` (Linux)
* `$XDG_CONFIG_HOME/climate_ref/ref.toml` (Linux)
* `~/Library/Application Support/climate_ref/ref.toml` (macOS)
* `%USERPROFILE%\AppData\Local\climate_ref\ref.toml` (Windows)

If no configuration file is found, the REF will use the default configuration.

This directory may contain significant amounts of data,
so for HPC systems it is recommended to set the `REF_CONFIGURATION` environment variable to a directory on a scratch filesystem.

This default configuration is equivalent to the following:

```toml
log_level = "INFO"

[paths]
log = "${REF_CONFIGURATION}/log"
scratch = "${REF_CONFIGURATION}/scratch"
software = "${REF_CONFIGURATION}/software"
results = "${REF_CONFIGURATION}/results"
dimensions_cv = "${REF_INSTALLATION_DIR}/packages/climate-ref-core/src/climate_ref_core/pycmec/cv_cmip7_aft.yaml"

[db]
database_url = "sqlite:///${REF_CONFIGURATION}/db/climate_ref.db"
run_migrations = true

[executor]
executor = "climate_ref.executor.LocalExecutor"

[executor.config]

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

## Managing configuration from the CLI

Use `ref config init` to create a supported starter `ref.toml` in `REF_CONFIGURATION`.
The command creates parent directories and refuses to overwrite an existing file unless you pass `--force`.

Individual scalar values can be inspected or changed with dotted keys:

```bash
ref config get paths.scratch
ref config set log_level DEBUG
ref config unset log_level
```

`ref config get` prints the effective value the REF will use at runtime,
so environment variables such as `REF_DATABASE_URL` take precedence over values in `ref.toml`.
When an environment variable shadows a requested key,
the CLI keeps stdout script-friendly and writes the notice to stderr.

Run `ref config validate` after hand-editing the file.
For CI or editor integrations, use `ref config validate --format json` and rely on the exit code:
0 means valid, 1 means invalid.

`ref config set` and `ref config unset` rewrite `ref.toml` from the parsed configuration model.
This is convenient for simple scalar changes,
but it does not preserve comments or custom key ordering in a hand-edited file.
Edit structured values such as `diagnostic_providers` and `executor.config` directly in TOML.

## Additional Environment Variables

Environment variables are used to control some aspects of the framework
outside of the configuration file.

### `REF_DATASET_CACHE_DIR`

Path where any datasets that are fetched via the `ref datasets fetch-data` command are stored.
This directory will be several GB in size,
so it is recommended to set this to a directory on a scratch filesystem
rather than a directory on your home filesystem.

This is used to cache the datasets so that they are not downloaded multiple times.
It is not recommended to ingest datasets from this directory (see `--output-dir` argument for `ref datasets fetch-data`).

This defaults to the following locations:

* `~/Library/Caches/climate_ref` (MacOS)
* `~/.cache/climate_ref` or the value of the `$XDG_CACHE_HOME/climate_ref`
  environment variable, if defined. (Linux)
* `%USERPROFILE%\AppData\Local\climate_ref\Cache` (Windows)

### `REF_TEST_DATA_DIR`

Override the location of the test data directory.
If this is not set, the test data directory will be inferred from the location of the test suite.

If this is set, then the sample data won't be updated.

### `REF_TEST_OUTPUT`

Path where the test output is stored.
This is used to store the output of the tests that are run in the test suite for later inspection.

## Grey list

The REF maintains a grey list:
datasets that are known to cause problems for particular diagnostics
and should be excluded from solving until the underlying issues are resolved.
The grey list is a YAML file listing facets to exclude per provider, diagnostic and source type.

!!! note "Naming"

    The configuration values below are currently named `ignore_datasets_*` for historical reasons.
    They will be renamed to `grey_list_*` in a future release;
    the old names will continue to work for a deprecation period.

Two configuration values control this behaviour:

* `ignore_datasets_file` (env `REF_IGNORE_DATASETS_FILE`) —
  the path to the grey list file.
  It defaults to a location under the user cache directory
  and is coerced to a filesystem path.
* `ignore_datasets_url` (env `REF_IGNORE_DATASETS_URL`) —
  the URL the grey list is fetched from.
  It defaults to the copy shipped in the Climate-REF repository.

The grey list is fetched lazily during solving,
not while the configuration is loaded.
Read-only commands such as `ref providers list` and `ref datasets list` never perform network I/O.
When a solve runs,
the cached file is refreshed from `ignore_datasets_url` only if it is missing or older than six hours,
so at most one download happens per six-hour window.

If the download fails and a cached copy already exists,
the cached copy is reused unchanged and a warning is logged.
If the download fails and no cached copy exists,
the solve fails with an error
rather than silently running without the grey list protections.

### Offline and air-gapped deployments

On systems without outbound network access (for example an air-gapped HPC cluster),
seed the grey list manually and disable fetching:

1. Copy `default_ignore_datasets.yaml` to a writable location on the target system.
2. Point `ignore_datasets_file` at that copy,
   for example `REF_IGNORE_DATASETS_FILE=/shared/config/default_ignore_datasets.yaml`.
3. Set `REF_IGNORE_DATASETS_URL=` (an empty string) to disable fetching entirely.

With an empty URL the solver never touches the network
and uses whatever grey list already exists at `ignore_datasets_file`.

## Configuration Options

<!-- This file is appended to by gen_config_stubs.py -->
