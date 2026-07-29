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

### `REF_DATASET_FETCH_WORKERS`

Maximum number of files fetched concurrently by `ref datasets fetch-data`.
This must be a positive integer and defaults to `4`.
Set it to `1` to fetch files sequentially,
or reduce it when network bandwidth or the remote server limits concurrent transfers.

### `REF_TEST_DATA_DIR`

Override the location of the test data directory.
If this is not set, the test data directory will be inferred from the location of the test suite.

If this is set, then the sample data won't be updated.

### `REF_TEST_OUTPUT`

Path where the test output is stored.
This is used to store the output of the tests that are run in the test suite for later inspection.

## Celery environment variables

These apply when the Celery executor is in use.
The full set of tuning knobs is listed in the Helm chart README,
and the ones below govern the wire format.

Tasks and results are encoded as JSON (`ref-json`).
What a process sends is fixed and cannot be changed by configuration,
so no deployment can put pickle back on the wire.

### `CELERY_TASK_COMPRESSION` and `CELERY_RESULT_COMPRESSION`

Codec used to compress task and result message bodies, defaulting to `gzip`.
Bodies are dominated by the datasets DataFrame carried in the execution definition
and shrink by roughly 80%, which cuts broker memory on a full solve.
Set either to an empty string to send uncompressed bodies.

### `CELERY_ACCEPT_CONTENT`

Comma separated content types a worker will decode, defaulting to `json,ref-json`.
This only widens what is accepted.
It will not revive messages queued by a release that still used pickle,
because those carry a pickled diagnostic and provider that current releases no longer define.
Purge the queues when upgrading from such a release and re-solve.

## Grey list

The REF maintains a grey list:
datasets that are known to cause problems for particular diagnostics
and should be excluded from solving until the underlying issues are resolved.
The grey list is a YAML file listing facets to exclude per provider, diagnostic and source type.

!!! note "Naming"

    The configuration values below are currently named `ignore_datasets_*` for historical reasons.
    They will be renamed to `grey_list_*` in a future release.
    The old names will continue to work for a deprecation period.

The grey list is resolved from the first of three layers that is available:

1. `ignore_datasets_file`, if it is set.
2. A copy refreshed from `ignore_datasets_url` into the local cache.
3. The copy shipped inside the `climate_ref` package.

The third layer is always available,
so a solve never depends on the network or on a writable filesystem.

Two configuration values control this behaviour:

* `ignore_datasets_file` (env `REF_IGNORE_DATASETS_FILE`):
  a path to a grey list you manage yourself.
  Leave it unset to use the packaged copy.
  Setting it also disables fetching, because an explicit file is yours to manage.
* `ignore_datasets_url` (env `REF_IGNORE_DATASETS_URL`):
  the URL the grey list is refreshed from.
  It defaults to the copy served from the `main` branch of the Climate-REF repository.

Refreshing happens lazily during solving, not while the configuration is loaded,
so read-only commands such as `ref providers list` never perform network I/O.
When a solve runs,
the cached file is refreshed only if it is missing or older than six hours,
so at most one download happens per six-hour window.

Refreshing is best effort.
An unreachable network, an unwritable cache directory, and an HTTP error are all non-fatal.
The solve logs a warning and falls back to the cached copy if there is one,
and to the packaged copy otherwise.
Each provider logs which layer it read the grey list from at debug level.

A cached copy that has not been refreshed for 30 days is ignored in favour of the packaged copy.
Without that bound,
a cache left behind by an older release would shadow the newer packaged copy indefinitely
on a host that can never reach the network.
On such a host the cache can still shadow a newer packaged copy for up to 30 days after an upgrade.
Delete the cached file to read the packaged copy immediately.

### Offline and air-gapped deployments

Nothing needs to be configured.
The grey list and the dimensions controlled vocabulary are both read straight out of the
installed packages, so the REF runs with no outbound network access
and on a read-only filesystem.

On a host with no route to the internet,
set `REF_IGNORE_DATASETS_URL=` (an empty string) to skip the refresh attempt.
This avoids waiting for the request to time out on every solve.
It changes nothing else, since a failed refresh already falls back to the packaged copy.

To pin the grey list to a version you control,
point `REF_IGNORE_DATASETS_FILE` at your own copy.

## Configuration Options

<!-- This file is appended to by gen_config_stubs.py -->
