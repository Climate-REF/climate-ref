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

The *grey list* is a YAML file that lists facets which should be excluded from
specific diagnostics — for example, datasets that are known to crash or produce invalid output.
The datasets in the grey list are filtered before solving for the relevant diagnostic.

The file format is:

```yaml
provider:
  diagnostic:
    source_type:
      - facet: value
      - other_facet: [other_value1, other_value2]
```

Two configuration values control how the grey list is loaded; both can be set
in your `ref.toml` or via environment variables.

### `grey_list_file` / `REF_GREY_LIST_FILE`

Path to the grey list file on disk.
Defaults to `grey_list.yaml` inside your `REF_CONFIGURATION` directory (alongside `ref.toml`, the database,
etc.). This location must be writable by the user as the grey
list may be updated periodically.

### `grey_list_url` / `REF_GREY_LIST_URL`

URL the solver fetches the grey list from.
Defaults to
`config/default_grey_list.yaml` on the `main` branch of the
`Climate-REF/climate-ref` GitHub repository.
Override this to point at a
fork or internal mirror.

The download is **lazy and explicit**: it only runs once at the start of a
solve (`ExecutionSolver.build_from_db`), and only when the on-disk copy is
missing or older than 6 hours.
Read-only commands like `ref providers list`
or `ref datasets list` never touch the network.

### Offline / air-gapped use (HPC)

To run completely offline — for example on an HPC compute node with no
outbound network — set the URL to an empty value:

```bash
export REF_GREY_LIST_URL=
```

or in `ref.toml`:

```toml
grey_list_url = ""
```

When fetching is disabled the solver simply uses whatever file is at
`grey_list_file`.
A missing file is treated as an empty grey list, so you
do not have to seed the file by hand;
if you want to apply a specific
grey list, either copy `config/default_grey_list.yaml` from the repository
to your `grey_list_file` location ahead of time,
or fetch it once before disabling the URL on the compute nodes.

## Configuration Options

<!-- This file is appended to by gen_config_stubs.py -->
