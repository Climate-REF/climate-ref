# Configuration

This tutorial assumes that you have already installed Climate-REF and are using a Linux or MacOS operating system.
The `ref` CLI tool should be available in your terminal after installation.
For installation instructions, see [Installation](../installation.md).

Climate-REF uses a [TOML](https://toml.io/en/) configuration file to specify data paths, output directories, and other settings. In this step, we will generate and customize your configuration file.

Additional information about the configuration file can be found in the [Configuration documentation](../configuration.md).


## 1. Select a location for storing your configuration

The most important part of the REF configuration is the location where the REF will store its data and results.
This is determined using the `$REF_CONFIGURATION` environment variable.
This can use up a large amount of disk space, so it is important to choose a location with sufficient storage.

If no value is provided a default location will be used, but this will not be suitable for most users
who use shared computing facilities.

This environment variable can be set in your shell configuration file (e.g., `.bashrc`, `.zshrc`, etc.)
or exported directly in your terminal session (assuming a bash shell):

```bash
export REF_CONFIGURATION="/path/to/your/ref/configuration"
```


## 2. Generate

Climate-REF provides a script to write out the default configuration.

```bash
mkdir $REF_CONFIGURATION
ref config list > $REF_CONFIGURATION/ref.toml
```

This command will create the `$REF_CONFIGURATION` directory and create a `ref.toml` inside it with the default configuration settings.

/// admonition | Note

The location that the REF looks for the configuration file can be viewed by running a CLI command using the `-v` flag:

```
$ ref -v config list
2025-05-28 10:45:29.244 +10:00 | DEBUG    | climate_ref.cli - Configuration loaded from: /path/to/your/climate-ref/.ref/ref.toml
...
```

///

## 3. Edit your configuration

Open `$REF_CONFIGURATION/ref.toml` in your editor of choice.
You will see a template configuration file with sections for logging, paths, database settings, and diagnostic providers.
These should be customized to suit your environment and preferences.

Additional information about the configuration file can be found in the [Configuration documentation](../configuration.md).

An example configuration file might look like this with some placeholders:

```toml
log_level = "INFO"
log_format = "<green>{time:YYYY-MM-DD HH:mm:ss.SSS Z}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>"

[paths]
log = "$REF_CONFIGURATION/log"
scratch = "$REF_CONFIGURATION/scratch"
software = "$REF_CONFIGURATION/software"
results = "$REF_CONFIGURATION/results"
dimensions_cv = "$REF_INSTALL_DIR/climate-ref-core/src/climate_ref_core/pycmec/cv_cmip7_aft.yaml"

[db]
database_url = "sqlite:///$REF_CONFIGURATION/db/climate_ref.db"
run_migrations = true
max_backups = 5

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


The particularly important sections to customize are:

- **paths**: Set the paths for logs, scratch space, software, and results. These should point to directories where you have write access.
- **db**: Configure the database URL. By default, it uses SQLite, but you can change it to a PostgreSQL or other database if needed.
- **executor**: Set the executor type. The default is `LocalExecutor`, but you can change it to `CeleryExecutor` or `HPCExecutor` for distributed execution (see the [Executor documentation](../how-to-guides/executors.md) for more details).
- **diagnostic_providers**: List the diagnostic providers you want to use. The default includes ESMValTool, ILAMB, and PMP. You can add or remove providers as needed.

## 4. Environment variables

Optionally, you can export environment variables instead of hardcoding paths. See the [Environment Variables documentation](../configuration.md#additional-environment-variables) for more details.

One important environment variable is `REF_DATASET_CACHE_DIR`,
which specifies where the REF will cache downloaded datasets.
This can be GBs of data, so it is recommended to set this to a scratch filesystem or a location with sufficient disk space.

This can be set as follows:

```bash
export REF_DATASET_CACHE_DIR="/path/to/your/dataset/cache"
```

If environment variables are set, Climate-REF will use their values in preference to those found in the configuration file.


## 5. Validate your configuration

To ensure your configuration is valid and correctly read by the REF, you can run the following command:

```bash
ref config list
```

Your configuration should be displayed without errors and should include any changes you made in the `ref.toml` file.


## 6. Set up diagnostic providers

Diagnostic providers may need additional setup before they can be used,
such as creating conda environments or downloading reference data.
Run the following command to prepare all providers:

```bash
ref providers setup
```

This creates any required conda environments (installed in `$REF_CONFIGURATION/software` by default,
configurable via [paths.software](../configuration.md#paths_software)) and downloads reference data to the local cache.

/// admonition | HPC users
    type: tip

If you're on an HPC system where compute nodes lack internet access,
run this command on a login node before submitting jobs.

///

## Next steps

After configuring, proceed to the [Download Datasets](02-download-datasets.md) tutorial to load your data into Climate-REF.
