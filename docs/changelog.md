# Changelog

Versions follow [Semantic Versioning](https://semver.org/) (`<major>.<minor>.<patch>`).

Backward incompatible (breaking) changes will only be introduced in major versions
with advance notice in the **Deprecations** section of releases.

<!--
You should *NOT* be adding new changelog entries to this file,
this file is managed by towncrier.
See `changelog/README.md`.

You *may* edit previous changelogs to fix problems like typo corrections or such.
To add a new changelog entry, please see
`changelog/README.md`
and https://pip.pypa.io/en/latest/development/contributing/#news-entries,
noting that we use the `changelog` directory instead of news,
markdown instead of restructured text and use slightly different categories
from the examples given in that link.
-->

<!-- towncrier release notes start -->

## climate-ref 0.5.5 (2025-05-21)

### Improvements

- Added additional dimensions to the ILAMB and ESMValTool metric values.
  This includes additional information about the execution group that will be useful to end-users. ([#308](https://github.com/Climate-REF/climate-ref/pulls/308))
- Move the ILAMB datasets to S3 ([#309](https://github.com/Climate-REF/climate-ref/pulls/309))
- Clean ECS diagnostic (remove unused keys in ESMValTool recipes and avoid "cmip6" in diagnostic name) ([#310](https://github.com/Climate-REF/climate-ref/pulls/310))
- Clean TCR diagnostic (remove unused keys in ESMValTool recipes and avoid "cmip6" in diagnostic name) ([#311](https://github.com/Climate-REF/climate-ref/pulls/311))

### Improved Documentation

- Updated documentation to include more information about concepts within the REF. ([#312](https://github.com/Climate-REF/climate-ref/pulls/312))


## climate-ref 0.5.4 (2025-05-19)

### Bug Fixes

- Add additional dependencies to the `climate-ref-core` so that it is self-contained ([#307](https://github.com/Climate-REF/climate-ref/pulls/307))


## climate-ref 0.5.3 (2025-05-19)

### Features

- Diagnostic's have been split into two phases, executing which generates any outputs and then building a result
  object from the outputs.
  This split makes it easier to make modifications to how the results are translated into the CMEC outputs. ([#303](https://github.com/Climate-REF/climate-ref/pulls/303))

### Improvements

- Added automatic backup of SQLite database files before running migrations.
  Backups are stored in a `backups` directory adjacent to the database file and are named with timestamps.
  The number of backups to retain can be configured via the `db.max_backups` setting in the database configuration,
  with a default of 5 backups. ([#301](https://github.com/Climate-REF/climate-ref/pulls/301))
- Update to v0.6.0 of the sample data. ([#302](https://github.com/Climate-REF/climate-ref/pulls/302))
- Add a smoke test for the Celery deployment ([#304](https://github.com/Climate-REF/climate-ref/pulls/304))
- Add tests that the pypi releases are installable ([#306](https://github.com/Climate-REF/climate-ref/pulls/306))

### Improved Documentation

- Added page describing the required reference datasets ([#298](https://github.com/Climate-REF/climate-ref/pulls/298))


## climate-ref 0.5.2 (2025-05-15)

### Bug Fixes

- Fix missing dependency in migrations ([#297](https://github.com/Climate-REF/climate-ref/pulls/297))

### Improved Documentation

- Added documentation for configuration options ([#296](https://github.com/Climate-REF/climate-ref/pulls/296))


## climate-ref 0.5.1 (2025-05-14)

### Features

- Added an ESMValTool metric to compute climatologies and zonal mean profiles of
  cloud radiative effects. ([#241](https://github.com/Climate-REF/climate-ref/pulls/241))
- Add additional dimensions to the metrics produced by PMP.
  Added `climate_ref_core.pycmec.metric.CMECMetric.prepend_dimensions` ([#275](https://github.com/Climate-REF/climate-ref/pulls/275))
- Ensure that selectors are always alphabetically sorted ([#276](https://github.com/Climate-REF/climate-ref/pulls/276))
- Add data model for supporting series of metric values.
  This allows diagnostic providers to supply a collection of [climate_ref_core.metric_values.SeriesMetricValue][]
  extracted from an execution. ([#278](https://github.com/Climate-REF/climate-ref/pulls/278))
- The default executor ([climate_ref.executor.LocalExecutor][]) uses a process pool to enable parallelism.
  An alternative [climate_ref.executor.SynchronousExecutor][] is available for debugging purposes,
  which runs tasks synchronously in the main thread. ([#286](https://github.com/Climate-REF/climate-ref/pulls/286))

### Improvements

- Bumps the ilamb3 version to now contain all analysis modules and reformats its CMEC output bundle. ([#262](https://github.com/Climate-REF/climate-ref/pulls/262))
- Adds the ability to capture the output of an execution for regression testing. ([#274](https://github.com/Climate-REF/climate-ref/pulls/274))
- Update to v0.5.1 of the sample data. ([#279](https://github.com/Climate-REF/climate-ref/pulls/279))
- Update to v0.5.2 of the sample data. ([#282](https://github.com/Climate-REF/climate-ref/pulls/282))
- Add a CITATION.cff to the repository to make it easier to cite. ([#283](https://github.com/Climate-REF/climate-ref/pulls/283))
- Added the Assessment Fast Track-related services to the `docker-compose` stack alongside improved documentation for how to use the REF via docker containers. ([#287](https://github.com/Climate-REF/climate-ref/pulls/287))
- Added support for ingesting multiple directories at once.
  This is useful for ingesting large datasets that are split into multiple directories or via glob patterns.
  An example of this is importing the monthly and fx datasets from an archive of CMIP6 data:

  ```bash
  ref datasets ingest --source-type cmip6 path_to_archive/CMIP6/*/*/*/*/*/*mon path_to_archive/CMIP6/*/*/*/*/*/fx
  ``` ([#291](https://github.com/Climate-REF/climate-ref/pulls/291))
- Update the default log level to INFO from WARNING.
  Added the `-q` option to decrease the log level to WARNING. ([#292](https://github.com/Climate-REF/climate-ref/pulls/292))

### Bug Fixes

- Resolves an issue that was blocking some PMP executions from completing.
  Any additional dimensions are now logged and ignored rather than causing the execution to fail. ([#274](https://github.com/Climate-REF/climate-ref/pulls/274))
- Sets the environment variable `FI_PROVIDER=tcp` to use the TCP provider for libfabric (part of MPICH).
  The defaults were causing MPICH errors on some systems (namely macOS).
  This also removes the PMP provider's direct dependency on the source of `pcmdi_metric`. ([#281](https://github.com/Climate-REF/climate-ref/pulls/281))
- Support the use of empty metric bundles ([#284](https://github.com/Climate-REF/climate-ref/pulls/284))
- Reworked the lifetimes of the database transactions during the solve process.
  This is a fix for out of process executors where the transaction was not being committed until the end of a solve. ([#288](https://github.com/Climate-REF/climate-ref/pulls/288))
- Requery an Execution from the database when handling the result from the LocalExecutor.
  This ensures that the execution isn't stale and that the result is still valid. ([#293](https://github.com/Climate-REF/climate-ref/pulls/293))


## climate-ref 0.5.0 (2025-05-03)

### Breaking Changes

- Renamed packages to start with `climate_ref_` and removed `metrics` from the package name to avoid confusion.
  This changes the root name of the PyPi packages from `cmip_ref` to `climate-ref` and will require updating your `requirements.txt`, `pyproject.toml`, `setup.py`, or other dependency management files to list `climate-ref` instead of `cmip_ref`. ([#270](https://github.com/Climate-REF/climate-ref/pulls/270))
- Clarified the difference between a diagnostic and a metric.
  This caused significant refactoring of names of classes and functions throughout the codebase,
  as well as renaming of database tables.

  | Package                      | Old Name                   | New Name                      |
  |------------------------------|----------------------------|-------------------------------|
  | climate_ref_core.diagnostics | Metric                     | Diagnostic                    |
  | climate_ref_core.diagnostics | MetricExecutionDefinition  | ExecutionDefinition           |
  | climate_ref_core.diagnostics | MetricExecutionResult      | ExecutionResult               |
  | climate_ref.models.execution | MetricExecutionResultß     | Execution                     |
  | climate_ref.models.execution | MetricExecutionGroup       | ExecutionGroup                |
  | climate_ref.models.execution | ResultOutput               | ExecutionOutput               |
  | climate_ref_core.datasets    | MetricDataset              | ExecutionDatasetCollection    |
  | climate_ref_core.solver      | MetricSolver               | ExecutionSolver               |
  | climate_ref_core.providers   | MetricsProvider            | DiagnosticProvider            |
  | climate_ref_core.providers   | CommandLineMetricsProvider | CommandLineDiagnosticProvider |
  | climate_ref_core.providers   | CondaMetricsProvider       | CondaDiagnosticProvider       |
  | climate_ref.config           | MetricsProviderConfig      | DiagnosticProviderConfig      |

  This removes any previous database migrations and replaces them with a new clean migration.
  If you have an existing database, you will need to delete it and re-create it. ([#271](https://github.com/Climate-REF/climate-ref/pulls/271))


## cmip_ref 0.4.1 (2025-05-02)

### Breaking Changes

- Removed unnecessary prefixes in the metric slugs.
  This will cause duplicate results to be generated so we recommend starting with a clean REF installation. ([#263](https://github.com/Climate-REF/climate-ref/pulls/263))

### Features

- Added PMP's annual cycle metrics ([#221](https://github.com/Climate-REF/climate-ref/pulls/221))
- Add a `facets` attribute to a metric.
  This attribute is used to define the facets of the values that the metric produces. ([#255](https://github.com/Climate-REF/climate-ref/pulls/255))
- Added a diagnostic to calculate climate variables at global warming levels. ([#257](https://github.com/Climate-REF/climate-ref/pulls/257))
- Support multiple sets of data requirements ([#266](https://github.com/Climate-REF/climate-ref/pulls/266))

### Bug Fixes

- Retry downloads if they fail ([#267](https://github.com/Climate-REF/climate-ref/pulls/267))
- PMP annual cycle output JSON tranformed to be more comply with CMEC ([#268](https://github.com/Climate-REF/climate-ref/pulls/268))

### Improved Documentation

- Add deprecation notices to PyPi package README's ([#269](https://github.com/Climate-REF/climate-ref/pulls/269))


## cmip_ref 0.4.0 (2025-04-29)

### Breaking Changes

- Removed `climate_ref.solver.MetricSolver.solve_metric_executions` in preference for a standalone function `climate_ref.solver.solve_metric_executions`
  with identical functionality. ([#229](https://github.com/Climate-REF/climate-ref/pulls/229))
- Updated the algorithm to generate the unique identifier for a Metric Execution Group.
  This will cause duplicate entries in the database if the old identifier was used.
  We recommend removing your existing database and starting fresh. ([#233](https://github.com/Climate-REF/climate-ref/pulls/233))
- Removed the implicit treatment of the deepest dimension. The change will cause a validation error if the deepest dimension in the `RESULTS` is a dictionary. ([#246](https://github.com/Climate-REF/climate-ref/pulls/246))
- Ensure that the order of the source dataset types in the MetricExecutionGroup dataset key are stable ([#248](https://github.com/Climate-REF/climate-ref/pulls/248))

### Deprecations

- Removes support for Python 3.10.
  The minimum and default supported Python version is now 3.11. ([#226](https://github.com/Climate-REF/climate-ref/pulls/226))

### Features

- Add the basic framework for enforcing a controlled vocabulary
  for the results in a CMEC metrics bundle.
  This is still in the prototype stage
  and is not yet integrated into post-metric execution processing. ([#183](https://github.com/Climate-REF/climate-ref/pulls/183))
- Scalar values from the metrics are now stored in the database
  if they pass validation.
  The controlled vocabulary for these metrics is still under development. ([#185](https://github.com/Climate-REF/climate-ref/pulls/185))
- Added Zero Emission Commitment (ZEC) metric to the ESMValTool metrics package. ([#204](https://github.com/Climate-REF/climate-ref/pulls/204))
- Added Transient Climate Response to Cumulative CO2 Emissions (TCRE) metric to the ESMValTool metrics package. ([#208](https://github.com/Climate-REF/climate-ref/pulls/208))
- Add `ref datasets fetch-obs4ref-data` CLI command to fetch datasets that are in the process of being published to obs4MIPs and are appropriately licensed for use within the REF.
  The CLI command fetches the datasets and writes them to a local directory.
  These datasets can then be ingested into the REF as obs4MIPs datasets. ([#219](https://github.com/Climate-REF/climate-ref/pulls/219))
- Enabled metric providers to register registries of datasets for download.
  This unifies the fetching of datasets across the REF via the `ref datasets fetch-data` CLI command.
  Added registries for the datasets that haven't been published to obs4MIPs yet (`obs4REF`) as well as PMP annual cycle datasets. ([#227](https://github.com/Climate-REF/climate-ref/pulls/227))
- Capture log output for each execution and display via `ref executions inspect`. ([#232](https://github.com/Climate-REF/climate-ref/pulls/232))
- Added the option to install development versions of metrics packages. ([#236](https://github.com/Climate-REF/climate-ref/pulls/236))
- Added seasonal cycle and time series of sea ice area metrics. ([#239](https://github.com/Climate-REF/climate-ref/pulls/239))
- The unique group identifiers for a MetricExecutionGroup are now tracked in the database. These values are used for presentation. ([#248](https://github.com/Climate-REF/climate-ref/pulls/248))
- Added a new dataset source type to track PMP climatology data ([#253](https://github.com/Climate-REF/climate-ref/pulls/253))

### Improvements

- Refactored `MetricSolver.solve_metric_executions` to be a standalone function for easier testing.
  Also added some additional integration tests for the Extratropical Modes of Variability metric from PMP. ([#229](https://github.com/Climate-REF/climate-ref/pulls/229))
- The configuration paths are now all resolved to absolute paths ([#230](https://github.com/Climate-REF/climate-ref/pulls/230))
- Verified support for PostgreSQL database backends ([#231](https://github.com/Climate-REF/climate-ref/pulls/231))
- Updated the ESMValTool metric and output bundles. ([#235](https://github.com/Climate-REF/climate-ref/pulls/235))
- Update to v0.5.0 of the sample data ([#264](https://github.com/Climate-REF/climate-ref/pulls/264))

### Bug Fixes

- Relax some of the requirements for the availability of metadata in CMIP6 datasets. ([#245](https://github.com/Climate-REF/climate-ref/pulls/245))
- Added a missing migration and tests to ensure that the migration are up to date. ([#247](https://github.com/Climate-REF/climate-ref/pulls/247))
- Fixed how the path to ESMValTool outputs was determined,
  and added support for outputs in nested directories. ([#250](https://github.com/Climate-REF/climate-ref/pulls/250))
- Marked failing tests as xfail as a temporary solution. ([#259](https://github.com/Climate-REF/climate-ref/pulls/259))
- Fetch ESMValTool reference data in the integration test suite ([#265](https://github.com/Climate-REF/climate-ref/pulls/265))

### Improved Documentation

- Now following [SPEC-0000](https://scientific-python.org/specs/spec-0000/) for dependency support windows.
  Support for Python versions will be dropped after 3 years and support for key scientific libraries will be dropped after 2 years. ([#226](https://github.com/Climate-REF/climate-ref/pulls/226))
- Migrate from the use of ‘AR7 Fast Track’ to ‘Assessment Fast Track’ in response to the CMIP Panel decision to [change the name of the CMIP7 fast track](https://wcrp-cmip.org/fast-track-name-update/). ([#251](https://github.com/Climate-REF/climate-ref/pulls/251))

### Trivial/Internal Changes

- [#220](https://github.com/Climate-REF/climate-ref/pulls/220)


## cmip_ref 0.3.1 (2025-03-28)

### Trivial/Internal Changes

- [#218](https://github.com/Climate-REF/climate-ref/pulls/218)


## cmip_ref 0.3.0 (2025-03-28)

### Breaking Changes

- We changed the `ref` Command Line Interface to make the distinction between execution
  groups and individual executions clear. A metric execution is the evaluation of
  a specific metric for a specific set of input datasets. We group together all
  executions for the same set of input datasets which are re-run because
  the metric or the input datasets were updated or because a metric execution
  failed. For showing results, it is more useful to think in terms of execution groups.
  In particular, the `ref executions list` command was re-named to
  `ref executions list-groups`. ([#165](https://github.com/Climate-REF/climate-ref/pulls/165))

### Features

- Support ingesting obs4MIPs datasets into the REF ([#113](https://github.com/Climate-REF/climate-ref/pulls/113))
- Add extratropical modes of variability analysis using PMP ([#115](https://github.com/Climate-REF/climate-ref/pulls/115))
- Added management of conda environments for metrics package providers.

  Several new commands are available for working with providers:
  - `ref providers list` - List the available providers
  - `ref providers create-env` - Create conda environments for providers

  ([#117](https://github.com/Climate-REF/climate-ref/pulls/117))
- Added [CMECMetric.create_template][climate_ref_core.pycmec.metric.CMECMetric.create_template] method to create an empty CMEC metric bundle. ([#123](https://github.com/Climate-REF/climate-ref/pulls/123))
- Outputs from a metric execution and their associated metadata are now tracked in the database. This includes HTML, plots and data outputs.

  Metric providers can register outputs via the CMEC output bundle.
  These outputs are then ingested into the database if the execution was successful. ([#125](https://github.com/Climate-REF/climate-ref/pulls/125))
- Build and publish container images to [Github Container Registry](https://github.com/Climate-REF/climate-ref/pkgs/container/ref) ([#156](https://github.com/Climate-REF/climate-ref/pulls/156))
- Enable more variability modes for PMP modes of variability metrics ([#173](https://github.com/Climate-REF/climate-ref/pulls/173))
- Add a `--timeout` option to the `solve` cli command.
  This enables the user to set a maximum time for the solver to run. ([#186](https://github.com/Climate-REF/climate-ref/pulls/186))

### Improvements

- Cleanup of ilamb3 interface code, enabling IOMB comparisons. ([#124](https://github.com/Climate-REF/climate-ref/pulls/124))
- Migrate the PMP provider to use a REF-managed conda environment.

  For non-MacOS users, this should be created automatically.
  MacOS users will need to create the environment using the following command:

  ```bash
  MAMBA_PLATFORM=osx-64 uv run ref providers create-env --provider pmp
  ``` ([#127](https://github.com/Climate-REF/climate-ref/pulls/127))
- Fixed issue with `mypy` not being run across the celery package ([#128](https://github.com/Climate-REF/climate-ref/pulls/128))
- Added the `fetch-ref-data` make command to download reference data while it's not in obs4mips, yet. ([#155](https://github.com/Climate-REF/climate-ref/pulls/155))
- Improvements:
  - Drop the metric plugin version number in the environment name because the environment may not change between releases
  - Avoid calling micromamba update as this may not work for everyone
  - Print out the location of environments even when they are not installed
  - Do not mention conda as a requirement on the Hackathon page

  ([#160](https://github.com/Climate-REF/climate-ref/pulls/160))
- Add activity and institute to ESMValTool recipes to allow running with models
  and experiments that are not in the CMIP6 controlled vocabulary. ([#166](https://github.com/Climate-REF/climate-ref/pulls/166))
- Add an integration test for the CMIP7 AFT metric providers.
  This will be run nightly as part of the Climate-REF CI pipeline. ([#187](https://github.com/Climate-REF/climate-ref/pulls/187))
- Improved the error message when running `ref datasets list` with the `--column` argument.
  If a column is specified that is not available, the error message now only mentions
  the invalid column name(s) and shows a list of available columns. ([#203](https://github.com/Climate-REF/climate-ref/pulls/203))
- Do not list duplicate entries in dataframes shown from the command line. ([#210](https://github.com/Climate-REF/climate-ref/pulls/210))

### Bug Fixes

- Remove example metric for ILAMB ([#121](https://github.com/Climate-REF/climate-ref/pulls/121))
- Removed the "SCHEMA" attribute from the CMEC metric bundle as it is not part of the EMDS specification and unused. ([#123](https://github.com/Climate-REF/climate-ref/pulls/123))
- Fixed the validation error when 'attributes' value is a dict ([#133](https://github.com/Climate-REF/climate-ref/pulls/133))
- Fixed PMP's modes of variablity PDO metrics driver to use obs4MIP-complying reference dataset. Also update PMP's version to 3.9, which include turning off direct usage of conda as a part of the driver (to capture provenance info) ([#154](https://github.com/Climate-REF/climate-ref/pulls/154))
- Enforce the use of relative paths when copying files after an execution. This resolves an issue where files were not being copied to the correct location causing failures in PMP. ([#170](https://github.com/Climate-REF/climate-ref/pulls/170))
- If no obs4mips-compliant reference dataset is found in specified directory, give a meaningful error message. ([#174](https://github.com/Climate-REF/climate-ref/pulls/174))
- Fixed the behaviour of FacetFilter with `keep=False` so all facets need to match
  before excluding a file. ([#209](https://github.com/Climate-REF/climate-ref/pulls/209))

### Improved Documentation

- Renamed CMIP-REF to Climate-REF ([#119](https://github.com/Climate-REF/climate-ref/pulls/119))
- Add a landing page for hackathon attendees ([#120](https://github.com/Climate-REF/climate-ref/pulls/120))
- Fix the incorrect capitalisation of GitHub organisation ([#122](https://github.com/Climate-REF/climate-ref/pulls/122))
- Updated the getting started documentation. ([#126](https://github.com/Climate-REF/climate-ref/pulls/126))
- Update the roadmap to reflect progress as of 2025/03/10 ([#134](https://github.com/Climate-REF/climate-ref/pulls/134))
- Clarified language and other small fixes in the documentation. ([#178](https://github.com/Climate-REF/climate-ref/pulls/178))

### Trivial/Internal Changes

- [#161](https://github.com/Climate-REF/climate-ref/pulls/161), [#182](https://github.com/Climate-REF/climate-ref/pulls/182), [#184](https://github.com/Climate-REF/climate-ref/pulls/184), [#207](https://github.com/Climate-REF/climate-ref/pulls/207)


## cmip_ref 0.2.0 (2025-03-01)

### Breaking Changes

- Refactor `climate_ref.env` module to `climate_ref_core.env` ([#60](https://github.com/Climate-REF/climate-ref/pulls/60))
- Removed `climate_ref.executor.ExecutorManager` in preference to loading an executor using a fully qualified package name.

  This allows the user to specify a custom executor as configuration
  without needing any change to the REF codebase. ([#77](https://github.com/Climate-REF/climate-ref/pulls/77))
- Renamed the `$.paths.tmp` in the configuration to `$.paths.scratch` to better reflect its purpose.
  This requires a change to the configuration file if you have a custom configuration. ([#89](https://github.com/Climate-REF/climate-ref/pulls/89))
- The REF now uses absolute paths throughout the application.

  This removes the need for a `config.paths.data` directory and the `config.paths.allow_out_of_tree_datasets` configuration option.
  This will enable more flexibility about where input datasets are ingested from.
  Using absolute paths everywhere does add a requirement that datasets are available via the same paths for all nodes/container that may run the REF. ([#100](https://github.com/Climate-REF/climate-ref/pulls/100))
- An [Executor][climate_ref_core.executor.Executor] now supports only the asynchronous processing of tasks.
  A result is now not returned from the `run_metric` method,
  but instead optionally updated in the database.

  The `run_metric` method also now requires a `provider` argument to be passed in. ([#104](https://github.com/Climate-REF/climate-ref/pulls/104))

### Features

- Adds a `cmip-ref-celery` package to the REF that provides a Celery task queue.

  Celery is a distributed task queue that allows you to run tasks asynchronously.
  This package will be used as a test bed for running the REF in a distributed environment,
  as it can be deployed locally using docker containers. ([#60](https://github.com/Climate-REF/climate-ref/pulls/60))
- Add `metric_providers` and `executor` sections to the configuration which loads the metric provider and executor using a fully qualified package name. ([#77](https://github.com/Climate-REF/climate-ref/pulls/77))
- Implemented Pydantic data models to validate and serialize CMEC metric and output bundles. ([#84](https://github.com/Climate-REF/climate-ref/pulls/84))
- Add the `climate_ref_celery` CLI commands to the `ref` CLI tool.
  These commands should be available when the `climate_ref_celery` package is installed.
  The commands should be available in the `ref` CLI tool as `ref celery ...`. ([#86](https://github.com/Climate-REF/climate-ref/pulls/86))
- Add `fetch-sample-data` to the CLI under the `datasets` command.

  ```bash
  ref datasets fetch-sample-data --version v0.3.0 --force-cleanup
  ``` ([#96](https://github.com/Climate-REF/climate-ref/pulls/96))
- Add a [Celery](https://docs.celeryq.dev/en/stable/)-based executor
  to enable asynchronous processing of tasks. ([#104](https://github.com/Climate-REF/climate-ref/pulls/104))
- Add `ref executions list` and `ref executions inspect` CLI commands for interacting with metric executions. ([#108](https://github.com/Climate-REF/climate-ref/pulls/108))

### Improvements

- Move ILAMB/IOMB reference data initialization to a registry-dependent script. ([#83](https://github.com/Climate-REF/climate-ref/pulls/83))
- ILAMB gpp metrics added with full html output and plots. ([#88](https://github.com/Climate-REF/climate-ref/pulls/88))
- Saner error messages for configuration errors ([#89](https://github.com/Climate-REF/climate-ref/pulls/89))
- Centralised the declaration of environment variable overrides of configuration values.

  Renamed the `REF_OUTPUT_ROOT` environment variable to `REF_RESULTS_ROOT` to better reflect its purpose.
  It was previously unused. ([#92](https://github.com/Climate-REF/climate-ref/pulls/92))
- Sample data is now copied to the `test/test-data/sample-data` instead of symlinked.

  This makes it easier to use the sample data with remote executors as the data is now self-contained
  without any links to other parts of the file system. ([#96](https://github.com/Climate-REF/climate-ref/pulls/96))
- Integrated the pycmec validation models into ref core and metric packages ([#99](https://github.com/Climate-REF/climate-ref/pulls/99))
- Added ILAMB relationship analysis to the current metrics and flexibility to define new metrics in ILAMB via a yaml file. ([#101](https://github.com/Climate-REF/climate-ref/pulls/101))
- Sped up the test suite execution ([#103](https://github.com/Climate-REF/climate-ref/pulls/103))

### Improved Documentation

- Added an excerpt from the architecture design document ([#87](https://github.com/Climate-REF/climate-ref/pulls/87))
- Adds a roadmap to the documentation ([#98](https://github.com/Climate-REF/climate-ref/pulls/98))

### Trivial/Internal Changes

- [#97](https://github.com/Climate-REF/climate-ref/pulls/97), [#102](https://github.com/Climate-REF/climate-ref/pulls/102), [#116](https://github.com/Climate-REF/climate-ref/pulls/116), [#118](https://github.com/Climate-REF/climate-ref/pulls/118)


## cmip_ref 0.1.6 (2025-02-03)

### Features

- Added Equilibrium Climate Sensitivity (ECS) to the ESMValTool metrics package. ([#51](https://github.com/Climate-REF/climate-ref/pulls/51))
- Added Transient Climate Response (TCS) to the ESMValTool metrics package. ([#62](https://github.com/Climate-REF/climate-ref/pulls/62))
- Added the possibility to request datasets with complete and overlapping timeranges. ([#64](https://github.com/Climate-REF/climate-ref/pulls/64))
- Added a constraint for selecting supplementary variables, e.g. cell measures or
  ancillary variables. ([#65](https://github.com/Climate-REF/climate-ref/pulls/65))
- Added a sample metric to the ilamb metrics package. ([#66](https://github.com/Climate-REF/climate-ref/pulls/66))
- Added a sample metric to the PMP metrics package. ([#72](https://github.com/Climate-REF/climate-ref/pulls/72))
- - Added the standard ILAMB bias analysis as a metric. ([#74](https://github.com/Climate-REF/climate-ref/pulls/74))

### Bug Fixes

- - Added overlooked code to fully integrate ilamb into ref. ([#73](https://github.com/Climate-REF/climate-ref/pulls/73))
- Correct the expected configuration name to `ref.toml` as per the documentation. ([#82](https://github.com/Climate-REF/climate-ref/pulls/82))

### Improved Documentation

- Update the package name in the changelog.

  This will simplify the release process by fixing the extraction of changelog entries. ([#61](https://github.com/Climate-REF/climate-ref/pulls/61))

### Trivial/Internal Changes

- [#68](https://github.com/Climate-REF/climate-ref/pulls/68)


## cmip_ref 0.1.5 (2025-01-13)

### Trivial/Internal Changes

- [#56](https://github.com/Climate-REF/climate-ref/pulls/56)


## cmip_ref 0.1.4 (2025-01-13)

### Breaking Changes

- Adds an `ingest` CLI command to ingest a new set of data into the database.

  This breaks a previous migration as alembic's `render_as_batch` attribute should have been set
  to support targeting sqlite. ([#14](https://github.com/Climate-REF/climate-ref/pulls/14))
- Renames `ref ingest` to `ref datasets ingest` ([#30](https://github.com/Climate-REF/climate-ref/pulls/30))
- Prepend package names with `cmip_` to avoid conflicting with an existing `PyPI` package.

  This is a breaking change because it changes the package name and all imports.
  All package names will now begin with `cmip_ref`. ([#53](https://github.com/Climate-REF/climate-ref/pulls/53))

### Features

- Migrate to use UV workspaces to support multiple packages in the same repository.
  Adds a `climate-ref-example` package that will be used to demonstrate the integration of a metric
  package into the REF. ([#2](https://github.com/Climate-REF/climate-ref/pulls/2))
- Adds the placeholder concept of `Executor`'s which are responsible for running metrics
  in different environments. ([#4](https://github.com/Climate-REF/climate-ref/pulls/4))
- Adds the concept of MetricProvider's and Metrics to the core.
  These represent the functionality that metric providers must implement in order to be part of the REF.
  The implementation is still a work in progress and will be expanding in follow-up PRs. ([#5](https://github.com/Climate-REF/climate-ref/pulls/5))
- Add a collection of ESGF data that is required for test suite.

  Package developers should run `make fetch-test-data` to download the required data for the test suite. ([#6](https://github.com/Climate-REF/climate-ref/pulls/6))
- Adds the `ref` package with a basic CLI interface that will allow for users to interact with the database of jobs. ([#8](https://github.com/Climate-REF/climate-ref/pulls/8))
- Add `SqlAlchemy` as an ORM for the database alongside `alembic` for managing database migrations. ([#11](https://github.com/Climate-REF/climate-ref/pulls/11))
- Added a `DataRequirement` class to declare the requirements for a metric.

  This provides the ability to:

  * filter a data catalog
  * group datasets together to be used in a metric calculation
  * declare constraints on the data that is required for a metric calculation

  ([#15](https://github.com/Climate-REF/climate-ref/pulls/15))
- Add a placeholder iterative metric solving scheme ([#16](https://github.com/Climate-REF/climate-ref/pulls/16))
- Extract a data catalog from the database to list the currently ingested datasets ([#24](https://github.com/Climate-REF/climate-ref/pulls/24))
- Translated selected groups of datasets into `MetricDataset`s.
  Each `MetricDataset` contains all of the dataset's needed for a given execution of a metric.

  Added a slug to the `MetricDataset` to uniquely identify the execution
  and make it easier to identify the execution in the logs. ([#29](https://github.com/Climate-REF/climate-ref/pulls/29))
- Adds `ref datasets list` command to list ingested datasets ([#30](https://github.com/Climate-REF/climate-ref/pulls/30))
- Add database structures to represent a metric execution.
  We record previous executions of a metric to minimise the number of times we need to run metrics. ([#31](https://github.com/Climate-REF/climate-ref/pulls/31))
- Added option to skip any datasets that fail validation and to specify the number of cores to
  use when ingesting datasets.
  This behaviour can be opted in using the `--skip-invalid` and `--n-jobs` options respectively. ([#36](https://github.com/Climate-REF/climate-ref/pulls/36))
- Track datasets that were used for different metric executions ([#39](https://github.com/Climate-REF/climate-ref/pulls/39))
- Added an example ESMValTool metric. ([#40](https://github.com/Climate-REF/climate-ref/pulls/40))
- Support the option for different assumptions about the root paths between executors and the ref CLI.

  Where possible path fragments are stored in the database instead of complete paths.
  This allows the ability to move the data folders without needing to update the database. ([#46](https://github.com/Climate-REF/climate-ref/pulls/46))

### Improvements

- Add a bump, release and deploy flow for automating the release procedures ([#20](https://github.com/Climate-REF/climate-ref/pulls/20))
- Migrate test data into standalone [Climate-REF/ref-sample-data](https://github.com/Climate-REF/ref-sample-data) repository.

  The sample data will be downloaded by the test suite automatically into `tests/test-data/sample-data`,
  or manually by running `make fetch-test-data`. ([#49](https://github.com/Climate-REF/climate-ref/pulls/49))

### Bug Fixes

- Adds `version` field to the `instance_id` field for CMIP6 datasets ([#35](https://github.com/Climate-REF/climate-ref/pulls/35))
- Handle missing branch times.
  Fixes [#38](https://github.com/Climate-REF/climate-ref/issues/38). ([#42](https://github.com/Climate-REF/climate-ref/pulls/42))
- Move alembic configuration and migrations to `cmip_ref` package so that they can be included in the distribution. ([#54](https://github.com/Climate-REF/climate-ref/pulls/54))

### Improved Documentation

- Deployed documentation to https://climate-ref.readthedocs.io/en/latest/ ([#16](https://github.com/Climate-REF/climate-ref/pulls/16))
- General documentation cleanup.

  Added notebook describing the process of executing a notebook locally ([#19](https://github.com/Climate-REF/climate-ref/pulls/19))
- Add Apache licence to the codebase ([#21](https://github.com/Climate-REF/climate-ref/pulls/21))
- Improved developer documentation. ([#47](https://github.com/Climate-REF/climate-ref/pulls/47))

### Trivial/Internal Changes

- [#41](https://github.com/Climate-REF/climate-ref/pulls/41), [#44](https://github.com/Climate-REF/climate-ref/pulls/44), [#48](https://github.com/Climate-REF/climate-ref/pulls/48), [#52](https://github.com/Climate-REF/climate-ref/pulls/52), [#55](https://github.com/Climate-REF/climate-ref/pulls/55)
