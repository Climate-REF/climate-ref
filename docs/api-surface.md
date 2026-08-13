# climate-ref-core public API surface

This page lists what a diagnostic provider package is allowed to import from `climate-ref-core`, and what it can rely on staying put.
Breaking any of the stable interfaces below requires a major version bump and a migration guide.

## Stability tiers

| Tier            | Meaning                       |
| --------------- | ----------------------------- |
| **Stable**      | Covered by semver.            |
| **Provisional** | May change in a minor release |
| **Internal**    | No guarantees at all          |

Breaking changes to `Stable` feature require a major bump and may include deprecation messages.
Any `_`-prefixed names are also assumed to be Internal and may change at any time.

## `climate_ref_core.diagnostics` (Stable)

The module providers spend most of their time in.

| Symbol                  | Kind             | Description                                                          |
| ----------------------- | ---------------- | -------------------------------------------------------------------- |
| `Diagnostic`            | Class            | Base class all diagnostics subclass                                  |
| `AbstractDiagnostic`    | Protocol         | The same interface, for code that only needs to type-check it        |
| `CommandLineDiagnostic` | Class            | Base for diagnostics that shell out to a CLI tool                    |
| `DataRequirement`       | Class (attrs)    | The input datasets a diagnostic needs, and how to group them         |
| `ExecutionDefinition`   | Class (attrs)    | One execution: a diagnostic plus the exact datasets it will run on   |
| `ExecutionResult`       | Class (attrs)    | What an execution produced, its CMEC bundles and its `resource_usage` |
| `ensure_relative_path`  | Function         | Make a path relative to a root directory                             |
| `SeriesDefinition`      | Class (Pydantic) | Declares a 1-d array output, with its index and dimensions           |
| `FileDefinition`        | Class (Pydantic) | Declares an output file, with its dimensions                         |

`SeriesDefinition` and `FileDefinition` are re-exported from
[`climate_ref_core.metric_values.typing`](#climate_ref_coremetric_values-stable);
either import path works.

### Extension points

- Override `Diagnostic.run(definition) -> ExecutionResult`
- Override `Diagnostic.build_execution_result(definition) -> ExecutionResult`
- Set `Diagnostic.data_requirements`, `facets`, `slug` and `name`
- Set `Diagnostic.series` and `Diagnostic.files` to declare the metric values and files a run emits
- Bump `Diagnostic.version` whenever results change enough to need recomputation.
  The value is append-only: always increment, never reuse a number.
- Set `Diagnostic.reconstruction_inputs` for output globs to keep beyond what the CMEC bundle references
- Set `Diagnostic.test_data_spec` to opt in to test case support

## `climate_ref_core.providers` (Stable)

| Symbol                          | Kind     | Description                                              |
| ------------------------------- | -------- | -------------------------------------------------------- |
| `DiagnosticProvider`            | Class    | Registers and runs the diagnostics of one package        |
| `CommandLineDiagnosticProvider` | Class    | Provider whose diagnostics run as command line calls     |
| `CondaDiagnosticProvider`       | Class    | As above, inside a conda environment the provider manages |
| `import_provider`               | Function | Import a provider from a fully qualified name            |

### Extension points

- Override `DiagnosticProvider.configure(config)` for provider-level setup
- Override `setup_environment(config)`, `fetch_data(config)` and `ingest_data(config, db)`
- Override `validate_setup(config) -> bool`
- Call `provider.register(diagnostic)` to add a diagnostic

## `climate_ref_core.datasets` (Stable)

| Symbol                       | Kind          | Description                                                 |
| ---------------------------- | ------------- | ----------------------------------------------------------- |
| `SourceDatasetType`          | Enum          | Supported source types (CMIP6, obs4MIPs, PMPClimatology, …) |
| `DatasetCollection`          | Class (attrs) | The datasets of one source type needed for an execution     |
| `ExecutionDatasetCollection` | Class         | All of an execution's datasets, keyed by source type        |
| `FacetFilter`                | Class (attrs) | A filter applied to a data catalog                          |
| `Selector`                   | TypeAlias     | `tuple[tuple[str, str], ...]`                               |

## `climate_ref_core.constraints` (Stable)

Constraints run over each candidate group of datasets during the solve, and either
reshape the group or reject it.

| Symbol                        | Kind          | Description                                              |
| ----------------------------- | ------------- | -------------------------------------------------------- |
| `GroupConstraint`             | Protocol      | The interface the constraints below implement            |
| `RequireFacets`               | Class (attrs) | Reject groups missing the given facet values             |
| `IgnoreFacets`                | Class (attrs) | Drop datasets matching the given facet values            |
| `AddSupplementaryDataset`     | Class (attrs) | Pull in a cell measure or ancillary variable             |
| `RequireTimerange`            | Class (attrs) | Reject groups that don't cover a given time range        |
| `RequireContiguousTimerange`  | Class (attrs) | Reject groups with gaps in time                          |
| `RequireOverlappingTimerange` | Class (attrs) | Reject groups whose datasets don't overlap in time       |
| `AddParentDataset`            | Class (attrs) | Pull in a dataset's parent experiment                    |
| `SelectFirstMember`           | Class (attrs) | Keep a single ensemble member per group                  |
| `PartialDateTime`             | Class         | An underspecified date, for comparing against timeranges |
| `apply_constraint`            | Function      | Apply one constraint to a group                          |

## `climate_ref_core.executor` (Stable)

| Symbol                | Kind     | Description                                        |
| --------------------- | -------- | -------------------------------------------------- |
| `Executor`            | Protocol | Runs executions asynchronously; local, Celery, HPC |
| `execute_locally`     | Function | Run one execution in the current process           |
| `import_executor_cls` | Function | Import an executor from a fully qualified name     |

## `climate_ref_core.resources` (Provisional)

| Symbol              | Kind            | Description                                            |
| ------------------- | --------------- | ------------------------------------------------------ |
| `measure_resources` | Context manager | Measure wall time, CPU time and peak memory of a block |
| `ResourceUsage`     | Class (attrs)   | Frozen record of what one block of work cost           |
| `ResourceRecorder`  | Class           | Handle yielded by `measure_resources`, holding `usage` |
| `MemorySource`      | TypeAlias       | Provenance of a peak memory figure                     |

Provisional because the set of measured fields is expected to grow as more hosts are covered.
`ExecutionResult.resource_usage` carries a `ResourceUsage` back from the worker,
and defaults to `None`, so a provider that never sets it stays valid.

---

## `climate_ref_core.dataset_registry` (Stable)

Registries describe reference data that isn't published through ESGF.

| Symbol                      | Kind     | Description                                                   |
| --------------------------- | -------- | ------------------------------------------------------------- |
| `DatasetRegistryManager`    | Class    | Holds the named pooch registries                              |
| `dataset_registry_manager`  | Instance | The process-wide manager; providers register into this        |
| `RegistryEntry`             | Class    | One registry plus the metadata needed to ingest it            |
| `RegistryUseCase`           | Enum     | Whether a registry is catalog-ingestable or fetch-only        |
| `fetch_all_files`           | Function | Download a registry's files into an output directory          |
| `validate_registry_cache`   | Function | Check that cached files are present and match their checksums |
| `resolve_cache_dir`         | Function | Locate the cache directory a registry uses                    |
| `iter_reference_registries` | Function | Yield `(registry, source_type)` for the ingestable registries |
| `DATASET_URL`               | Constant | Base URL the registries download from                         |

## `climate_ref_core.testing` (Provisional)

| Symbol                     | Kind          | Description                                                |
| -------------------------- | ------------- | ---------------------------------------------------------- |
| `TestCase`                 | Class (attrs) | One test case for a diagnostic                             |
| `TestDataSpecification`    | Class (attrs) | The test cases a diagnostic declares, via `test_data_spec` |
| `TestCasePaths`            | Class (attrs) | Resolves where a test case's data lives                    |
| `collect_test_case_params` | Function      | Diagnostic/test-case pairs for `pytest.mark.parametrize`   |
| `load_datasets_from_yaml`  | Function      | Read an `ExecutionDatasetCollection` from YAML             |
| `save_datasets_to_yaml`    | Function      | Write an `ExecutionDatasetCollection` to YAML              |

## `climate_ref_core.exceptions` (Stable)

| Symbol                       | Kind  | Description                                     |
| ---------------------------- | ----- | ----------------------------------------------- |
| `RefException`               | Class | Base class for every exception below            |
| `InvalidExecutorException`   | Class | An executor could not be imported or configured |
| `InvalidProviderException`   | Class | A provider could not be imported or configured  |
| `InvalidDiagnosticException` | Class | A diagnostic failed validation at registration  |
| `ConstraintNotSatisfied`     | Class | A group did not satisfy a constraint            |
| `ResultValidationError`      | Class | An execution's output failed validation         |
| `ExecutionError`             | Class | An execution failed                             |
| `DiagnosticError`            | Class | The diagnostic's own computation raised         |
| `TestCaseError`              | Class | Base class for the test case errors below       |
| `TestCaseNotFoundError`      | Class | No test case by that name                       |
| `NoTestDataSpecError`        | Class | The diagnostic declares no `test_data_spec`     |
| `DatasetResolutionError`     | Class | A test case's datasets could not be resolved    |

## `climate_ref_core.pycmec` (Stable)

Models for the CMEC bundle formats. The package root is empty; import from the submodules.

### `climate_ref_core.pycmec.metric`

| Symbol              | Kind             | Description                                    |
| ------------------- | ---------------- | ---------------------------------------------- |
| `CMECMetric`        | Class (Pydantic) | A CMEC metric bundle                           |
| `MetricCV`          | Enum             | The bundle's controlled vocabulary of keys     |
| `MetricDimensions`  | Class (Pydantic) | The bundle's `DIMENSIONS` object               |
| `MetricResults`     | Class (Pydantic) | The bundle's `RESULTS` object                  |
| `remove_dimensions` | Function         | Strip dimensions from a raw bundle             |

### `climate_ref_core.pycmec.output`

| Symbol             | Kind             | Description                                |
| ------------------ | ---------------- | ------------------------------------------ |
| `CMECOutput`       | Class (Pydantic) | A CMEC output bundle                       |
| `OutputCV`         | Enum             | The bundle's controlled vocabulary of keys |
| `OutputProvenance` | Class (Pydantic) | The bundle's provenance object             |

### `climate_ref_core.pycmec.controlled_vocabulary`

| Symbol           | Kind          | Description                                            |
| ---------------- | ------------- | ------------------------------------------------------ |
| `CV`             | Class (attrs) | The dimensions and values executions are validated against |
| `Dimension`      | Class (attrs) | One dimension within the vocabulary                    |
| `DimensionValue` | Class (attrs) | One value a dimension permits                          |

## `climate_ref_core.esgf` (Provisional)

| Symbol                   | Kind     | Description                                             |
| ------------------------ | -------- | ------------------------------------------------------- |
| `ESGFRequest`            | Protocol | What a dataset request has to provide                   |
| `CMIP6Request`           | Class    | A CMIP6 request                                         |
| `CMIP7Request`           | Class    | A CMIP7 request                                         |
| `Obs4MIPsRequest`        | Class    | An obs4MIPs request                                     |
| `RegistryRequest`        | Class    | A request served from a pooch registry, not from ESGF   |
| `ESGFFetcher`            | Class    | Fetches the requested datasets and returns their paths  |
| `IntakeESGFMixin`        | Mixin    | Gives a request class its intake-esgf search            |
| `enable_ceda_solr_index` | Function | Add the CEDA Solr index to the ESGF search indices      |

## `climate_ref_core.metric_values` (Stable)

| Symbol              | Kind             | Description                                              |
| ------------------- | ---------------- | -------------------------------------------------------- |
| `SeriesMetricValue` | Class (Pydantic) | A 1-d array with its index and dimensions                |
| `ScalarMetricValue` | Class (Pydantic) | A single value with its dimensions                       |
| `MetricValueKind`   | TypeAlias        | `Literal["model", "reference"]`                          |

### `climate_ref_core.metric_values.typing`

| Symbol             | Kind             | Description                                                |
| ------------------ | ---------------- | ---------------------------------------------------------- |
| `SeriesDefinition` | Class (Pydantic) | Declares a series a diagnostic emits, ahead of running it  |
| `FileDefinition`   | Class (Pydantic) | Declares a file a diagnostic emits, ahead of running it    |

Both are also re-exported from
[`climate_ref_core.diagnostics`](#climate_ref_corediagnostics-stable).

## `climate_ref_core.source_types` (Stable)

| Symbol              | Kind      | Description                   |
| ------------------- | --------- | ----------------------------- |
| `SourceDatasetType` | Enum      | Supported source types        |
| `Selector`          | TypeAlias | `tuple[tuple[str, str], ...]` |

Both are defined here and re-exported by `climate_ref_core.datasets`.
Import from either.

## `climate_ref_core.logging` (Stable)

| Symbol               | Kind            | Description                                     |
| -------------------- | --------------- | ----------------------------------------------- |
| `add_log_handler`    | Function        | Add a sink to the loguru logger                 |
| `remove_log_handler` | Function        | Remove the default handler                      |
| `redirect_logs`      | Context manager | Send log output to a file for the duration      |
| `capture_logging`    | Function        | Route stdlib `logging` into loguru              |

## `climate_ref_core.esmvaltool_reference` (Internal)

The path conventions ESMValTool uses for observational and reanalysis data.

| Symbol                 | Kind       | Description                                       |
| ---------------------- | ---------- | ------------------------------------------------- |
| `ReferenceFacets`      | NamedTuple | The metadata a reference file's path encodes      |
| `parse_reference_path` | Function   | Read that metadata off a path                     |
| `drs_relative_parts`   | Function   | Split a path into its DRS-relative components     |
| `tier_from_segment`    | Function   | Read the tier number out of a `TierN` directory   |
| `PROJECT_ANCHORS`      | Constant   | The directory names that anchor a reference tree  |

## `climate_ref_core.cmip6_to_cmip7` (Internal)

Presents CMIP6 data under CMIP7 conventions.
Providers generally only need `get_dreq_entry`; the rest serves the REF's own ingestion
and will move as CMIP7 settles.

| Symbol                         | Kind     | Description                                             |
| ------------------------------ | -------- | ------------------------------------------------------- |
| `get_dreq_entry`               | Function | Look up a variable in the Data Request by compound name |
| `convert_cmip6_to_cmip7_attrs` | Function | Convert CMIP6 global attributes to CMIP7                |
| `convert_cmip6_dataset`        | Function | Convert a whole dataset in memory                       |
| `CMIP7Metadata`                | Class    | The CMIP7 attributes a conversion produces              |

## `climate_ref_core.env` (Internal)

| Symbol                    | Kind     | Description                                          |
| ------------------------- | -------- | ---------------------------------------------------- |
| `env`                     | Instance | The environs reader the REF reads its settings from  |
| `get_available_cpu_count` | Function | CPU count, respecting cgroup limits                  |

## Entry point contract

Providers register themselves in `pyproject.toml`:

```toml
[project.entry-points."climate-ref.providers"]
my_provider = "my_package:provider"
```

The attribute named on the right must be a `DiagnosticProvider` instance.
