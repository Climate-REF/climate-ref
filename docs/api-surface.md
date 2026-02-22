# climate-ref-core Public API Surface

This document catalogues the public API of `climate-ref-core` that diagnostic
provider packages depend on. Any breaking change to these interfaces requires
a major version bump and a migration guide.

## Stability Tiers

| Tier            | Meaning                                                |
| --------------- | ------------------------------------------------------ |
| **Stable**      | Covered by semver; breaking changes require major bump |
| **Provisional** | May change in minor releases with deprecation notice   |
| **Internal**    | Prefixed with `_`; no stability guarantee              |

---

## `climate_ref_core.diagnostics` (Stable)

The primary module providers interact with.

| Symbol                  | Kind          | Description                                      |
| ----------------------- | ------------- | ------------------------------------------------ |
| `Diagnostic`            | Class         | Base class all diagnostics must subclass         |
| `AbstractDiagnostic`    | Protocol      | Protocol defining the diagnostic interface       |
| `CommandLineDiagnostic` | Class         | Base for diagnostics that shell out to CLI tools |
| `DataRequirement`       | Class (attrs) | Declares what datasets a diagnostic needs        |
| `ExecutionDefinition`   | Class (attrs) | Immutable description of a single execution      |
| `ExecutionResult`       | Class (attrs) | Result of running a diagnostic                   |
| `ensure_relative_path`  | Function      | Resolve a path relative to root_directory        |

### Key extension points

- Override `Diagnostic.run(definition) -> ExecutionResult`
- Override `Diagnostic.build_execution_result(definition) -> ExecutionResult`
- Set `Diagnostic.data_requirements`, `facets`, `slug`, `name`
- Set `Diagnostic.test_data_spec` for test case support

---

## `climate_ref_core.providers` (Stable)

| Symbol                          | Kind     | Description                               |
| ------------------------------- | -------- | ----------------------------------------- |
| `DiagnosticProvider`            | Class    | Registry for diagnostics from one package |
| `CommandLineDiagnosticProvider` | Class    | Provider that executes CLI commands       |
| `CondaDiagnosticProvider`       | Class    | Provider that manages a conda environment |
| `import_provider`               | Function | Import a provider by fully qualified name |

### Key extension points

- Override `DiagnosticProvider.configure(config)` for provider-level setup
- Override `setup_environment(config)`, `fetch_data(config)`, `ingest_data(config, db)`
- Override `validate_setup(config) -> bool`
- Call `provider.register(diagnostic)` to add diagnostics

---

## `climate_ref_core.datasets` (Stable)

| Symbol                       | Kind          | Description                                            |
| ---------------------------- | ------------- | ------------------------------------------------------ |
| `SourceDatasetType`          | Enum          | Enum of source types (CMIP6, obs4MIPs, PMPClimatology) |
| `DatasetCollection`          | Class (attrs) | A collection of datasets of one source type            |
| `ExecutionDatasetCollection` | Class (attrs) | Multi-source-type dataset bundle                       |
| `FacetFilter`                | Class (attrs) | Filter datasets by facet values                        |
| `Selector`                   | TypeAlias     | `tuple[tuple[str, str], ...]`                          |

---

## `climate_ref_core.constraints` (Stable)

| Symbol                        | Kind          | Description                                 |
| ----------------------------- | ------------- | ------------------------------------------- |
| `GroupConstraint`             | Protocol      | Interface for dataset grouping constraints  |
| `RequireFacets`               | Class (attrs) | Require specific facet values               |
| `IgnoreFacets`                | Class (attrs) | Exclude datasets matching facets            |
| `AddSupplementaryDataset`     | Class (attrs) | Attach supplementary data to groups         |
| `RequireTimerange`            | Class (attrs) | Require a minimum time range                |
| `RequireContiguousTimerange`  | Class (attrs) | Require contiguous time coverage            |
| `RequireOverlappingTimerange` | Class (attrs) | Require overlapping time ranges             |
| `SelectParentExperiment`      | Class (attrs) | Add parent experiment data                  |
| `PartialDateTime`             | Class         | Partial datetime for time range constraints |
| `apply_constraint`            | Function      | Apply a constraint to grouped data          |

---

## `climate_ref_core.executor` (Stable)

| Symbol                | Kind     | Description                                |
| --------------------- | -------- | ------------------------------------------ |
| `Executor`            | Protocol | Interface for execution backends           |
| `execute_locally`     | Function | Run a diagnostic in the current process    |
| `import_executor_cls` | Function | Import an executor by fully qualified name |

---

## `climate_ref_core.dataset_registry` (Stable)

| Symbol                     | Kind     | Description                        |
| -------------------------- | -------- | ---------------------------------- |
| `DatasetRegistryManager`   | Class    | Manages named pooch registries     |
| `dataset_registry_manager` | Instance | Singleton registry manager         |
| `fetch_all_files`          | Function | Download all files from a registry |
| `validate_registry_cache`  | Function | Verify cached file checksums       |
| `DATASET_URL`              | Constant | Base URL for dataset downloads     |

---

## `climate_ref_core.testing` (Stable)

| Symbol                     | Kind          | Description                                     |
| -------------------------- | ------------- | ----------------------------------------------- |
| `TestCase`                 | Class (attrs) | A single test case definition                   |
| `TestDataSpecification`    | Class (attrs) | Collection of test cases for a diagnostic       |
| `TestCasePaths`            | Class (attrs) | Path resolver for test case data                |
| `RegressionValidator`      | Class (attrs) | Validate outputs from stored regression data    |
| `validate_cmec_bundles`    | Function      | Validate CMEC metric/output bundles             |
| `collect_test_case_params` | Function      | Collect pytest parametrize params from provider |
| `load_datasets_from_yaml`  | Function      | Load ExecutionDatasetCollection from YAML       |
| `save_datasets_to_yaml`    | Function      | Save ExecutionDatasetCollection to YAML         |

---

## `climate_ref_core.exceptions` (Stable)

| Symbol                       | Kind  | Description                       |
| ---------------------------- | ----- | --------------------------------- |
| `RefException`               | Class | Base exception for all REF errors |
| `InvalidExecutorException`   | Class | Invalid executor configuration    |
| `InvalidProviderException`   | Class | Invalid provider configuration    |
| `InvalidDiagnosticException` | Class | Invalid diagnostic registration   |
| `ConstraintNotSatisfied`     | Class | Dataset constraint not met        |
| `ResultValidationError`      | Class | Result validation failure         |
| `ExecutionError`             | Class | Execution failure                 |
| `DiagnosticError`            | Class | Diagnostic runtime error          |
| `TestCaseError`              | Class | Base test case error              |
| `TestCaseNotFoundError`      | Class | Test case not found               |
| `NoTestDataSpecError`        | Class | Diagnostic has no test data spec  |
| `DatasetResolutionError`     | Class | Dataset resolution failure        |

---

## `climate_ref_core.pycmec` (Stable)

| Symbol                 | Kind             | Description                |
| ---------------------- | ---------------- | -------------------------- |
| `CMECMetric`           | Class (Pydantic) | CMEC metric bundle model   |
| `CMECOutput`           | Class (Pydantic) | CMEC output bundle model   |
| `ControlledVocabulary` | Class (Pydantic) | CMEC controlled vocabulary |

---

## `climate_ref_core.esgf` (Provisional)

| Symbol                | Kind          | Description                              |
| --------------------- | ------------- | ---------------------------------------- |
| `ESGFRequest`         | Class (attrs) | ESGF data request specification          |
| `CMIP6Request`        | Class (attrs) | CMIP6-specific ESGF request              |
| `CMIP7Request`        | Class (attrs) | CMIP7-specific ESGF request              |
| `Obs4MIPsRequest`     | Class (attrs) | obs4MIPs-specific ESGF request           |
| `ESGFDataFetcher`     | Class         | Fetch data from ESGF                     |
| `ESGFRequestRegistry` | Class         | Registry of ESGF requests per diagnostic |

---

## `climate_ref_core.source_types` (Stable)

| Symbol              | Kind | Description                                            |
| ------------------- | ---- | ------------------------------------------------------ |
| `SourceDatasetType` | Enum | Canonical source type enum (re-exported from datasets) |

---

## `climate_ref_core.logging` (Stable)

| Symbol               | Kind            | Description                            |
| -------------------- | --------------- | -------------------------------------- |
| `add_log_handler`    | Function        | Add a loguru handler                   |
| `remove_log_handler` | Function        | Remove the default handler             |
| `redirect_logs`      | Context manager | Redirect logs to file during execution |
| `capture_logging`    | Function        | Capture stdlib logging into loguru     |

---

## `climate_ref_core.env` (Internal)

| Symbol                    | Kind     | Description             |
| ------------------------- | -------- | ----------------------- |
| `env`                     | Instance | Environs Env instance   |
| `get_available_cpu_count` | Function | Get available CPU count |

---

## Entry Point Contract

Providers register via `pyproject.toml`:

```toml
[project.entry-points."climate-ref.providers"]
my_provider = "my_package:provider"
```

The `provider` attribute must be a `DiagnosticProvider` instance.
