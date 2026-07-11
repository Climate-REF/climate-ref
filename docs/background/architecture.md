# Architecture

This page describes how the Rapid Evaluation Framework (REF) is put together:
the systems involved, the packages and their layering, the components inside each package,
and the runtime flows that connect them.
It is a description of the system as implemented.
For the vocabulary and the ideas behind the workflow, start with [Concepts](../concepts.md).
For the stable extension points that providers depend on, see the [API surface](../api-surface.md).

The REF was designed for the CMIP7 Assessment Fast Track (AFT):
near-real-time benchmarking of Earth System Models against observational datasets as they are submitted,
producing scalar, timeseries and gridded diagnostics along with figures and web pages.
The CMIP7 Model Benchmarking Task Team (MBTT) identified the initial set of diagnostics,
which are implemented by existing benchmarking packages (ESMValTool, ILAMB, PMP).

## Design goals

The key objectives that shaped the design:

- **Modular.** This is a community project.
  It should be easy for existing and future benchmarking packages to integrate with the framework,
  and the developer experience for providers is paramount.
- **Extensible.** The AFT targets a small subset of possible diagnostics.
  The set will grow over time in data volume and complexity.
- **Reusable.** The REF targets multiple deployment environments and a range of users,
  so components should be reusable in different contexts.
  Not every component needs to be installed for a usable system.
  A headless deployment without the API and portal is fully supported.

And the constraints:

- Deployment environments at modelling centres cannot be controlled,
  so the REF must support several ways of running (local process pool, task queue, HPC schedulers).
- Different users have different areas of interest,
  which drives the ability to run the REF locally rather than only as a centralised service.
- The development is Python based to be accessible to the research community.

The design principles that follow from these:

- **Separation of concerns.** Providers focus on the science.
  The framework tracks state and decides what to run.
  A provider developer needs in-depth knowledge of only a small part of the system.
- **Resilience.** One bad input dataset must not cause a cascading failure.
  Failures are recorded per execution and retried according to their failure class.

## System context

The REF sits between the data archives and the people who want evaluation results.

```mermaid
C4Context
    title Climate REF system context
    Person(operator, "Operator", "Runs the REF at a modelling centre or evaluation hub")
    Person(researcher, "Researcher", "Explores evaluation results")
    System(ref, "Climate REF", "Ingests dataset metadata, solves for required diagnostics and executes them")
    System(refapp, "REF web application", "Serves results through a REST API and a web portal")
    System_Ext(esgf, "ESGF", "Published CMIP and obs4MIPs datasets")
    System_Ext(refdata, "Reference data stores", "obs4REF and provider reference data registries")
    System_Ext(science, "Diagnostic packages", "ESMValTool, ILAMB, PMP")
    Rel(operator, ref, "Ingests data and triggers solves", "CLI")
    Rel(ref, esgf, "Fetches datasets", "intake-esgf")
    Rel(ref, refdata, "Fetches reference data", "HTTPS")
    Rel(ref, science, "Executes diagnostics with")
    Rel(refapp, ref, "Reads database and result files")
    Rel(researcher, refapp, "Browses results", "HTTPS")
```

Two systems are under our control:

- **Climate REF** (this repository) is the compute engine.
  It indexes datasets, decides which diagnostics need to run, runs them, and records the results.
- The **[REF web application](https://github.com/Climate-REF/ref-app)** is a separate repository
  containing a read-only REST API and a React frontend for browsing the results.
  It is deployable at modelling centres as well as centrally.

The science itself lives in the external benchmarking packages.
The REF does not perform any scientific calculations.

## Containers

Zooming in one level, these are the deployable pieces and stores that make up a full installation.

```mermaid
C4Container
    title Climate REF containers
    Person(operator, "Operator")
    Person(researcher, "Researcher")
    System_Boundary(ref, "Climate REF") {
        Container(cli, "ref CLI", "Python, typer", "Ingest, solve, providers, executions and test-cases commands")
        ContainerDb(db, "Database", "SQLite or PostgreSQL", "Datasets, executions and metric values, owned by alembic migrations")
        Container(results, "Results store", "Filesystem", "Execution output bundles, figures and logs")
        Container(envs, "Provider environments", "conda (micromamba)", "One isolated environment per diagnostic provider")
        Container(workers, "Celery workers", "Python, Redis broker", "Optional distributed execution, one queue per provider")
    }
    System_Boundary(app, "REF web application") {
        Container(api, "Backend API", "FastAPI", "Read-only REST API over the REF database and results")
        Container(spa, "Frontend", "React, TanStack", "Single page app using the generated OpenAPI client")
    }
    System_Ext(esgf, "ESGF and reference data stores", "Dataset sources")
    Rel(operator, cli, "Runs", "shell")
    Rel(cli, db, "Reads and writes", "SQLAlchemy")
    Rel(cli, envs, "Executes diagnostics in", "subprocess")
    Rel(cli, workers, "Dispatches executions to", "Celery over Redis")
    Rel(workers, envs, "Execute diagnostics in")
    Rel(workers, db, "Persist results")
    Rel(cli, results, "Writes outputs")
    Rel(cli, esgf, "Fetches data from")
    Rel(api, db, "Reads", "SQLAlchemy, read-only")
    Rel(api, results, "Streams files from")
    Rel(spa, api, "Calls", "JSON over HTTPS")
    Rel(researcher, spa, "Uses")
```

Notes on the individual containers:

- The **`ref` CLI** is the single entry point for operating the compute engine.
  All orchestration (ingest, solve, execute, inspect) happens through it.
- The **database** is SQLite by default and PostgreSQL in production deployments.
  The schema is owned by alembic migrations in the `climate-ref` package.
  It records what data is available, what has run, and every scalar and series metric value produced.
- The **results store** is a directory tree of execution outputs:
  CMEC metric and output bundles, figures, data files and logs.
- Each **provider environment** is a conda environment built with micromamba,
  so providers with incompatible dependencies can coexist.
  See `ref providers setup`.
- **Celery workers** are optional.
  The default local executor runs executions in a process pool without any extra services.
  See [Executors](../how-to-guides/executors.md) for the full set of options.

## Package layering

The monorepo is a uv workspace with seven packages.
Source dependencies point in one direction, towards `climate-ref-core`.

```mermaid
flowchart TD
    celery["climate-ref-celery<br/>task queue adapter"]
    app["climate-ref<br/>CLI, config, database, solver, executors, results"]
    core["climate-ref-core<br/>Diagnostic, DataRequirement, constraints, Executor, CMEC bundles"]
    providers["climate-ref-example / -esmvaltool / -pmp / -ilamb<br/>diagnostic providers"]
    refapp["ref-app backend<br/>FastAPI read layer"]
    celery --> app
    app --> core
    providers --> core
    refapp --> app
    refapp --> core
    app -. "entry points, FQN strings only" .-> providers
```

| Package | Role | Depends on |
| --- | --- | --- |
| `climate-ref-core` | Domain abstractions: `Diagnostic`, `DataRequirement`, constraints, the `Executor` protocol, CMEC bundle handling | third-party only |
| `climate-ref` | The application: CLI, configuration, database, solver, executors, results read layer | `climate-ref-core` |
| `climate-ref-celery` | Celery app and `CeleryExecutor` for distributed execution | `climate-ref`, `climate-ref-core` |
| provider packages | Thin wrappers exposing each benchmarking package's diagnostics | `climate-ref-core` |
| ref-app backend | Read-only API over the REF database and results (separate repository) | `climate-ref`, `climate-ref-core` |

Three properties of this layering do most of the architectural work:

- `climate-ref-core` is a dependency leaf.
  It imports no other REF package and knows nothing about the database.
  Providers can be written and tested against it alone.
- Providers are plugins.
  Each registers itself through the `climate-ref.providers` entry point group,
  and the application refers to providers only by name in configuration, never by import.
  Installing a provider package makes it available.
  Enabling it is a configuration decision.
- Executors are resolved from configuration by fully qualified name against the
  [Executor][climate_ref_core.executor.Executor] protocol,
  so new execution backends plug in without changes to the framework.

## Inside the compute engine

The `climate-ref` package contains the components that orchestrate the workflow.

```mermaid
C4Component
    title Compute engine components (climate-ref package)
    Container_Boundary(app, "climate-ref") {
        Component(cli, "cli", "typer", "Thin command layer that parses arguments and delegates")
        Component(config, "config", "environs, TOML", "Configuration, executor and provider selection")
        Component(adapters, "dataset adapters", "pandas", "Find, parse and validate CMIP6, CMIP7, obs4MIPs and PMP datasets")
        Component(catalog, "data catalog", "pandas", "DataFrame view of the ingested datasets per source type")
        Component(registry, "provider registry", "", "Loads providers from entry points and registers them")
        Component(solver, "solver", "", "Matches catalogs against data requirements to produce executions")
        Component(executors, "executors", "", "Local, synchronous and HPC execution backends")
        Component(handling, "result handling", "", "Parses CMEC bundles, ingests metric values, copies outputs")
        Component(reader, "results reader", "", "Read-layer facade for querying stored results")
        Component(models, "models and database", "SQLAlchemy, alembic", "ORM models, migrations and session handling")
    }
    Rel(cli, adapters, "ingest")
    Rel(cli, solver, "solve")
    Rel(config, registry, "provider FQNs")
    Rel(solver, catalog, "loads")
    Rel(solver, registry, "iterates diagnostics of")
    Rel(solver, executors, "submits executions to")
    Rel(executors, handling, "hands results to")
    Rel(adapters, models, "registers datasets")
    Rel(solver, models, "creates execution groups")
    Rel(handling, models, "writes metric values")
    Rel(reader, models, "queries")
```

- **CLI** (`climate_ref.cli`): typer commands grouped by noun
  (`datasets`, `solve`, `providers`, `executions`, `diagnostics`, `db`, `test-cases`).
  The commands stay thin and delegate to the components below.
- **Configuration** (`climate_ref.config`): a TOML file under `REF_CONFIGURATION`,
  overridable per field through `REF_*` environment variables.
  It selects the executor and the enabled providers by fully qualified name.
- **Dataset adapters** (`climate_ref.datasets`): one adapter per source type
  (CMIP6, CMIP7, obs4MIPs, PMP climatologies).
  An adapter finds local files, parses their metadata into a catalog DataFrame,
  validates it, and registers datasets and files in the database.
  Reference datasets (obs4REF and ESMValTool reference data) are ingested through a separate
  declarative path driven by the providers rather than through these adapters.
  See [Datasets](datasets.md) for the parsers and the two-phase ingestion design.
- **Solver** (`climate_ref.solver`): the heart of the compute engine.
  It loads the data catalog for each source type,
  applies each diagnostic's [data requirements][climate_ref_core.diagnostics.DataRequirement]
  (facet filters, then grouping, then group constraints),
  and produces the set of executions that need to run.
  Group hashing against previously recorded executions decides what is out of date.
- **Provider registry** (`climate_ref.provider_registry`): loads the configured providers
  through their entry points and records them in the database.
- **Executors** (`climate_ref.executor`): local process pool (default), synchronous (in process),
  and HPC schedulers.
  The Celery executor lives in its own package.
- **Result handling** (`climate_ref.executor.result_handling`): parses the CMEC output bundles,
  ingests scalar and series metric values, registers output files, and copies outputs into the results store.
- **Results reader** (`climate_ref.results`): a read-layer facade for querying stored results as DataFrames,
  used for local analysis and notebooks.
  See [Reading results locally](../how-to-guides/reading-results-locally.md).
- **Test cases and regression** (`climate_ref.cli.test_cases` with `climate_ref_core.regression`):
  per-diagnostic regression baselines captured, compared and gated in CI.
  See [Regression baselines](regression-baselines.md).

## Runtime flows

### Ingest

Ingestion extracts metadata from locally available datasets and indexes it in the database.
The REF only knows about data that has been ingested.

```mermaid
sequenceDiagram
    autonumber
    actor Op as Operator
    participant CLI as ref datasets ingest
    participant Ad as DatasetAdapter
    participant DB as Database
    Op->>CLI: ref datasets ingest --source-type cmip6 PATH
    CLI->>Ad: find_local_datasets(PATH)
    Ad-->>CLI: catalog DataFrame
    CLI->>Ad: validate_data_catalog()
    loop each dataset in the catalog
        CLI->>Ad: register_dataset()
        Ad->>DB: write Dataset and DatasetFile rows
    end
    CLI-->>Op: ingestion summary
```

### Solve and execute

A solve compares the ingested datasets against every diagnostic's data requirements,
records the resulting execution groups and executions,
and hands the out-of-date executions to the configured executor.

```mermaid
sequenceDiagram
    autonumber
    actor Op as Operator
    participant CLI as ref solve
    participant Solver as ExecutionSolver
    participant Reg as ProviderRegistry
    participant DB as Database
    participant Ex as Executor
    participant Diag as Diagnostic (provider env)
    participant RH as Result handling
    Op->>CLI: ref solve
    CLI->>Solver: solve_required_executions()
    Solver->>Reg: build_from_config()
    Solver->>DB: load data catalogs
    loop each provider / diagnostic
        Solver->>Solver: apply filters, group_by, constraints
        Solver->>DB: create or update ExecutionGroup / Execution
        Solver->>Ex: run(definition)
    end
    Ex->>Diag: execute (conda env subprocess)
    Diag-->>Ex: output bundle (CMEC JSON)
    Ex->>RH: handle_execution_result()
    RH->>DB: ingest scalar and series metric values
    RH->>RH: copy outputs to results store
    Ex-->>CLI: join()
```

The executor boundary is crossed with plain data structures.
The solver passes an [ExecutionDefinition][climate_ref_core.diagnostics.ExecutionDefinition]
describing the datasets and output location,
and receives an [ExecutionResult][climate_ref_core.diagnostics.ExecutionResult]
built from the CMEC bundles the diagnostic wrote.
Outputs follow the
[Earth System Metrics and Diagnostics Standards (EMDS)](https://github.com/Earth-System-Diagnostics-Standards/EMDS).

### Distributed execution with Celery

With the Celery executor, executions are dispatched to long-lived workers through a Redis broker.
Each provider gets its own queue and its own worker image,
so provider environments stay isolated and can be updated independently.

```mermaid
sequenceDiagram
    autonumber
    participant Solver as Solver process
    participant CE as CeleryExecutor
    participant Redis as Redis broker
    participant W as Provider worker
    participant DB as Database
    Solver->>CE: run(definition)
    CE->>Redis: send_task("provider.diagnostic", definition)
    Redis->>W: deliver task
    W->>W: execute_locally() in provider environment
    W-->>Redis: ExecutionResult
    Redis->>W: handle_result callback
    W->>DB: handle_execution_result()
    Solver->>CE: join()
```

See [Docker deployment](../how-to-guides/docker_deployment.md) for the containerised form of this setup,
and [Running on HPC](../how-to-guides/hpc_executor.md) for scheduler-based execution.

## The web application

The [ref-app](https://github.com/Climate-REF/ref-app) repository provides the visualisation layer:
a FastAPI backend and a React frontend.
It is currently coupled to the AFT deployment of the REF and is best treated as a reference implementation.

```mermaid
C4Component
    title Web application backend components (ref-app)
    Container_Boundary(api, "Backend API (ref_backend)") {
        Component(routes, "api/routes", "FastAPI routers", "diagnostics, executions, datasets, explorer, aft, results, utils")
        Component(deps, "api/deps", "FastAPI Depends", "AppContext bundling session, config, settings and provider registry")
        Component(services, "core services", "Python", "Metric value filtering, outlier detection, collections, AFT index")
        Component(dtos, "models", "pydantic", "Response DTOs built from ORM objects")
        Component(static, "static metadata", "YAML", "AFT collection content and per-provider diagnostic overrides")
    }
    Container_Boundary(lib, "climate-ref library") {
        Component(refmodels, "climate_ref.models", "SQLAlchemy", "Shared ORM models")
        Component(refconfig, "climate_ref.config", "", "REF configuration and paths")
    }
    ContainerDb_Ext(db, "REF database", "SQLite or PostgreSQL", "Populated by the compute engine")
    Container_Ext(results, "Results store", "Filesystem")
    Rel(routes, deps, "resolves")
    Rel(routes, services, "delegates to")
    Rel(routes, dtos, "serialises with")
    Rel(services, refmodels, "queries")
    Rel(routes, refmodels, "queries")
    Rel(dtos, static, "applies overrides from")
    Rel(refmodels, db, "reads")
    Rel(routes, results, "streams files from")
    Rel(refconfig, results, "locates")
```

Key characteristics:

- The backend is a **read-only consumer** of the REF database.
  It opens the database without running migrations
  (SQLite can be opened immutable so the state volume can be mounted read-only)
  and reports the results the compute engine has already produced.
- It depends on the `climate-ref` library directly for configuration, models and the provider registry.
  The database schema remains owned by `climate-ref` migrations.
- **AFT display metadata** (collection descriptions, plain-language summaries, explorer cards)
  is maintained as YAML under `backend/static/` rather than in the database,
  so scientific content can be edited without touching either the schema or the frontend.
- The frontend consumes a **TypeScript client generated from the OpenAPI schema**.
  Changing the API means regenerating the client, which keeps the two sides honest.
- Result files (figures, logs, bundles, archives) are streamed straight from the results store on disk.

## Known boundaries and debt

The layering above is enforced by convention and review, not by tooling,
and a few boundaries are weaker than the diagrams suggest:

- The solver both computes the required executions and reads and writes the database in the same module.
  The pure solving logic already yields plain DTOs, so a cleaner split is available.
- `CondaDiagnosticProvider` places conda environment management inside `climate-ref-core`,
  which drags process and download concerns into the innermost package.
- The ref-app backend queries the `climate-ref` ORM models directly rather than a published read contract,
  which couples the two repositories through the database schema.
  The `climate_ref.results` reader is the seam for narrowing that contract,
  and adoption has started with the metric value endpoints
  ([ref-app #39](https://github.com/Climate-REF/ref-app/pull/39)).
  The remaining routes and DTO builders still query the ORM directly.

These are recorded here so that readers do not mistake the target picture for the current one.
