[](){#concepts}
[](){#basic-concepts}

# Concepts

This page explains the ideas and vocabulary behind the Rapid Evaluation Framework (REF).
It is understanding-oriented: it describes *what* the pieces are and *why* they exist,
rather than walking through commands.
For a hands-on introduction, start with the [Getting Started](getting-started/01-configure.md) guide.
For exact commands and options, see the [CLI](cli.md) and [Configuration](configuration.md) references.

The REF is a set of Python packages that manage the execution of evaluation calculations against climate datasets,
somewhat like a CI/CD pipeline for climate data.
Crucially, **the REF does not perform any scientific calculations itself**.
Instead it coordinates the workflow — deciding what needs to be computed, running it, and tracking the results —
and delegates the actual calculations to external **diagnostic providers**.
In short, providers focus on the science of comparison while the REF takes care of the workflow:
determining what to compute, how to compute it, and where the results are stored.

## The evaluation workflow

Before any evaluation runs, you [configure the REF](getting-started/01-configure.md)
and [acquire some datasets](getting-started/02-download-datasets.md).
The evaluation itself then proceeds through four main phases:

```mermaid
flowchart LR
    Ingest --> Solve
    Solve --> Execute
    Execute --> Visualise
```

- **Ingest** — Extract metadata from locally available [datasets](background/datasets.md) and index it in a local database, so the REF knows what data is available.
- **Solve** — Compare the ingested datasets against each diagnostic's data requirements to determine which calculations need to run (and which are already up to date).
- **Execute** — Run the required calculations and collect their outputs. Executions can run locally, on a distributed task queue, or on an HPC system.
- **Visualise** — Explore the results through a web interface, an API, or the CLI.

## Key terms

The REF uses a small amount of jargon consistently throughout the documentation:

| Term | Meaning |
| --- | --- |
| **Diagnostic provider** | A package that supplies evaluation code (e.g. ESMValTool, ILAMB, PMP), declares the diagnostics it offers, and defines the rules for when each should run. The REF also builds the software environment each provider needs. |
| **Diagnostic** | A single calculation or analysis — for example, comparing a model's surface-temperature climatology against an observational reference. |
| **Data requirements** | A declaration, attached to each diagnostic, of which datasets it needs and how they should be grouped. The solver uses these to decide what can run. |
| **Ingest** | The act of registering datasets with the REF by extracting their metadata into the local database. The REF only knows about data that has been ingested. |
| **Solve** | The process of matching ingested datasets to diagnostics' data requirements to produce a work list of executions. |
| **Execution group** | All executions of one diagnostic for one logical grouping of data (e.g. one model + experiment), tracked together across successive dataset versions. |
| **Execution** | A single run of one diagnostic against one specific set of dataset versions, producing outputs. |
| **Metric value** | A scalar (or timeseries) output of an execution, used to compare model performance and to drive plots. |
| **Executor** | The component that decides *where and how* executions run (locally, via Celery, or on HPC). |
| **Dirty flag** | A per-execution-group marker indicating the group needs to be (re)run. |

## Datasets

The REF supports a variety of input datasets, including CMIP6, CMIP7+, obs4MIPs, and other observational datasets.
Datasets fall into two broad roles:

- **Target datasets** — the model output under evaluation (typically CMIP6/CMIP7).
- **Reference datasets** — the observational or reference data that diagnostics compare against (typically obs4MIPs / obs4REF).

When a dataset is ingested, the metadata that uniquely describes it is stored in the database —
for example the model that produced it, the experiment that was run, the variable and its units, and the time period covered.
The available facets (dimensions) of this metadata depend on the dataset type.
This metadata, combined with a diagnostic's data requirements, is what the solver uses to determine which executions are required.

The REF requires that input datasets are CMOR-compliant,
but it does **not** verify the values of attributes against the CMIP6 or CMIP7 controlled vocabularies.
This is deliberate: it allows local, pre-publication, or custom datasets to be evaluated
even if they are never intended for publication to ESGF.

Metadata can be extracted in two ways, trading ingestion speed against completeness.
The **complete parser** opens every netCDF file to read full metadata
and is the current default for both CMIP6 and CMIP7.
The **DRS parser** infers metadata from file paths and directory structure without opening files,
which is dramatically faster on large archives and parallel (e.g. Lustre) file systems;
it is currently opt-in and is expected to become the default in a future release.
With the DRS parser, the remaining metadata is filled in lazily at solve time,
only for the files that actually match a diagnostic's requirements.
See [Datasets in the REF](background/datasets.md) for the full treatment.

/// admonition | Note

Non-CMOR-compliant datasets are not planned for support in the immediate future,
as this would require additional development.
See [issue #299](https://github.com/Climate-REF/climate-ref/issues/299) for the requirements being gathered.

///

## Diagnostic providers

The REF supports a variety of diagnostic providers through a generic interface for running [diagnostics][climate_ref_core.diagnostics.Diagnostic],
so that different providers can be used interchangeably.
Each provider is responsible for performing its calculations;
we recommend that the science is encapsulated in a separate library and that the provider is a thin wrapper around it.

A provider generally exposes several diagnostics.
A minimal reference implementation lives in the [climate-ref-example](https://github.com/Climate-REF/climate-ref/tree/main/packages/climate-ref-example) package,
and providers register themselves through an entry point (see the [API surface](api-surface.md#entry-point-contract)).
Note that installing a provider package makes it *available*;
it must still be enabled by listing it in your configuration before it will run.

For the CMIP7 Assessment Fast Track (AFT), the REF uses these providers:

- [ESMValTool](https://esmvaltool.org/)
- [ILAMB and IOMB](https://ilamb.org/)
- [PMP](https://pcmdi.llnl.gov/research/metrics/)

The REF also manages the software environment each provider requires,
creating isolated conda environments where needed so that providers with incompatible dependencies can coexist.

## Diagnostics

A diagnostic represents a specific calculation or analysis performed on a dataset or group of datasets,
usually to benchmark model performance by comparing against observations of the same quantity.
Each diagnostic implements the [Diagnostic][climate_ref_core.diagnostics.Diagnostic] protocol,
which declares the diagnostic's [data requirements](how-to-guides/dataset-selection.py) and how it is run.
The solver uses those requirements to decide whether the diagnostic can — and needs to — be executed against the available data.

## Solving

A **solve** matches the ingested datasets against every diagnostic's data requirements
and produces the set of executions that need to run.

### Execution groups

A single diagnostic can be executed many times — for different models, experiments, or variables —
and may need re-running when newer versions of its input datasets are ingested.
The REF groups all executions that share a logical identity (but may differ by dataset version) into an **execution group**.

Each group has a unique identifier built from the keys used to group its datasets.
For example, if a diagnostic's data requirements group CMIP6 datasets by `source_id` and `experiment_id`,
a group might be identified as `cmip6_historical_ACCESS-ESM1-5`.
This grouping lets the REF tell whether a group's results are up to date with the latest versions of its inputs.

Each execution group carries a **dirty flag** indicating whether it needs to run:

- A group is dirty when **first created**.
- When **new data is ingested** that changes a group's inputs, the dataset hash changes and the solver schedules a new execution regardless of the flag.
- The flag is **cleared** when an execution succeeds, or when it fails with a **diagnostic error** (a bug in the diagnostic logic) — so a broken diagnostic is not retried indefinitely against the same data.
- A **system error** (out-of-memory, disk full, worker crash) leaves the flag **set**, so the execution is retried automatically on the next solve.

Diagnostic failures are not retried automatically;
operators can re-flag specific groups with `ref executions flag-dirty`,
or retry all failures at once with `ref solve --rerun-failed` (see [Solve](getting-started/04-solve.md)).

### Executions

An **execution** is one run of a single diagnostic against one specific set of dataset versions.
The required execution groups and executions — along with the datasets each needs — are recorded in the database,
which gives the REF a durable record of what has run, what succeeded, and what failed.

## Executing

Once the solver has identified the out-of-date execution groups,
each execution runs its diagnostic against the chosen datasets and produces outputs.

### Outputs and metric values

Depending on what is being evaluated, an execution's outputs can include:

- a single scalar value
- a timeseries
- plots
- data files
- HTML reports

The scalar and timeseries outputs are referred to as **metric values**.
They are used to compare model performance and to generate plots across executions.
Outputs follow the [Earth System Metrics and Diagnostics Standards (EMDS)](https://github.com/Earth-System-Diagnostics-Standards/EMDS),
a community standard for reporting, which makes the results straightforward to distribute and compare.

### Executors and execution environments

How and where an execution runs is encapsulated by an **executor**.
By default the REF uses the **local executor**, which runs executions in a process pool on the machine you invoke it from.
Other executors support distributed and large-scale execution:

- **Local** — process pool on the local machine (default).
- **Celery** — a distributed task queue for parallel execution across workers.
- **HPC** — submission to HPC schedulers (see [Running on HPC](how-to-guides/hpc_executor.md)).
- **Kubernetes** — planned for cloud-based execution.

The executor is selected through the `[executor]` section of your configuration (or the `REF_EXECUTOR` environment variable);
see [Configuration](configuration.md) and the [Executors how-to guide](how-to-guides/executors.md).

## Visualising results

After a successful execution, its outputs are stored and made available through an API, a web interface, or the CLI.
An [example API and frontend](https://github.com/Climate-REF/ref-app) is under active development
and is intended to be deployable at modelling centres as well as used to surface results via ESGF.
A Python API for querying a local set of results is also planned.
For how to locate result files on disk today, see [Visualise](getting-started/05-visualise.md).

## Where to next

- [Getting Started](getting-started/01-configure.md) — configure, ingest, solve, and visualise step by step.
- [Datasets in the REF](background/datasets.md) — dataset types, parsers, and two-phase ingestion in depth.
- [Architecture](background/architecture.md) — how the components fit together.
- [Adding custom diagnostics](how-to-guides/adding_custom_diagnostics.md) — write your own provider.
- [API surface](api-surface.md) — the stable extension points and the provider entry-point contract.
