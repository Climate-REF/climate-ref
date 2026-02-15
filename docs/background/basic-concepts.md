# Basic Concepts

The Climate-REF (Rapid Evaluation Framework) is a comprehensive framework for performing climate model evaluation and benchmarking.

The Climate-REF doesn't perform any calculations itself,
instead delegates these operations to external diagnostic providers.
These providers are responsible for translating a set of datasets

The operation of the Climate-REF is split into four main phases:

```mermaid
flowchart LR
    Ingest --> Solve
    Solve --> Execute
    Execute --> Visualise
```

* **Ingest** Ingesting [datasets](./datasets.md) that can be used for analysis into a local database. This includes support for CMIP6, Obs4MIPs, and other observational datasets.
* **Solve** Using the ingested datasets and the data requirements from the metrics, the REF determines which metrics need to be executed.
* **Execute** The metrics are executed and the results collated. We support multiple different ways of running metrics that may be useful for different use cases.
* **Visualise** The results of the metrics are visualised. This can be in the form of plots, tables, or other outputs.


## Ingest (`ref datasets ingest`)

We require metadata about each of the available datasets to determine which diagnostics should be executed.
The metadata required for each dataset depends on the `Source Type` of the data
(see [datasets](./datasets.md) for more information about what information is used from each dataset source).
We currently support:

* obs4MIPs
* CMIP6

CMIP7-era files will be supported in the near future once we have some example CMORised output.

The `ingest` phase iterates over the local datasets and extracts metadata,
which is then indexed in a database.

For CMIP6 datasets, the default DRS parser extracts metadata from file paths and directory structure
without opening each file, making ingestion very fast even for large archives.
Full metadata (e.g. exact time ranges) is extracted lazily at solve time,
only for datasets that match a diagnostic's requirements.
This two-phase approach dramatically reduces ingestion time on HPC parallel file systems.

The complete parser can be used instead by setting `cmip6_parser: "complete"` in the configuration,
which opens every file during ingestion to extract all metadata upfront.

The REF requires that input datasets are CMOR-compliant,
but does not verify any of the attributes that may be in any CMIP controlled vocabularies.
This is to allow for local datasets to be ingested into the REF that may never be intended for publication via ESGF.


/// admonition | Note

Non-CMOR-compliant datasets are not planned on being supported in the immediate future,
without support as this would require additional development.
There is an [open issue](https://github.com/Climate-REF/climate-ref/issues/299) to capture the requirements
for ingesting non-CMOR-compliant datasets.

///


## Solve (`ref solve`)

### Diagnostic Providers

The REF aims to support a variety of different sources of diagnostics providers by providing a generic interface for running [diagnostics][climate_ref_core.diagnostics.Diagnostic].
This allows for the use of different diagnostic providers to be used interchangeably.
These providers are responsible for performing the calculations and analyses.
We recommend that the calculations are encapsulated in a separate library,
and the diagnostic provider consists of a thin wrapper around the library.

Each diagnostic provider generally provides a number of different diagnostics that can be calculated.
An example implementation of a diagnostic provider is provided in the [climate-ref-example](https://github.com/Climate-REF/climate-ref/tree/main/packages/climate-ref-example) package.

For the Assessment Fast Track (AFT), we are using the following diagnostic providers:

* [ESMValTool](https://esmvaltool.org/)
* [ILAMB and IOMB](https://ilamb.org/)
* [PMP](https://pcmdi.llnl.gov/research/metrics/)

### Diagnostics

A diagnostic represents a specific calculation or analysis that can be performed on a dataset
or group of datasets with the aim for benchmarking the performance of different models.
These diagnostic often evaluate specific aspects of the Earth system and are compared against
observations of the same quantities.

Each diagnostic implements the [Diagnostic][climate_ref_core.diagnostics.Diagnostic] protocol.
This describes the [data requirements](../how-to-guides/dataset-selection.py) of the diagnostic and
how to run the diagnostic.
The solver uses these requirements to determine if the diagnostic requires execution.

## Execution Groups

During the `solve` phase,
the REF solver will figure out which (set of) datasets fulfill the requirements to run a given diagnostic.
Generally, each given diagnostic can be executed for many different (sets of) datasets,
e.g. model results from different models or different variables.
Additionally, there might be multiple versions of datasets,
and a metric will need to be re-executed when new versions of datasets become available.

We group all executions for different versions of datasets together into an execution group.
Each execution group has a unique identifier consisting of the unique keys used to group the datasets together.
For example if a diagnostic's data requirements grouped CMIP6 datasets by `source_id` and `experiment_id`,
then an example execution group would be `cmip6_historical_ACCESS-ESM1-5`.

This enables us to determine if the results for the execution group are up to date,
so if the metric is evaluated for the most up-to-date version of the input datasets.

The required execution groups and executions are stored in a database,
along with the datasets that are required for each execution.

## Execute

Once the solver has determined which execution groups are out of date,
the next step is to perform an execution.
This execution will run the diagnostic, with a given set of datasets and produce a number of outputs.

This execution may be performed in a number of different ways.

The result of an execution can be a range of different outcomes depending on what is being evaluated.
These outputs can include:

* A single scalar value
* Timeseries
* Plots
* Data files
* HTML reports

These timeseries and scalars are often named metric values.
These values are used to compare the performance of different models,
and are used by the frontend to generate plots and visualisations across the different executions.

The Earth System Metrics and Diagnostics Standards
([EMDS](https://github.com/Earth-System-Diagnostics-Standards/EMDS))
provide a community standard for reporting outputs.
This enables the ability to generate standardised outputs that can be distributed.


## Visualise

After a successful execution,
the outputs from the execution are stored in a database.
These outputs are then made available through an API/web interface or CLI tool.

An [example API/frontend for the REF](https://github.com/Climate-REF/ref-app) is being developed,
and will be used to visualise the results of the REF via the ESGF.
This application will be able to be deployed to modelling centers as well.

A Python-based API will also be made available to interact with a local set of results.
