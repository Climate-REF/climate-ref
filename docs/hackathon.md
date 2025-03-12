# Hackathon 2025

**Met Office, Exeter, UK, 10 - 14 May 2025**

This hybrid hackathon is being run by the Model Benchmarking Task Team and the REF delivery team. We welcome attendance from technical and domain scientists from modelling centres involved in the CMIP AR7 Fast Track, observation dataset providers, as well as ESGF nodes and developers.

During the hackathon, there will also be dedicated drop-ins for wider community interest:

* **10 March 14:00 UTC – 15:00 UTC**: Hackathon launch - providing a brief overview of the REF and a status update by the AR7 FT REF delivery team
* **11 March 17:00 – 18:00 UTC**: Modelling Centres
* **13 March 08:00 – 09:00 UTC**: Modelling Centres
* **13 March 11:00 – Midday UTC**: Observation dataset providers

## Technical Requirements

* A laptop with Python 3.10 or later installed
* A GitHub account
* Docker installed (Optional)
* [uv](https://docs.astral.sh/uv) installed

## What can you do before the hackathon?

Before attending the hackathon, it would be useful to clone the package and set up your local environment by following the [development installation](development.md#development-installation) instructions.

Additionally, clone the [sample data repository](https://github.com/Climate-REF/ref-sample-data). Depending on your area of interest, you may wish to add additional sample data to the test suite.

After installing the database, you can run the test suite using `make test` to ensure that everything is working as expected. This will fetch the sample data and run the tests.

Some metric providers require additional test data that isn't in Obs4MIPs.
We are working on adding these datasets to obs4MIPs before the  launch of the AR7 FT REF.
To help save time during the hackathon, run the following commands beforehand to cache the ilamb and iomb reference datasets (~4GB).

```bash
uv run python scripts/fetch-ilamb-data.py ilamb.txt
uv run python scripts/fetch-ilamb-data.py iomb.txt
```

If there are any issues with the installation, please raise an issue in the [issue tracker](https://github.com/Climate-REF/climate-ref/issues) so that we can help you get set up.

### Additional Data

We welcome the testing of ingesting additional local datasets into the REF.
We currently support ingesting CMIP6-like and obs4MIPs datasets,
but are interested in hearing about other datasets that you would like to see supported.

We have tested a subset of CMIP6 data,
but are particularly interested in any non-published datasets that you may have access to.
This will help us to ensure that the REF can support a wide range of datasets.

The metrics that are currently available in the REF are relatively limited so not all datasets will be useful for testing.

### Reading Material

It is recommended to read through the [developer documentation](development.md) to get an understanding of the REF and how we collaborate.

For those interested in learning more about the REF,
we recommend reading the [Architecture design document (background/architecture.md).
This outlines the design of the REF and provides some background about the project.

## Brief technical overview of the REF

The REF is a Python package that provides a framework for running benchmarking climate models. It is designed to be flexible and extensible, allowing users to define their own metrics and ingest their own climate datasets.

The REF consists of 4 main steps, which are shown in the diagram below:

```mermaid
flowchart LR
    Ingest --> Solve
    Solve --> Execute
    Execute --> Visualise
```

* **Ingest** Ingesting datasets that can be used for analysis into a local database. This includes support for CMIP6, Obs4MIPs, and other observational datasets.
* **Solve** Using the ingested datasets and the data requirements from the metrics, the REF determines which metrics need to be executed.
* **Execute** The metrics are executed and the results collated. We support multiple different ways of running metrics that may be useful for different use cases.
* **Visualise** The results of the metrics are visualised. This can be in the form of plots, tables, or other outputs.

### Metrics
At the core of the REF is the [Metric][cmip_ref_core.metrics.Metric] protocol.
This protocol defines the common interface that all metrics must implement.
A metric defines the different datasets that a metric requires (see [dataset-selection](how-to-guides/dataset-selection.py)), and how to calculate a value from them.
How a metric is actually calculated depends on which metrics provider the metric comes from.

The rest of the complexity that comes from figuring out which datasets to use, how to run the metric, and how to visualise the results is handled by the REF.

### Metric Providers
Metrics are grouped into packages, one for each of the metric providers selected for the AR7 Fast Track.
These metric packages (ESMValTool, ILAMB and PMP) each have different ways of calculating metrics.

For some metric providers (ILAMB and ESMValTool),
an additional conda environment will be required
to run the metrics locally.
This is still a work in progress ([#117](https://github.com/Climate-REF/climate-ref/pull/117))
and is expected to be available to use by the time of the hackathon.

### Output

The output of a metric calculation can be a range of different outcomes:

* Scalar values
* Timeseries
* Plots
* Tables
* HTML reports

To support the capture of these different outputs in a standardised way,
the REF uses the [Earth System Metrics and Diagnostics Standards (EMDS)](https://github.com/Earth-System-Diagnostics-Standards/EMDS).
A REF-specific extension to this format will be developed as
part of this process.

These outputs will be ingested into a database
and made available through an API/web interface or CLI tool.
This is still early work so input into how we can expose the results in a meaningful way is welcome.

### Compute Engine

The REF is designed to be run on a local machine,
but we are also working on support for running the REF on cloud-based systems and HPC systems.

The main interface with the REF application for local users,
is through the command line interface tool, `ref`.

```
$ ref --help
```

### Executors

An [executor][cmip_ref_core.executor.Executor] is responsible for running a metric calculation in an aync manner.
The REF supports multiple executors,
each of which is responsible for running a metric calculation in a different way.

* [Local][cmip_ref.executor.local.LocalExecutor] - Runs the metric calculation locally
* [Celery][cmip_ref_celery.executor.CeleryExecutor] - Runs the metric calculation using Celery

One of the outcomes of this Hackathon will be to add support for running metrics on HPC systems.

## Finally

We are working towards a beta release of the REF in the coming months,
so this project is under heavy development.
There may be aspects that do not work as expected,
but welcome any possible involvement.
Please raise [issues](https://github.com/Climate-REF/climate-ref/issues)
if anything doesn’t work as expected or if you have any features that you would like to see implemented.
