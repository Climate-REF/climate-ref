This part of the project documentation
will focus on an **understanding-oriented** approach.
Here, we will describe the background of the project,
as well as reasoning about how it was implemented.

Points we will aim to cover:

- Context and background on the library
- Why it was created
- Help the reader make connections

We will aim to avoid writing instructions or technical descriptions here,
they belong elsewhere.

## Metric Providers

The REF aims to support a variety of metric providers.
These providers are responsible for performing the calculations and analyses.

Each metric provider generally provides a number of different metrics that can be calculated.
An example implementation of a metric provider is provided in the `ref_metrics_example` package.

### Metrics

A metric represents a specific calculation or analysis that can be performed on a dataset
or group of datasets with the aim for benchmarking the performance of different models.
These metrics often represent a specific aspects of the Earth system and are compared against
observations of the same quantities.

A metric depends upon a set of input model data and observation datasets.

The result of a metric calculation can be a range of different outcomes:

* A single scalar value
* Timeseries
* Plots

The Earth System Metrics and Diagnostics Standards
([EMDS](https://github.com/Earth-System-Diagnostics-Standards/EMDS))
provide a community standard for reporting outputs.
This enables the ability to generate standardised outputs that can be distributed.

## Datasets

The REF aims to support a variety of input datasets,
including CMIP6, CMIP7+, Obs4MIPs, and other observational datasets.

When ingesting these datasets into the REF,
the metadata used to uniquely describe the datasets is stored in a database.
This metadata includes information such as:

* the model that produced the dataset
* the experiment that was run
* the variable and units of the data
* the time period of the data

The facets (or dimensions) of the metadata depend on the dataset type.
This metadata, in combination with the data requirements from a Metric,
are used to determine which new metric executions are required.

## Execution Environments

The REF aims to support the execution of metrics in a variety of environments.
This includes local execution, testing, cloud-based execution, and execution on HPC systems.

The currently supported execution environments are:

* Local

The following environments are planned to be supported in the future:

* Kubernetes (for cloud-based execution)
* Slurm (for HPC systems)
* Celery (for local testing)

The selected executor is defined using the `REF_EXECUTOR` environment variable.
See the [Configuration](../configuration.md) page for more information.

## Metric Execution Groups

When actually running metrics with a given set of ingested datasets, the REF
will figure out which (set of) datasets fulfill the requirements to run a given metric.
Generally, each given metric can be executed for many different (sets of) datasets,
e.g. model results from different models. Additionally, there might be multiple
versions of datasets, and a metric will need to be re-executed when new versions
of datasets become available. Within the REF, we group all executions for different
versions of datasets together into a metric execution group, so the metric execution
group would be specific to a specific metric and e.g. a specific model. This enables us
to determine if the results for the metric execution group are up to date, so
if the metric is evaluated for the most up-to-date version of the input datasets.
