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
or set of datasets with the aim for benchmarking the performance of different models.
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

## Execution Environments

The REF aims to support the execution of metrics in a variety of environments.
This includes local execution, testing, cloud-based execution, and execution on HPC systems.

The currently supported execution environments are:

* Local

The following environments are planned to be supported in the future:

* Kubernetes (for cloud-based execution)
* Subprocess (for HPC systems)

The selected executor is defined using the `REF_EXECUTOR` environment variable.
See the [Configuration](configuration.md) page for more information.
