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
This metadata, in combination with the data requirements from a Diagnostic,
are used to determine which new diagnostic executions are required.

The REF requires that input datasets are CMOR-compliant,
but does not verify any of the attributes that may be in the CMIP6 or CMIP7 controlled vocabularies.
This is to allow for local datasets to be ingested into the REF that may never be intended for publication as part of CMIP7.


## Diagnostic Providers

The REF aims to support a variety of different sources of diagnostics providers by providing a generic interface for running diagnostics.
This allows for the use of different diagnostic providers to be used interchangeably.
These providers are responsible for performing the calculations and analyses.
We recommend that the calculations are encapsulated in a separate library,
and the diagnostic provider consists of a thin wrapper around the library.

Each diagnostic provider generally provides a number of different diagnostics that can be calculated.
An example implementation of a diagnostic provider is provided in the `climate-ref-example` package.

### Diagnostics

A diagnostic represents a specific calculation or analysis that can be performed on a dataset
or group of datasets with the aim for benchmarking the performance of different models.
These diagnostic often evaluate specific aspects of the Earth system and are compared against
observations of the same quantities.

A diagnostic depends upon a set of input model data and observation datasets.
Each diagnostic declares the datasets that it requires via [data requirements](../how-to-guides/dataset-selection.py).
The solver uses these requirements to determine if the diagnostic requires execution.


### Execution Groups

When actually running diagnostics with a given set of ingested datasets,
the REF solver will figure out which (set of) datasets fulfill the requirements to run a given diagnostic.
Generally, each given diagnostic can be executed for many different (sets of) datasets,
e.g. model results from different models.
Additionally, there might be multiple versions of datasets,
and a diagnostic will need to be re-executed when new versions of datasets become available.
Within the REF, we group all executions for different versions of datasets together into a diagnostic execution group,
so the diagnostic execution group would be specific to a specific diagnostic and e.g. a specific model.
This enables us to determine if the results for the diagnostic execution group are up to date,
so if the diagnostic is evaluated for the most up-to-date version of the input datasets.

### Execution

Once the solver has determined which datasets are required to run a given diagnostic,
the REF will perform an execution.
This execution will run the diagnostic, with a given set of datasets and produce a number of outputs.

The result of an execution can be a range of different outcomes depending on what is being evaluated.
These outputs can include:

* A single scalar value
* Timeseries
* Plots
* Data files
* HTML reports

These timeseries and scalars are often named diagnostic values.
These values are used to compare the performance of different models,
and are used by the frontend to generate plots and visualisations across the different executions.

The Earth System Metrics and Diagnostics Standards
([EMDS](https://github.com/Earth-System-Diagnostics-Standards/EMDS))
provide a community standard for reporting outputs.
This enables the ability to generate standardised outputs that can be distributed.

After a successful execution,
the outputs from the execution are stored in a database.
These outputs are then made available through an API/web interface or CLI tool.


## Execution Environments

The REF aims to support the execution of diagnostics in a variety of environments.
This includes local execution, testing, cloud-based execution, and execution on HPC systems.

The currently supported execution environments are:

* Local
* Celery

The following environments are planned to be supported in the future:

* Kubernetes (for cloud-based execution)

The selected executor is defined using the `REF_EXECUTOR` environment variable.
See the [Configuration](../configuration.md) page for more information.
