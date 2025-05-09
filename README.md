# Rapid Evaluation Framework (REF)

<!--- --8<-- [start:description] -->

**Status**: This project is in active development. We expect to be ready for beta releases in Q2 2025.

The Rapid Evaluation Framework(REF) is a set of Python packages that provide the ability to manage the execution of calculations against climate datasets.
The aim is to be able to evaluate climate data against a set of reference data in near-real time as datasets are published,
and to update any produced data and figures as new datasets become available.
This is somewhat analogous to a CI/CD pipeline for climate data.

REF is a community project, and we welcome contributions from anyone.

## Deployments

The concept of the REF was proposed by the CMIP Model Benchmarking Task Team (MBTT),
to enable the near-realtime evaluation of CMIP7 Assessment Fast Track (AFT) datasets.
This particular deployment will feature a set of diagnostics selected by
the MBTT in consultation with the community (URL with the CMIP7 AFT configuration to come).

The individual components of the REF have been designed to be useful for
applications past the lifetime of the CMIP7 AFT.
These components can be extended to support evaluating other types of climate data.
The REF is agnostic to what types of data and metrics are to be performed.


[![CI](https://github.com/Climate-REF/climate-ref/actions/workflows/ci.yaml/badge.svg?branch=main)](https://github.com/Climate-REF/climate-ref/actions/workflows/ci.yaml)
[![Coverage](https://codecov.io/gh/Climate-REF/climate-ref/branch/main/graph/badge.svg)](https://codecov.io/gh/Climate-REF/climate-ref)
[![Docs](https://readthedocs.org/projects/climate-ref/badge/?version=latest)](https://climate-ref.readthedocs.io)

**PyPI :**
[![PyPI](https://img.shields.io/pypi/v/climate-ref.svg)](https://pypi.org/project/climate-ref/)
[![PyPI: Supported Python versions](https://img.shields.io/pypi/pyversions/climate-ref.svg)](https://pypi.org/project/climate-ref/)

**Other info :**
[![Licence](https://img.shields.io/github/license/Climate-REF/climate-ref.svg)](https://github.com/Climate-REF/climate-ref/blob/main/LICENCE)
[![SPEC 0 — Minimum Supported Dependencies](https://img.shields.io/badge/SPEC-0-green?labelColor=%23004811&color=%235CA038)](https://scientific-python.org/specs/spec-0000/)
[![Last Commit](https://img.shields.io/github/last-commit/Climate-REF/climate-ref.svg)](https://github.com/Climate-REF/climate-ref/commits/main)
[![Contributors](https://img.shields.io/github/contributors/Climate-REF/climate-ref.svg)](https://github.com/Climate-REF/climate-ref/graphs/contributors)

<!--- --8<-- [end:description] -->

## Installation

<!--- --8<-- [start:installation] -->

CMIP Rapid Evaluation Framework can be installed with pip, mamba or conda
(mamba and conda releases are not yet implemented):


```bash
pip install climate-ref[metrics]
mamba install -c conda-forge climate-ref
conda install -c conda-forge climate-ref
```

<!--- --8<-- [end:installation] -->

## Getting started
<!--- --8<-- [start:getting-started] -->

### As a diagnostics provider

Diagnostic providers are the core of the REF.
They define the diagnostics that will be calculated and the data that will be used to calculate them,
by providing a consistent interface for the REF to interact with.

These diagnostic providers can be run as standalone applications or as part of the REF.
This allows them to be used in other packages or applications that may not be using the REF compute engine.

### As a modelling center

The REF is designed to enable Modelling Centers to quickly evaluate their data against a set of reference data.
The data under test here may not be published to ESGF yet,
but the REF can still be used to evaluate it.

The REF requires some reference data to be available to run the diagnostics.
Some of the reference datasets needed by the REF are available on ESGF yet.
The following command will download the reference datasets needed by the REF and store them in a local directory (`datasets/obs4ref`) as well as some sample CMIP6 datasets that we used in our test suite:

```bash
ref datasets fetch-data --registry obs4ref --output-dir datasets/obs4ref
ref datasets fetch-data --registry sample-data --output-dir datasets/sample-data
```

These datasets can then be ingested into the REF and the metrics solved using:

```bash
uv run ref datasets ingest --source-type cmip6 datasets/sample-data/CMIP6/
uv run ref datasets ingest --source-type obs4mips datasets/obs4ref
uv run ref solve
```

The executed metrics can then be viewed using the `ref executions list-groups` and `ref executions inspect` commands.

### As a devops engineer

The REF can also be deployed as a standalone set of services that don't require any user interaction.
This is useful for running the REF to automatically evaluate data as it is published to ESGF.

The REF can be run as a set of docker containers, with each service running in its own container.
This allows for easy scaling and deployment of the REF.
These docker containers are not yet published, but can be built from the source code.

```
docker-compose build
```

An example docker-compose file is provided in the repository for the CMIP7 AFT deployment of the REF,
using the Celery executor and Redis as the message broker.
This can be run with:

```
docker-compose up
```

<!--- --8<-- [end:getting-started] -->

Full documentation can be found at:
[climate-ref.readthedocs.io](https://climate-ref.readthedocs.io/en/latest/).
We recommend reading the docs there because the internal documentation links
don't render correctly on GitHub's viewer.


### For contributors

<!--- sec-begin-installation-dev -->

For information on how to contribute see https://climate-ref.readthedocs.io/en/latest/development/.

<!--- sec-end-installation-dev -->
