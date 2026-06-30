# Climate REF (Rapid Evaluation Framework)

[![PyPI version](https://badge.fury.io/py/climate-ref.svg)](https://badge.fury.io/py/climate-ref)
[![Documentation Status](https://readthedocs.org/projects/climate-ref/badge/?version=latest)](https://climate-ref.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/downloads/)

**Status**: This project is in active development. We expect to be ready for beta releases in Q2 2025.

The Rapid Evaluation Framework (REF) is a set of Python packages that provide the ability to manage the execution of calculations against climate datasets.
The aim is to be able to evaluate climate data against a set of reference data in near-real time as datasets are published,
and to update any produced data and figures as new datasets become available.
This is somewhat analogous to a CI/CD pipeline for climate data.

## Installation

```bash
pip install climate-ref
```

If you want to use the diagnostic providers for the Assessment Fast Track, you can install them with:

```bash
pip install "climate-ref[aft-providers]"
```

## Quick Start

```bash
# Ingest some observation datasets
ref datasets fetch-data --registry obs4ref --output-directory datasets/obs4ref
ref datasets fetch-data --registry sample-data --output-directory datasets/sample-data

# Run metrics against your climate data
ref solve
```

## Features

- Real-time evaluation of climate datasets
- Support for multiple metrics providers (PMP, ILAMB, ESMValTool)
- Distributed processing capabilities
- Extensible architecture for adding new metrics providers
- Command-line interface for easy interaction

## Documentation

For detailed documentation, please visit [https://climate-ref.readthedocs.io/](https://climate-ref.readthedocs.io/)

## Contributing

REF is a community project, and we welcome contributions from anyone. Please see our [Contributing Guide](https://climate-ref.readthedocs.io/en/latest/contributing/) for more information.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
