# climate-ref-celery

[![PyPI version](https://badge.fury.io/py/climate-ref-celery.svg)](https://badge.fury.io/py/climate-ref-celery)
[![Documentation Status](https://readthedocs.org/projects/climate-ref/badge/?version=latest)](https://climate-ref.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/downloads/)

This package provides Celery task generation and worker management for the Climate REF (Rapid Evaluation Framework).

The `climate_ref_celery` package provides a CLI tool(`ref-celery`) to start a worker instance for a diagnostics provider.
This worker instance will listen for tasks related to a provider and execute them.
The compute engine worker will then collect the results of these tasks and store them in the database.
This allows for the REF to be run in a distributed manner,
with multiple workers running on different machines with a centrally managed database.

## Installation

```bash
pip install climate-ref-celery
```

## Features

- Distributed task execution for metrics providers
- Worker management and monitoring
- Automatic task routing and scheduling
- Results collection and storage
- Support for multiple worker instances

## Usage

### Starting a Worker

To start a worker instance for a specific metrics provider:

```bash
# Using the standalone CLI
ref-celery start-worker --package climate_ref_pmp

# Or using the main REF CLI
ref celery start-worker --package climate_ref_pmp
```

### Configuration

Each worker instance may not share the same configuration as the orchestrator.
This is because the worker may be running on a different machine with different resources available or
directories.

Each worker instance requires access to a shared input data directory and the output directory.
If the worker is deployed as a docker container these directories can be mounted as volumes.

Each worker instance can be configured independently. Required configurations include:

- Shared input data directory
- Output directory for results
- Database connection settings


## Documentation

For detailed documentation, please visit [https://climate-ref.readthedocs.io/](https://climate-ref.readthedocs.io/)

## Contributing

Contributions are welcome! Please see the main project's [Contributing Guide](https://climate-ref.readthedocs.io/en/latest/contributing/) for more information.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
