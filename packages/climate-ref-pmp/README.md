# climate-ref-pmp

[![PyPI version](https://badge.fury.io/py/climate-ref-pmp.svg)](https://badge.fury.io/py/climate-ref-pmp)
[![Documentation Status](https://readthedocs.org/projects/climate-ref/badge/?version=latest)](https://climate-ref.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/badge/python-3.12%2B-blue)](https://www.python.org/downloads/)

This package integrates the [PCMDI Metrics Package (PMP)](https://github.com/PCMDI/pcmdi_metrics) as a metrics provider for the Climate REF (Rapid Evaluation Framework).

## Installation

```bash
pip install climate-ref-pmp
```

## Prerequisites

- PMP must be installed in your environment
- Required climate data files must be available in the specified input directory

## Usage

PMP has some additional preprocessed climatology datasets that should be ingested so the REF compute engine
can find them.

```
ref datasets fetch-data --registry pmp-climatology --output directory data/pmp-climatology
ref datasets ingest --source-type pmp-climatology data/pmp-climatology
```

For detailed usage instructions, see [Running Metrics Locally](https://climate-ref.readthedocs.io/en/latest/how-to-guides/running-metrics-locally/).

## Documentation

For detailed documentation, please visit:
- [Climate REF Documentation](https://climate-ref.readthedocs.io/)
- [PMP Documentation](http://pcmdi.github.io/pcmdi_metrics/)

## Contributing

Contributions are welcome! Please see the main project's [Contributing Guide](https://climate-ref.readthedocs.io/en/latest/contributing/) for more information.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
