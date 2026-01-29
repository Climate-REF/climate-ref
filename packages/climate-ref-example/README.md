# climate-ref-example

[![PyPI version](https://badge.fury.io/py/climate-ref-example.svg)](https://badge.fury.io/py/climate-ref-example)
[![Documentation Status](https://readthedocs.org/projects/climate-ref/badge/?version=latest)](https://climate-ref.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/downloads/)

This package provides an example implementation of a diagnostic provider for the Climate REF (Rapid Evaluation Framework).
It serves as a template and reference for developers who want to create their own diagnostic providers.

## Installation

```bash
pip install climate-ref-example
```

## Features

- Example implementation of a diagnostic provider
- Global mean timeseries diagnostic demonstration
- Complete implementation of all required interfaces
- Test data specification for reproducible testing

## Usage

Enable the provider in your `ref.toml` configuration:

```toml
[[diagnostic_providers]]
provider = "climate_ref_example:provider"
```

Then run diagnostics:

```bash
ref solve --provider example
```

## Example Diagnostic

The package implements a `GlobalMeanTimeseries` diagnostic that calculates the annual mean global mean timeseries for CMIP6 datasets:

```python
from climate_ref_core.providers import DiagnosticProvider
from climate_ref_core.diagnostics import Diagnostic, DataRequirement, ExecutionDefinition, ExecutionResult
from climate_ref_core.datasets import FacetFilter, SourceDatasetType

class GlobalMeanTimeseries(Diagnostic):
    """Calculate the annual mean global mean timeseries for a dataset."""

    name = "Global Mean Timeseries"
    slug = "global-mean-timeseries"

    data_requirements = (
        DataRequirement(
            source_type=SourceDatasetType.CMIP6,
            filters=(FacetFilter(facets={"variable_id": ("tas", "rsut")}),),
            group_by=("source_id", "variable_id", "experiment_id", "variant_label"),
        ),
    )
    facets = ("source_id", "variable_id", "experiment_id", "variant_label", "region", "metric", "statistic")

    def execute(self, definition: ExecutionDefinition) -> None:
        # Perform the diagnostic calculation
        ...

    def build_execution_result(self, definition: ExecutionDefinition) -> ExecutionResult:
        # Build and return the execution result
        ...

# Register diagnostics with a provider
provider = DiagnosticProvider("Example", __version__)
provider.register(GlobalMeanTimeseries())
```

## Creating Your Own Provider

Use this package as a template:

```bash
cp -r packages/climate-ref-example packages/climate-ref-myprovider
```

See the [Adding Custom Diagnostics](https://climate-ref.readthedocs.io/en/latest/how-to-guides/adding_custom_diagnostics/) guide for detailed instructions.

## Documentation

For detailed documentation, please visit [https://climate-ref.readthedocs.io/](https://climate-ref.readthedocs.io/)

## Contributing

Contributions are welcome! Please see the main project's [Contributing Guide](https://climate-ref.readthedocs.io/en/latest/development/) for more information.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
