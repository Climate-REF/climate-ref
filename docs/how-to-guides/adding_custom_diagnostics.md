# Adding Custom Diagnostics

/// admonition | Caveat
    type: warning

This workflow is currently untested,
as the core development team focuses on Assessment Fast Track providers.
If you add custom diagnostics, please contribute improvements to this documentation or even better open-source your provider package.

///

The REF delegates all calculations to diagnostic providers.
To add your own diagnostics, you must create a provider package
and implement one or more classes based on the [Diagnostic][climate_ref_core.diagnostics.AbstractDiagnostic] protocol.

This protocol defines the interface that all diagnostics must implement, including:

- `slug`: A unique identifier for the diagnostic.
- `name`: A human-readable name for the diagnostic.
- `data_requirements`: A collection of data requirements needed to run the diagnostic.
- `facets`: The facets that this diagnostic provides metric values for.
- `def execute(self, definition: ExecutionDefinition) -> None`:
  The method that executes the diagnostic logic, taking an `ExecutionDefinition` object as input.
- `def build_execution_result(self, definition: ExecutionDefinition) -> ExecutionResult`:
  The method that builds the execution result, returning an `ExecutionResult` object.

## 1. Scaffold a new provider

Use the [climate-ref-example](https://github.com/Climate-REF/climate-ref/tree/main/packages/climate-ref-example) package as a template:

```bash
cp -r packages/climate-ref-example packages/climate-ref-myprovider
sed -i '' 's/climate_ref_example/climate_ref_myprovider/g' packages/climate-ref-myprovider/**/*.py
```

Rename modules and `pyproject.toml` metadata to match your provider name (e.g., `climate_ref_myprovider`).

## 2. Provider dependencies

You can also install any additional dependencies your diagnostics require in the `pyproject.toml` file.
These dependencies will be installed with the other provider dependencies when the REF is installed.
Instead, it is recommended to use a `conda` environment for your provider.

This involves creating an `environment.yml` file that is used to generate a `conda-lock.yml` lock file.
This lock file contains the exact versions of all dependencies required to run your diagnostics on the different
support environments. The [ESMValTool provider](https://github.com/Climate-REF/climate-ref/tree/main/packages/climate-ref-esmvaltool/src/climate_ref_esmvaltool/requirements) has an example of a `environment.yml` file.

This lockfile can be generated using the `uvx` command line tool, which is part of the `uv` package manager (see the [Development Docs](../development.md) for more information on how to install `uv`).

```bash
uvx conda-lock -p linux-64 -p osx-64 -p osx-arm64 -f packages/climate-ref-myprovider/src/climate_ref_myprovider/requirements/environment.yml
mv conda-lock.yml packages/climate-ref-myprovider/src/climate_ref_myprovider/requirements/conda-lock.yml
```

## 3. Implement Diagnostic classes

Inside your provider package, create classes that implement the [Diagnostic protocol][climate_ref_core.diagnostics.AbstractDiagnostic].

/// admonition | Offline execution requirement
    type: warning

Diagnostics **must not** make network calls during `execute()` or `build_execution_result()`.
The REF is designed to run on HPC compute nodes that typically lack internet access.

All required data must be either:

- Provided via the `definition.datasets` (input data from CMIP6, obs4MIPs, etc.)
- Pre-fetched during provider setup via `fetch_data()` lifecycle hook

If your diagnostic needs reference datasets, climatologies, or auxiliary files,
implement the `fetch_data()` hook on your provider (see [Section 5](#5-provider-lifecycle-hooks-optional)).

///

```python
from climate_ref_core.diagnostics import Diagnostic, ExecutionResult, ExecutionDefinition, DataRequirement
from climate_ref_core.datasets import FacetFilter, SourceDatasetType
from climate_ref_core.constraints import AddSupplementaryDataset, RequireContiguousTimerange
from climate_ref_core.pycmec.metric import CMECMetric
from climate_ref_core.pycmec.output import CMECOutput


class GlobalMeanTimeseries(Diagnostic):
    """
    Calculate the annual mean global mean timeseries for a dataset
    """

    name = "Global Mean Timeseries"
    slug = "global-mean-timeseries"

    data_requirements = (
        DataRequirement(
            source_type=SourceDatasetType.CMIP6,
            filters=(
                FacetFilter(facets={"variable_id": ("tas", "rsut")}),
            ),
            # Run the diagnostic on each unique combination of model, variable, experiment, and variant
            group_by=("source_id", "variable_id", "experiment_id", "variant_label"),
            constraints=(
                # Add cell areas to the groups
                AddSupplementaryDataset.from_defaults("areacella", SourceDatasetType.CMIP6),
                RequireContiguousTimerange(group_by=("instance_id",)),
            ),
        ),
    )
    facets = ("region", "metric", "statistic")

    def execute(self, definition: ExecutionDefinition) -> None:
        """
        Run a diagnostic

        Parameters
        ----------
        definition
            A description of the information needed for this execution of the diagnostic
        """
        # This is where one would hook into however they want to run
        # their benchmarking packages.
        # cmec-driver, python calls, subprocess calls all would work

        input_datasets = definition.datasets[SourceDatasetType.CMIP6]

        # calculation_function would be your own function to process the data
        # calculate_annual_mean_timeseries(input_files=input_datasets.path.to_list()).to_netcdf(
        #     definition.output_directory / "annual_mean_global_mean_timeseries.nc"
        # )
        pass # Replace with your calculation logic

    def build_execution_result(self, definition: ExecutionDefinition) -> ExecutionResult:
        """
        Create a result object from the output of the diagnostic
        """
        # This involves loading some computed data and formatting it into a CMECOutput and CMECMetric bundle.
        # ds = xr.open_dataset(
        #     definition.output_directory / "annual_mean_global_mean_timeseries.nc"
        # )
        #
        # return ExecutionResult.build_from_output_bundle(
        #     definition,
        #     cmec_output_bundle=format_cmec_output_bundle(ds), # Your formatting function
        #     cmec_metric_bundle=format_cmec_metric_bundle(ds), # Your formatting function
        # )
        return ExecutionResult.build_from_output_bundle(
            definition,
            cmec_output_bundle=CMECOutput.create_template(),
            cmec_metric_bundle=CMECMetric.create_template(),
        )
```

If your diagnostic must run in its own Conda environment,
extend [CommandLineDiagnostic][climate_ref_core.diagnostics.CommandLineDiagnostic] instead.


## 4. Register your diagnostics

In your package entry point (e.g. `__init__.py`), register all diagnostics:

```python
import importlib.metadata

from climate_ref_core.providers import DiagnosticProvider
from .example import GlobalMeanTimeseries # Assuming your diagnostic is in example.py

__version__ = importlib.metadata.version("climate-ref-myprovider") # Or your package name

# Initialise the diagnostics manager and register the example diagnostic
provider = DiagnosticProvider("MyProvider", __version__) # Replace "MyProvider" with your provider name
provider.register(GlobalMeanTimeseries())

# If you have more diagnostics, you can register them as well:
# from .another_metric import AnotherMetric
# provider.register(AnotherMetric())

```

The REF will discover providers listed under the `"climate-ref.providers`
entrypoint group in `pyproject.toml`. For example, if your provider module is named
`climate_ref_myprovider` and the provider instance is named `provider`
as in the examples above, you would add the following to your `pyproject.toml`:

```toml
[project.entry-points."climate-ref.providers"]
myprovider = "climate_ref_myprovider:provider"
```

## 5. Provider lifecycle hooks (optional)

If your provider needs to run on HPC systems where compute nodes lack internet access,
you can implement lifecycle hooks to prepare for offline execution.
These hooks are called by `ref providers setup` before any diagnostics are run.

### Available hooks

| Hook | Purpose | When to use |
|------|---------|-------------|
| `setup_environment(config)` | Set up execution environment | Conda env creation, tool installation |
| `fetch_data(config)` | Download required data | Reference datasets, auxiliary files |
| `post_setup(config)` | Post-setup tasks | Tasks requiring both env and data |
| `validate_setup(config)` | Validate setup is complete | Return `True` if ready for offline execution |

All hooks receive the application `Config` object and **must be idempotent** (safe to call multiple times).

### Choosing a provider base class

The REF provides several provider base classes depending on your needs:

| Base Class | Use Case |
|------------|----------|
| `DiagnosticProvider` | Pure Python diagnostics, no special environment needed |
| `CommandLineDiagnosticProvider` | Diagnostics that run via subprocess |
| `CondaDiagnosticProvider` | Diagnostics requiring an isolated conda environment |

`CondaDiagnosticProvider` automatically implements `setup_environment()` to create conda environments.

### Example: Provider with data fetching

If your diagnostics need reference data, override `fetch_data()`:

```python
from climate_ref_core.providers import DiagnosticProvider

class MyProvider(DiagnosticProvider):
    def __init__(self):
        super().__init__("MyProvider", __version__)
        self._registry = None  # DatasetRegistry for reference data

    def fetch_data(self, config):
        """Download reference datasets for offline execution."""
        if self._registry is None:
            return

        # Fetch all registered files to the pooch cache
        self._registry.fetch_all_files()

    def validate_setup(self, config):
        """Check all required data is available."""
        if self._registry is None:
            return True
        # Check that all files are cached
        return all(
            self._registry.is_cached(name)
            for name in self._registry.list_files()
        )
```

### Example: Conda-based provider

For providers using conda environments, extend `CondaDiagnosticProvider`:

```python
from climate_ref_core.providers import CondaDiagnosticProvider

class MyCondaProvider(CondaDiagnosticProvider):
    def __init__(self):
        super().__init__("MyCondaProvider", __version__)

    def fetch_data(self, config):
        """Fetch data after conda env is ready."""
        # Called after setup_environment() creates the conda env
        # Download any reference data here
        pass

    def post_setup(self, config):
        """Run any post-setup tasks."""
        # Called after both env and data are ready
        pass
```

The conda environment is defined by a `conda-lock.yml` file in your package's `requirements/` directory.
See the [ESMValTool provider](https://github.com/Climate-REF/climate-ref/tree/main/packages/climate-ref-esmvaltool/src/climate_ref_esmvaltool/requirements) for an example.

### Running setup

Users run provider setup with:

```bash
# Setup all providers
ref providers setup

# Setup specific provider
ref providers setup --provider myprovider

# Validate setup
ref providers setup --validate-only
```

## 6. Write basic tests

Add unit tests under `packages/climate-ref-myprovider/tests/` to verify the data requirements and execution logic of your diagnostics.

An example test to check that the data requirements are correct might look like this:

```python

import pandas as pd
from climate_ref_myprovider import GlobalMeanTimeseries
from climate_ref_myprovider import provider as myprovider_provider

from climate_ref.solver import solve_executions
from climate_ref_core.datasets import SourceDatasetType


def test_expected_executions():
    diagnostic = GlobalMeanTimeseries()
    data_catalog = {
        SourceDatasetType.CMIP6: pd.DataFrame(
            [
                ["ts", "ACCESS-ESM1-5", "historical", "r1i1p1f1", "mon", "gn"],
                ["ts", "ACCESS-ESM1-5", "ssp119", "r1i1p1f1", "mon", "gn"],
                ["ts", "ACCESS-ESM1-5", "historical", "r2i1p1f1", "mon", "gn"],
                ["pr", "ACCESS-ESM1-5", "historical", "r1i1p1f1", "mon", "gn"],
            ],
            columns=("variable_id", "source_id", "experiment_id", "member_id", "frequency", "grid_label"),
        ),
    }
    executions = list(solve_executions(data_catalog, diagnostic, provider=myprovider_provider))
    assert len(executions) == 3

    # ts
    assert executions[0].datasets[SourceDatasetType.CMIP6].selector == (
        ("experiment_id", "historical"),
        ("member_id", "r1i1p1f1"),
        ("source_id", "ACCESS-ESM1-5"),
        ("variable_id", "ts"),
    )
```

We also recommend writing an integration test that runs the diagnostic end-to-end, and saves the output to a known location.
These results are then checked into the repository to ensure that the diagnostic produces consistent results across runs.

This involves two tests:
one for the diagnostic execution and another checking that result from building the execution result
from the regression output.
The first test is marked as `slow` to indicate that it may take longer to run and is only run if the `--slow`
argument is passed to pytest.
The regression output is regenerated if `--force-regen` is passed to pytest,


```python
import pytest
from climate_ref_myprovider import provider as myprovider_provider

from climate_ref_core.diagnostics import Diagnostic

diagnostics = [pytest.param(diagnostic, id=diagnostic.slug) for diagnostic in myprovider_provider.diagnostics()]


@pytest.mark.slow
@pytest.mark.parametrize("diagnostic", diagnostics)
def test_diagnostics(diagnostic: Diagnostic, diagnostic_validation):
    validator = diagnostic_validation(diagnostic)

    definition = validator.get_definition()
    validator.execute(definition)


@pytest.mark.parametrize("diagnostic", diagnostics)
def test_build_results(diagnostic: Diagnostic, diagnostic_validation):
    validator = diagnostic_validation(diagnostic)

    definition = validator.get_regression_definition()
    validator.validate(definition)
    validator.execution_regression.check(definition.key, definition.output_directory)
```

These tests can be run using:
```bash
pytest  --slow -k "[global-mean-timeseries]" --force-regen
```

The `global-mean-timeseries` is the slug (or the subset of the slug) of the diagnostic you want to test.

## 7. Enable your provider in configuration

Edit your `ref.toml` configuration file to include your new provider:

```toml
[[diagnostic_providers]]
provider = "climate_ref_myprovider:provider"
```

Next time you run a `ref` command you should see your provider being added to the database.

## 8. Update Controlled Vocabulary (optional)

If your metrics use new facets in its metric output (e.g. custom experiment IDs or grid labels),
extend the controlled vocabulary in `climate-ref-core`:

- Copy the default CV (located in `packages/climate-ref-core/src/climate_ref_core/pycmec/cv_cmip7_aft.yaml` or on [GitHub](https://github.com/Climate-REF/climate-ref/blob/main/packages/climate-ref-core/src/climate_ref_core/pycmec/cv_cmip7_aft.yaml).
- Modify it to include your new facets or values.
- Update your [configuration][paths_dimensions_cv] to point to your custom CV file:

```toml
[paths]
dimensions_cv = "/path/to/my/custom/cv_custom.yaml"
```

---

Once complete, run:

```bash
ref solve --provider myprovider
```

and inspect results with `ref executions list-group` and `ref executions inspect <group_id>`.

---

Contributions welcome! Please submit PRs to improve this guide and examples.
