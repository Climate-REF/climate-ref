# Adding a diagnostic

The REF does no science itself.
Every calculation lives in a diagnostic,
and diagnostics are shipped in provider packages.
This page builds a new provider from the
[climate-ref-example](https://github.com/Climate-REF/climate-ref/tree/main/packages/climate-ref-example) package
and takes it as far as a pull request.

If you are adding a diagnostic to an existing provider,
read this page for the concepts, then the guide for that provider,
because each one wraps its own tool:

- [ESMValTool](adding-esmvaltool-diagnostics.md)
- [PMP](adding-pmp-diagnostics.md)
- [ILAMB](adding-ilamb-diagnostics.md)

## How a diagnostic runs

1. `ref solve` reads the datasets in the database
   and matches them against each diagnostic's `data_requirements`.
2. Each matching group of datasets becomes an execution.
   The solver hands it to the diagnostic as an
   [ExecutionDefinition][climate_ref_core.diagnostics.ExecutionDefinition].
   This holds the input datasets and an output directory.
3. `execute()` does the calculation and writes files into that directory.
4. `build_execution_result()` reads those files back
   and describes them as an output bundle, a metric bundle and a list of series.
   The REF stores these in its database.

Step 4 can be re-run later from the stored files alone.
Only the bundles and the files the output bundle lists are stored,
so step 4 must work from those.
List any other file it reads in `reconstruction_inputs`.

## 1. Create the provider package

Copy the example provider and rename it.
Replace `myprovider` with your provider's name.

```bash
cp -r packages/climate-ref-example packages/climate-ref-myprovider
cd packages/climate-ref-myprovider
mv src/climate_ref_example src/climate_ref_myprovider
rm -r tests/test-data
grep -rl 'climate.ref.example' . | xargs perl -pi -e 's/climate([_-])ref\1example/climate$1ref$1myprovider/g'
```

The test data holds the example's regression baselines,
so it is removed rather than copied.
The last command renames both `climate_ref_example` and `climate-ref-example`.

Then edit by hand:

- `pyproject.toml`: the `description`, `authors`, `dependencies`
  and the entry point name (`example = ...` becomes `myprovider = ...`).
- `src/climate_ref_myprovider/__init__.py`: the provider name in `DiagnosticProvider("Example", __version__)`.
  The provider slug is derived from this name.
- `tests/unit/test_provider.py`: the expected name, slug and number of diagnostics.
- `README.md`.

A provider in this repository also needs adding to the workspace.
In the root `pyproject.toml`, add `climate-ref-myprovider` to `dependencies`
and to `[tool.uv.sources]` with `{ workspace = true }`.
Then run `uv sync`.

A provider can also live in its own repository.
It then only needs to be installed in the same environment as `climate-ref`.

### Dependencies

There are two ways to provide the software a diagnostic needs.

- Python dependencies in `pyproject.toml` are installed alongside the REF.
  Use this for small, pure Python diagnostics like the example.
- A conda environment is created by `ref providers setup` and kept separate from the REF.
  Use this when the diagnostic wraps a tool with its own dependency stack,
  which is how ESMValTool and PMP work.
  See [Conda-based providers](#conda-based-providers).

## 2. Write the diagnostic

A diagnostic is a subclass of [Diagnostic][climate_ref_core.diagnostics.Diagnostic].
This is the CMIP6 half of `GlobalMeanTimeseries` from
[`example.py`](https://github.com/Climate-REF/climate-ref/blob/main/packages/climate-ref-example/src/climate_ref_example/example.py).

```python
import xarray as xr

from climate_ref_core.constraints import AddSupplementaryDataset, RequireContiguousTimerange
from climate_ref_core.datasets import FacetFilter, SourceDatasetType
from climate_ref_core.diagnostics import DataRequirement, Diagnostic, ExecutionDefinition, ExecutionResult
from climate_ref_core.pycmec.metric import CMECMetric


class GlobalMeanTimeseries(Diagnostic):
    name = "Global Mean Timeseries"
    slug = "global-mean-timeseries"

    data_requirements = (
        DataRequirement(
            source_type=SourceDatasetType.CMIP6,
            filters=(FacetFilter(facets={"variable_id": ("tas", "rsut")}),),
            group_by=("source_id", "variable_id", "experiment_id", "variant_label"),
            constraints=(
                RequireContiguousTimerange(group_by=("instance_id",)),
                AddSupplementaryDataset.from_defaults("areacella", SourceDatasetType.CMIP6),
            ),
        ),
    )
    facets = ("source_id", "variable_id", "experiment_id", "variant_label", "region", "metric", "statistic")

    def execute(self, definition: ExecutionDefinition) -> None:
        input_datasets = definition.datasets[SourceDatasetType.CMIP6]
        result = calculate_annual_mean_timeseries(input_files=input_datasets.path.to_list())
        result.to_netcdf(definition.output_directory / "annual_mean_global_mean_timeseries.nc")

    def build_execution_result(self, definition: ExecutionDefinition) -> ExecutionResult:
        ds = xr.open_dataset(definition.output_directory / "annual_mean_global_mean_timeseries.nc")
        selectors = definition.datasets[SourceDatasetType.CMIP6].selector_dict()

        metric_bundle = CMECMetric(**format_cmec_metric_bundle(ds)).prepend_dimensions(
            {
                "source_id": selectors["source_id"],
                "variable_id": selectors["variable_id"],
                "experiment_id": selectors["experiment_id"],
                "variant_label": selectors["variant_label"],
            }
        )
        return ExecutionResult.build_from_output_bundle(
            definition,
            cmec_output_bundle=format_cmec_output_bundle(ds),
            cmec_metric_bundle=metric_bundle,
        )
```

`calculate_annual_mean_timeseries`, `format_cmec_output_bundle` and `format_cmec_metric_bundle`
are plain functions in the same file.

The class attributes are:

| Attribute | Meaning |
| --- | --- |
| `name` | Human readable name. |
| `slug` | Identifier, unique within the provider. Used on the command line and in result paths. |
| `version` | Integer, defaults to `1`. Bump it when the results change, so existing executions are recalculated. |
| `data_requirements` | Which datasets the diagnostic needs and how they are grouped into executions. See [step 3](#3-select-the-input-data). |
| `facets` | The dimensions of the metric values. See [Facets and the controlled vocabulary](#facets-and-the-controlled-vocabulary). |
| `test_data_spec` | The test cases used to check the diagnostic. See [step 7](#7-test-it). |

A diagnostic must not make network calls in `execute()` or `build_execution_result()`,
because the REF often runs on HPC compute nodes without internet access.
Data comes from `definition.datasets`
or is fetched in advance by the provider's [setup hooks](#6-prepare-for-offline-execution).

## 3. Select the input data

Each [DataRequirement][climate_ref_core.diagnostics.DataRequirement] names a source type,
filters the datasets of that type,
groups them into executions with `group_by`,
and checks each group with `constraints`.
`definition.datasets[source_type]` then holds the datasets of one group,
and `selector_dict()` returns the `group_by` facet values that define it.

A diagnostic that compares against observations adds a requirement for the reference data.
The obs4REF registry carries datasets that obs4MIPs has not published yet,
so the requirement must declare it as a fallback.
The solver does not fold obs4REF into obs4MIPs on its own.

```python
DataRequirement(
    source_type=SourceDatasetType.obs4MIPs,
    fallback_source_types=(SourceDatasetType.obs4REF,),
    filters=(FacetFilter(facets={"source_id": "HadISST-1-1", "variable_id": "ts"}),),
    group_by=("source_id", "variable_id"),
)
```

The data is delivered under `obs4MIPs` whichever collection it came from.
A dataset ingested as `obs4MIPs` wins a version tie.

`data_requirements` can also hold several tuples of requirements.
Each tuple is an alternative, so the diagnostic runs if any of them is met.
The example uses this to accept either CMIP6 or CMIP7 data.

### Constraints

Constraints run on each group after filtering and grouping.
They drop groups that cannot be used, or add or remove datasets within a group.
These are in [climate_ref_core.constraints][climate_ref_core.constraints]:

| Constraint | Use |
| --- | --- |
| `RequireFacets` | Keep a group only if it has the given facet values, eg. every variable a diagnostic needs. |
| `IgnoreFacets` | Drop datasets with the given facet values. |
| `RequireTimerange` | Keep a group only if it covers a given period. |
| `RequireContiguousTimerange` | Keep a group only if its files have no gaps in time. |
| `RequireOverlappingTimerange` | Keep a group only if its datasets share a period. |
| `AddSupplementaryDataset` | Add a cell measure or ancillary variable, eg. `areacella`. Use `from_defaults`. |
| `AddParentDataset` | Add the parent experiment of each dataset, eg. `piControl` for `abrupt-4xCO2`. |
| `SelectFirstMember` | Keep a single ensemble member per group. |
| `SelectSingleGrid` | Keep a single grid per variable, for groups that would otherwise mix grids. The choice is set by a `GridPreference`, `PreferNativeGrid` by default. |

The [dataset selection guide](dataset-selection.py) covers filters, grouping and writing your own constraint.

## 4. Write the results

`ExecutionResult.build_from_output_bundle` writes three files into the output directory.

| File | Holds | Built from |
| --- | --- | --- |
| `output.json` | The files a user can look at: data, plots and HTML. | `cmec_output_bundle` |
| `diagnostic.json` | Scalar metric values. | `cmec_metric_bundle` |
| `series.json` | One dimensional values, such as a time series. | `series` |

### Output bundle

Start from [CMECOutput.create_template()][climate_ref_core.pycmec.output.CMECOutput]
and register each file under `data`, `plots` or `html`.
Filenames are relative to the output directory.

```python
cmec_output = CMECOutput.create_template()
cmec_output["data"]["annual_mean_timeseries"] = {
    "filename": "annual_mean_global_mean_timeseries.nc",
    "long_name": "Annual Mean Global Mean Timeseries",
    "description": "Area-weighted annual mean global mean timeseries for the input variable.",
}
cmec_output["plots"]["timeseries"] = {
    "filename": "timeseries.png",
    "long_name": "Global mean timeseries",
    "description": "Annual global mean of the input variable.",
}
```

After a successful run the REF copies the three JSON files and every file the output bundle lists
from the scratch directory to the results directory.
Nothing else is kept.
A listed file that does not exist fails the execution.

### Metric bundle

A metric bundle is a nested dictionary of scalar values.
`DIMENSIONS.json_structure` names each level of nesting,
and `RESULTS` holds the values in that order.

```python
cmec_metric = {
    "DIMENSIONS": {
        "json_structure": ["region", "metric", "statistic"],
        "region": {"global": {}},
        "metric": {"tos": {}},
        "statistic": {"rmse": {}, "mean-bias": {}},
    },
    "RESULTS": {
        "global": {"tos": {"rmse": 0.41, "mean-bias": -0.12}},
    },
}
```

`prepend_dimensions` adds the facets of the input data as outer levels.
After the call in step 2 each value is identified by
`source_id`, `variable_id`, `experiment_id`, `variant_label`, `region`, `metric` and `statistic`.
These are exactly the names in `facets`.

### Series

A series is a list of values with an index, such as an annual time series.
Build each one as a [SeriesMetricValue][climate_ref_core.metric_values.typing.SeriesMetricValue]
and pass the list to `build_from_output_bundle` as `series=`.

```python
SeriesMetricValue(
    dimensions={"source_id": "HadISST-1-1", "region": "global", "metric": "tos", "statistic": "mean"},
    values=[18.1, 18.2, 18.2],
    index=[2000, 2001, 2002],
    index_name="year",
    kind="reference",
)
```

Set `kind="reference"` on a series or scalar computed from observations.
Leave the default `kind="model"` on everything else.
Results can be filtered on `kind`.

`GlobalMeanSurfaceTemperatureBias` in
[`surface_temperature.py`](https://github.com/Climate-REF/climate-ref/blob/main/packages/climate-ref-example/src/climate_ref_example/surface_temperature.py)
writes a model series, a reference series and a bias series.

### Facets and the controlled vocabulary

Every dimension of a metric value or a series must be defined in the controlled vocabulary,
[`cv_cmip7_aft.yaml`](https://github.com/Climate-REF/climate-ref/blob/main/packages/climate-ref-core/src/climate_ref_core/pycmec/cv_cmip7_aft.yaml).
Each dimension is a column in the database.
A dimension that is not in the file only logs a warning,
and the value is stored without it,
so check the log when you add a dimension.
Reuse existing dimensions and values where you can,
because the dashboard filters on them across providers.

To add a dimension for a provider in this repository,
add it to `cv_cmip7_aft.yaml` in the same pull request.
A provider outside the repository can point
[`paths.dimensions_cv`][paths_dimensions_cv] at its own copy of the file.

## 5. Register the provider

The provider is an instance of [DiagnosticProvider][climate_ref_core.providers.DiagnosticProvider]
with every diagnostic registered on it.

```python
import importlib.metadata

from climate_ref_core.providers import DiagnosticProvider
from climate_ref_myprovider.example import GlobalMeanTimeseries

__version__ = importlib.metadata.version("climate-ref-myprovider")

provider = DiagnosticProvider("MyProvider", __version__)
provider.register(GlobalMeanTimeseries())
```

The REF finds providers through the `climate-ref.providers` entry point group in `pyproject.toml`.

```toml
[project.entry-points."climate-ref.providers"]
myprovider = "climate_ref_myprovider:provider"
```

Every installed provider is enabled unless `ref.toml` lists `[[diagnostic_providers]]`
or `REF_DIAGNOSTIC_PROVIDERS` is set.
If either is, add yours to it.

```toml
[[diagnostic_providers]]
provider = "climate_ref_myprovider:provider"
```

Check that the REF can see it:

```bash
ref providers list
ref providers show --provider myprovider
```

## 6. Prepare for offline execution

`ref providers setup` runs once, on a machine with internet access,
to get each provider ready to run offline.
A provider overrides these methods as needed.
Each must be safe to run more than once.

| Method | Purpose |
| --- | --- |
| `setup_environment(config)` | Create the software environment. `CondaDiagnosticProvider` creates its conda environment here. |
| `fetch_data(config)` | Download reference data and auxiliary files. |
| `data_to_ingest(config)` | Return the fetched directories that need ingesting into the database, as `DataIngestRequest`s. |
| `validate_setup(config)` | Return `True` when the environment and data are complete. Run at the end of `ref providers setup`, or alone with `--validate-only`. |
| `check_health(config)` | Return a list of problems that would stop a diagnostic launching. Keep it cheap, because `ref solve` runs it every time. |
| `get_data_path()` | Where the fetched data lives. Shown by `ref providers list`. |

Reference data is usually described by a dataset registry,
a list of files and checksums that [pooch](https://www.fatiando.org/pooch/) downloads into a cache.
This is how the PMP provider fetches and checks its climatologies:

```python
from climate_ref_core.dataset_registry import dataset_registry_manager, fetch_all_files, validate_registry_cache
from climate_ref_core.providers import CondaDiagnosticProvider


class PMPDiagnosticProvider(CondaDiagnosticProvider):
    def fetch_data(self, config):
        registry = dataset_registry_manager["pmp-climatology"]
        fetch_all_files(registry, "pmp-climatology", output_dir=None)

    def validate_setup(self, config):
        if not super().validate_setup(config):
            return False
        registry = dataset_registry_manager["pmp-climatology"]
        return not validate_registry_cache(registry, "pmp-climatology")
```

If the reference data is already published through obs4MIPs or obs4REF,
request it with a `DataRequirement` instead and leave these hooks alone.

### Conda-based providers

Subclass [CondaDiagnosticProvider][climate_ref_core.providers.CondaDiagnosticProvider]
and add `requirements/environment.yml` to the provider module.
Generate the lockfile it installs from:

```bash
uvx conda-lock -p linux-64 -p osx-64 -p osx-arm64 \
    -f packages/climate-ref-myprovider/src/climate_ref_myprovider/requirements/environment.yml
mv conda-lock.yml packages/climate-ref-myprovider/src/climate_ref_myprovider/requirements/conda-lock.yml
```

The diagnostics then subclass [CommandLineDiagnostic][climate_ref_core.diagnostics.CommandLineDiagnostic]
and return the command to run from `build_cmd(definition)`.
The provider runs it inside the conda environment.
`build_execution_result` still runs in the REF's own environment,
so it can only read the files the command wrote.

## 7. Test it

### Unit tests

Test the calculation and a single run of the diagnostic.
The `climate_ref` pytest plugin provides fixtures for the sample data
and for building an `ExecutionDefinition`.
The sample data fixtures need the data downloaded first, with `make fetch-test-data`.
This test is from
[`tests/unit/test_metrics.py`](https://github.com/Climate-REF/climate-ref/blob/main/packages/climate-ref-example/tests/unit/test_metrics.py):

```python
import pytest
from climate_ref_example.example import GlobalMeanTimeseries

from climate_ref_core.datasets import DatasetCollection


@pytest.fixture
def metric_dataset(cmip6_data_catalog) -> DatasetCollection:
    selected_dataset = cmip6_data_catalog[
        cmip6_data_catalog["instance_id"].isin(
            {
                "CMIP6.ScenarioMIP.CSIRO.ACCESS-ESM1-5.ssp126.r1i1p1f1.Amon.tas.gn.v20210318",
                "CMIP6.ScenarioMIP.CSIRO.ACCESS-ESM1-5.ssp126.r1i1p1f1.fx.areacella.gn.v20210318",
            }
        )
    ]
    return DatasetCollection(
        selected_dataset,
        "instance_id",
        selector=(
            ("source_id", "CSIRO.ACCESS-ESM1-5"),
            ("variable_id", "tas"),
            ("experiment_id", "ssp126"),
            ("variant_label", "r1i1p1f1"),
        ),
    )


def test_example_metric(metric_dataset, definition_factory):
    diagnostic = GlobalMeanTimeseries()
    definition = definition_factory(diagnostic=diagnostic, cmip6=metric_dataset)
    definition.output_directory.mkdir(parents=True, exist_ok=True)

    result = diagnostic.run(definition)

    assert result.successful
    assert (definition.output_directory / result.metric_bundle_filename).is_file()
```

### Test cases and regression baselines

A test case is a small, named set of real input data, declared in `test_data_spec`.
CI pins each test case to a regression baseline,
so a change to the diagnostic's numbers shows up in review.

```python
from climate_ref_core.esgf import CMIP6Request
from climate_ref_core.testing import TestCase, TestDataSpecification


class GlobalMeanTimeseries(Diagnostic):
    ...

    test_data_spec = TestDataSpecification(
        test_cases=(
            TestCase(
                name="default",
                description="Historical tas from ACCESS-ESM1-5 with cell areas",
                requests=(
                    CMIP6Request(
                        slug="example-tas",
                        facets={
                            "source_id": "ACCESS-ESM1-5",
                            "experiment_id": "historical",
                            "variable_id": "tas",
                            "member_id": "r1i1p1f1",
                            "table_id": "Amon",
                        },
                        time_span=("2000-01", "2014-12"),
                    ),
                    CMIP6Request(
                        slug="example-areacella",
                        facets={
                            "source_id": "ACCESS-ESM1-5",
                            "experiment_id": "historical",
                            "variable_id": "areacella",
                            "member_id": "r1i1p1f1",
                            "table_id": "fx",
                        },
                    ),
                ),
            ),
        ),
    )
```

Fetch the data and run the case:

```bash
ref test-cases fetch --provider myprovider --diagnostic global-mean-timeseries
ref test-cases run --provider myprovider --diagnostic global-mean-timeseries --test-case default
```

`fetch` writes `catalog.yaml` and the first `run` writes `manifest.json` and `regression/`,
all under `tests/test-data/global-mean-timeseries/default/`.
Check the numbers, then follow
[Adding a test case](testing-diagnostics.md#adding-a-test-case) to commit them and get them minted.

`tests/integration/test_diagnostics.py` turns every test case into a pytest test:

```python
from climate_ref_myprovider import provider

from climate_ref.testing import create_no_drift_test

test_run_test_cases = create_no_drift_test(provider)
```

[Testing diagnostics](testing-diagnostics.md) covers the request types, the CLI and debugging.

## 8. Run it

`ref solve` only finds datasets that are in the database.
`ref providers setup` ingests the provider's reference data,
but the model data has to be ingested separately.
See [Ingest datasets](../getting-started/03-ingest.md).

```bash
ref providers setup --provider myprovider
ref datasets ingest --source-type cmip6 /path/to/cmip6
ref solve --provider myprovider --one-per-diagnostic
ref executions list-groups --provider myprovider
ref executions inspect <group_id>
```

`--one-per-diagnostic` runs a single execution of each diagnostic, which is enough for a first check.

## Before you open a pull request

For a provider in this repository:

- Add the package to `[tool.importlinter]` in the root `pyproject.toml`,
  both to `root_packages` and to the provider layer.
  Providers may import `climate_ref_core` but not `climate_ref`.
- Add a `test-diagnostic-myprovider` target to the `Makefile`
  and list it under `test-diagnostics`.
- Add the provider to the `provider` matrix in `.github/workflows/ci.yaml`
  and `.github/workflows/ci-integration.yaml`,
  and to `.github/workflows/provider-compat.yml`.
- Add it to `PROVIDERS` in `docs/gen_diagnostic_stubs.py`
  and to the Diagnostics section of the `mkdocs.yml` nav,
  and add a `write_module_page` call to `docs/gen_doc_stubs.py`.
- Add a changelog fragment (see `changelog/README.md`).

For any change to a diagnostic:

- Bump `version` if the results change.
  [Changing a diagnostic's results](testing-diagnostics.md#changing-a-diagnostics-results)
  explains what happens next.
- Add or update a test case.

Ask in the pull request, or open an [issue](https://github.com/Climate-REF/climate-ref/issues), if you get stuck.
