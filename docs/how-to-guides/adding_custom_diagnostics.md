# Adding a diagnostic

A diagnostic is one calculation the REF can run, such as a global mean time series.
Diagnostics live in provider packages.
This page builds a new provider from
[climate-ref-example](https://github.com/Climate-REF/climate-ref/tree/main/packages/climate-ref-example).
Read its code alongside this page.

Adding to an existing provider?
Skim this page, then read the guide for that provider:
[ESMValTool](adding-esmvaltool-diagnostics.md), [PMP](adding-pmp-diagnostics.md)
or [ILAMB](adding-ilamb-diagnostics.md).

## How a diagnostic runs

1. `ref solve` matches the datasets in the database against each diagnostic's `data_requirements`.
   Each matching group of datasets becomes one execution.
2. `execute()` runs the calculation and writes files to the output directory.
3. `build_execution_result()` reads those files and describes the results.
   The REF stores them.

Step 3 can be re-run later from the stored files, so it must only read files the results list.

## 1. Copy the example

```bash
cp -r packages/climate-ref-example packages/climate-ref-myprovider
cd packages/climate-ref-myprovider
mv src/climate_ref_example src/climate_ref_myprovider
rm -r tests/test-data
grep -rl 'climate.ref.example' . | xargs perl -pi -e 's/climate([_-])ref\1example/climate$1ref$1myprovider/g'
```

Then edit by hand:

- `pyproject.toml`: the description, authors, dependencies and the entry point name.
- `src/climate_ref_myprovider/__init__.py`: the provider name. The slug is made from it.
- `tests/unit/test_provider.py`: the expected name, slug and number of diagnostics.
- `README.md`.

In the root `pyproject.toml`, add `climate-ref-myprovider` to `dependencies`
and to `[tool.uv.sources]` with `{ workspace = true }`.
Then run `uv sync`.

## 2. Write the diagnostic

Subclass [Diagnostic][climate_ref_core.diagnostics.Diagnostic].
This is a cut-down `GlobalMeanTimeseries` from
[`example.py`](https://github.com/Climate-REF/climate-ref/blob/main/packages/climate-ref-example/src/climate_ref_example/example.py):

```python
class GlobalMeanTimeseries(Diagnostic):
    name = "Global Mean Timeseries"
    slug = "global-mean-timeseries"
    facets = ("source_id", "variable_id", "experiment_id", "variant_label", "region", "metric", "statistic")

    data_requirements = (
        DataRequirement(
            source_type=SourceDatasetType.CMIP6,
            filters=(FacetFilter(facets={"variable_id": ("tas", "rsut")}),),
            group_by=("source_id", "variable_id", "experiment_id", "variant_label"),
            constraints=(AddSupplementaryDataset.from_defaults("areacella", SourceDatasetType.CMIP6),),
        ),
    )

    def execute(self, definition: ExecutionDefinition) -> None:
        files = definition.datasets[SourceDatasetType.CMIP6].path.to_list()
        result = calculate_annual_mean_timeseries(input_files=files)
        result.to_netcdf(definition.output_directory / "annual_mean_global_mean_timeseries.nc")

    def build_execution_result(self, definition: ExecutionDefinition) -> ExecutionResult:
        ds = xr.open_dataset(definition.output_directory / "annual_mean_global_mean_timeseries.nc")
        selectors = definition.datasets[SourceDatasetType.CMIP6].selector_dict()
        metric_bundle = CMECMetric(**format_cmec_metric_bundle(ds)).prepend_dimensions(selectors)
        return ExecutionResult.build_from_output_bundle(
            definition,
            cmec_output_bundle=format_cmec_output_bundle(ds),
            cmec_metric_bundle=metric_bundle,
        )
```

- `slug` must be unique within the provider.
- `version` starts at `1`. Bump it when the results change.
- A diagnostic must not use the network when it runs,
  because the REF often runs on compute nodes with no internet.

## 3. Choose the input data

Each [DataRequirement][climate_ref_core.diagnostics.DataRequirement]
picks datasets with `filters`, splits them into executions with `group_by`,
and checks each group with `constraints`.
The built-in constraints are in [climate_ref_core.constraints][climate_ref_core.constraints].
The most used are `AddSupplementaryDataset`, `RequireContiguousTimerange`,
`SelectFirstMember` and `SelectSingleGrid`.

To compare against observations, add an obs4MIPs requirement with obs4REF as a fallback:

```python
DataRequirement(
    source_type=SourceDatasetType.obs4MIPs,
    fallback_source_types=(SourceDatasetType.obs4REF,),
    filters=(FacetFilter(facets={"source_id": "HadISST-1-1", "variable_id": "ts"}),),
    group_by=("source_id", "variable_id"),
)
```

To accept either CMIP6 or CMIP7, make `data_requirements` a tuple of tuples.
Each inner tuple is one alternative.
The example does this.

The [dataset selection guide](dataset-selection.py) has worked examples.

## 4. Describe the results

`ExecutionResult.build_from_output_bundle` takes three things:

- `cmec_output_bundle`: the files to keep, such as data and plots.
  Start from `CMECOutput.create_template()`.
  Only listed files are kept, and a listed file that is missing fails the execution.
- `cmec_metric_bundle`: scalar values, nested by the names in `DIMENSIONS.json_structure`.
  `prepend_dimensions` adds the dataset facets on the outside.
  The result must use exactly the names in `facets`.
- `series`: one-dimensional values, such as time series, as
  [SeriesMetricValue][climate_ref_core.metric_values.typing.SeriesMetricValue]s.

Set `kind="reference"` on values computed from observations.
[`surface_temperature.py`](https://github.com/Climate-REF/climate-ref/blob/main/packages/climate-ref-example/src/climate_ref_example/surface_temperature.py)
shows model, reference and bias series.

Every dimension must be in the controlled vocabulary,
[`cv_cmip7_aft.yaml`](https://github.com/Climate-REF/climate-ref/blob/main/packages/climate-ref-core/src/climate_ref_core/pycmec/cv_cmip7_aft.yaml).
An unknown dimension is dropped with only a warning, so check the log.
Reuse existing dimensions where you can, and add new ones to that file in the same pull request.

## 5. Register the provider

Create the provider in `__init__.py` and register each diagnostic on it:

```python
provider = DiagnosticProvider("MyProvider", __version__)
provider.register(GlobalMeanTimeseries())
```

The REF finds it through the entry point in `pyproject.toml`:

```toml
[project.entry-points."climate-ref.providers"]
myprovider = "climate_ref_myprovider:provider"
```

Check it shows up:

```bash
ref providers list
ref providers show --provider myprovider
```

If `ref.toml` lists `[[diagnostic_providers]]`, or `REF_DIAGNOSTIC_PROVIDERS` is set,
only those providers are enabled, so add yours there.

## 6. Setup and conda

`ref providers setup` runs once, with internet, to get a provider ready to run offline.
Override these [DiagnosticProvider][climate_ref_core.providers.DiagnosticProvider] methods if you need to:

- `fetch_data` downloads reference data that is not in obs4MIPs or obs4REF.
- `data_to_ingest` says which fetched data to load into the database.
- `validate_setup` checks everything is in place.

To run a tool with its own dependencies, as ESMValTool and PMP do,
subclass [CondaDiagnosticProvider][climate_ref_core.providers.CondaDiagnosticProvider]
and add `requirements/environment.yml`.
The diagnostics subclass [CommandLineDiagnostic][climate_ref_core.diagnostics.CommandLineDiagnostic]
and return their command from `build_cmd`.
After changing `environment.yml`, regenerate the lock file:

```bash
uvx conda-lock -p linux-64 -p osx-64 -p osx-arm64 \
    -f packages/climate-ref-myprovider/src/climate_ref_myprovider/requirements/environment.yml
mv conda-lock.yml packages/climate-ref-myprovider/src/climate_ref_myprovider/requirements/conda-lock.yml
```

## 7. Test it

- Unit tests: copy
  [`tests/unit/test_metrics.py`](https://github.com/Climate-REF/climate-ref/blob/main/packages/climate-ref-example/tests/unit/test_metrics.py).
  Run `make fetch-test-data` first.
- Test cases: add a `test_data_spec` with a small set of real data.
  CI checks its results on every pull request.
  [Testing diagnostics](testing-diagnostics.md) explains how.

## 8. Run it

Load some model data, then solve:

```bash
ref providers setup --provider myprovider
ref datasets ingest --source-type cmip6 /path/to/cmip6
ref solve --provider myprovider --one-per-diagnostic
ref executions list-groups --provider myprovider
```

## Before you open a pull request

For a new provider in this repository, add it to:

- `[tool.importlinter]` in the root `pyproject.toml`, in `root_packages` and the provider layer.
- The `Makefile`, as a `test-diagnostic-myprovider` target listed under `test-diagnostics`.
- The `provider` matrices in `.github/workflows/ci.yaml`, `ci-integration.yaml` and `provider-compat.yml`.
- `PROVIDERS` in `docs/gen_diagnostic_stubs.py`, the `mkdocs.yml` nav,
  and a `write_module_page` call in `docs/gen_doc_stubs.py`.

For any change, add a changelog fragment (see `changelog/README.md`),
add or update a test case, and bump `version` if the results change.
