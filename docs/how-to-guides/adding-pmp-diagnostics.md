# Adding a PMP diagnostic

A PMP diagnostic in the REF runs a PCMDI Metrics Package (PMP) driver script
in the provider's conda environment,
then converts the CMEC JSON file PMP writes into a REF result.

Read [Adding a diagnostic](adding_custom_diagnostics.md) first.
This page covers only what is specific to PMP.
Paths are relative to `packages/climate-ref-pmp/src/climate_ref_pmp/`.

## The diagnostic class

PMP diagnostics subclass [CommandLineDiagnostic][climate_ref_core.diagnostics.CommandLineDiagnostic].
`build_cmd` returns the command, and the provider runs it in the conda environment.
`ExtratropicalModesOfVariability` in `diagnostics/variability_modes.py` is the simplest:
one command and one CMEC file.
In outline:

```python
class ExtratropicalModesOfVariability(CommandLineDiagnostic):
    reconstruction_inputs = PMP_RECONSTRUCTION_INPUTS
    facets = (
        "kind", "mip_id", "source_id", "member_id", "experiment_id",
        "reference_source_id", "mode", "season", "method", "statistic",
    )

    def build_cmd(self, definition: ExecutionDefinition) -> Iterable[str]:
        model_source_type = get_model_source_type(definition)
        ...
        params = {
            "variability_mode": self.mode_id,
            "modpath": modpath,
            "mip": model_source_type.value,
            "results_dir": str(definition.output_directory),
            "cmec": None,
            "no_provenance": None,
        }
        return build_pmp_command(
            driver_file="variability_modes_driver.py",
            parameter_file=self.parameter_file,
            **params,
        )

    def build_execution_result(self, definition: ExecutionDefinition) -> ExecutionResult:
        results_files = list(definition.output_directory.glob(f"*_{mip}_*_cmec.json"))
        if len(results_files) != 1:
            return ExecutionResult.build_from_failure(definition)

        png_files = [definition.as_relative_path(f) for f in sorted(definition.output_directory.glob("*.png"))]
        data_files = [definition.as_relative_path(f) for f in sorted(definition.output_directory.glob("*.nc"))]
        cmec_output, cmec_metric = process_json_result(results_files[0], png_files, data_files)

        cmec_metric = cmec_metric.remove_dimensions(["model", "realization", "reference"]).prepend_dimensions(
            {"kind": "model", "mip_id": ..., "source_id": ..., "member_id": ..., ...}
        )
        return ExecutionResult.build_from_output_bundle(
            definition, cmec_output_bundle=cmec_output, cmec_metric_bundle=cmec_metric
        )
```

Register the instance in `__init__.py` with `provider.register(...)`.
One class can back several diagnostics with different constructor arguments,
as the modes of variability and ENSO diagnostics do.

`AnnualCycle` needs two commands, one to compute climatologies and one to compute metrics.
It defines its own `build_cmds` and overrides `execute` to run them in turn.

## Driver scripts and parameter files

`build_pmp_command(driver_file, parameter_file, **kwargs)` in `pmp_driver.py` builds the command.

- `driver_file` is usually a script that `pcmdi_metrics` installs in the conda environment,
  such as `variability_modes_driver.py` or `mean_climate_driver.py`.
- `parameter_file` is a file in `params/`.
- Each keyword becomes a flag.
  A list becomes `--key v1 v2`, a value becomes `--key value`,
  and `None` or `""` becomes a bare `--key`.

A driver written for the REF goes in `drivers/`, as `enso_driver.py` does.
It runs with the conda environment's `python`,
so it can only import packages from `requirements/environment.yml`.
Find it with `_get_resource("climate_ref_pmp.drivers", "my_driver.py", use_resources=True)`
and run it as `["python", driver_file, ...]`.

## Results

`process_json_result(json_filename, png_files, data_files)` reads PMP's `*_cmec.json`
and returns the output bundle and the metric bundle.
The usual steps after that are:

- Drop PMP's own `model`, `realization` and `reference` dimensions with `remove_dimensions`.
- Prepend the REF dimensions with `prepend_dimensions`.
  These must match `facets`.
  Every value gets `kind="model"`, because PMP's scalars score the model.
- Sort the globs, so `output.json` lists files in the same order on every machine.

Set `reconstruction_inputs = PMP_RECONSTRUCTION_INPUTS`.
This keeps the `*_cmec.json` files in the stored baseline,
so a replay can rebuild the result from them.

`AnnualCycle` also turns PMP's monthly values into series.
It moves the `CalendarMonths` values out of the metric bundle
into `SeriesMetricValue`s indexed by `month_number`.

## Reference data

Most PMP diagnostics compare against observations from obs4MIPs, with obs4REF as a fallback.
The data is read from `definition.datasets[SourceDatasetType.obs4MIPs]` either way.

The annual cycle uses PMP's own climatologies instead.
These are listed in `dataset_registry/pmp_climatology.txt`,
fetched and ingested by `ref providers setup`,
and requested with the `PMPClimatology` source type:

```python
DataRequirement(
    source_type=SourceDatasetType.PMPClimatology,
    filters=(FacetFilter({"source_id": ("ERA-5",), "variable_id": ("psl",)}),),
    group_by=("variable_id", "source_id"),
)
```

Each line of the registry file is a file path and its md5 checksum.
The path must follow
`PMP_obs4MIPsClims/{variable_id}/{grid_label}/{version}/{variable_id}_mon_{source_id}_{institution}_{grid_label}_{time_range}_AC_{version}_{resolution}.nc`.
Test cases read the facets from the path,
but ingest reads them from the file's global attributes,
so the two must agree.
Adding a climatology means hosting the file on the REF data server and adding its line.
Ask a maintainer to upload it.
Test cases fetch climatologies with `RegistryRequest(registry_name="pmp-climatology", ...)`.

## Conda environment

`requirements/environment.yml` pins `pcmdi_metrics` and the other conda packages.
After changing it, regenerate `requirements/conda-lock.yml`
(see [Conda-based providers](adding_custom_diagnostics.md#conda-based-providers)).

The provider sets some environment variables for every command it runs:

- The thread limits (`OMP_NUM_THREADS` and others) default to `2`, unless any of them is already set.
- `FI_PROVIDER` defaults to `tcp`, to work around an MPICH error on macOS.
- `DASK_SCHEDULER` defaults to `synchronous`, to avoid an xarray deadlock.
  A driver should not expect Dask to run in parallel.
- `PCMDI_CONDA_EXE` points at the conda executable the provider uses.

## Tests

On top of a test case and its regression baseline,
`tests/unit/test_solve_regression.py` compares the solver's executions
with a fixture per diagnostic in `tests/unit/test_solve_regression/`.
Generate it for a new diagnostic, or after changing a data requirement:

```bash
uv run pytest packages/climate-ref-pmp/tests/unit/test_solve_regression.py --force-regen
```

Check that the new fixture is not empty.
These tests solve against the parquet catalogs in `tests/test-data/esgf-catalog/`,
and a diagnostic whose data is not in them gets an empty fixture that still passes.
Regenerate the catalogs with `scripts/generate_esgf_catalog.py` if your variables are missing.
