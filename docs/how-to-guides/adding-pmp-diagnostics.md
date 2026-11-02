# Adding a PMP diagnostic

A PMP diagnostic runs a PCMDI Metrics Package (PMP) script in the provider's conda environment,
then reads the JSON file it writes.
Read [Adding a diagnostic](adding_custom_diagnostics.md) first.
Paths below are in `packages/climate-ref-pmp/src/climate_ref_pmp/`.

## The diagnostic class

PMP diagnostics subclass [CommandLineDiagnostic][climate_ref_core.diagnostics.CommandLineDiagnostic].
`ExtratropicalModesOfVariability` in `diagnostics/variability_modes.py` is the simplest one to copy.

- `build_cmd` returns the command, built with `build_pmp_command(driver_file, parameter_file, **kwargs)`.
  Each keyword becomes a `--key value` flag.
- `driver_file` is usually a script that PMP installs, such as `variability_modes_driver.py`.
  A script written for the REF goes in `drivers/`.
- `parameter_file` is a file in `params/`.

Register the diagnostic in `__init__.py` with `provider.register(...)`.

## Results

`process_json_result` turns PMP's `*_cmec.json` file into the REF's output and metric bundles.
Then:

- Remove PMP's own `model`, `realization` and `reference` dimensions with `remove_dimensions`.
- Add the REF's dimensions with `prepend_dimensions`, matching `facets`.
- Set `reconstruction_inputs = PMP_RECONSTRUCTION_INPUTS`, so the JSON file is kept for replays.

## Reference data

Most PMP diagnostics use obs4MIPs data with an obs4REF fallback.
The annual cycle uses PMP's own climatologies instead.
They are listed in `dataset_registry/pmp_climatology.txt`
and requested with `SourceDatasetType.PMPClimatology`.
Ask a maintainer to upload a new climatology.

## Conda environment

`requirements/environment.yml` pins PMP.
After changing it, regenerate the lock file
(see [Setup and conda](adding_custom_diagnostics.md#6-setup-and-conda)).

The provider limits each command to 2 threads and runs Dask synchronously, one task at a time.

## Tests

Besides the test case,
the solver's executions are compared with a fixture in `tests/unit/test_solve_regression/`.
Create or update it with:

```bash
uv run pytest packages/climate-ref-pmp/tests/unit/test_solve_regression.py --force-regen
```

Check the new fixture is not empty.
An empty fixture means your data is not in `tests/test-data/esgf-catalog/`.
Add it with `scripts/generate_esgf_catalog.py`.
