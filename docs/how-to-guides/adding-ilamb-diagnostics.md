# Adding an ILAMB diagnostic

ILAMB diagnostics are written in YAML, not Python.
Each entry becomes one diagnostic that runs an [ilamb3](https://github.com/rubisco-sfa/ilamb3) analysis.
Read [Adding a diagnostic](adding_custom_diagnostics.md) first.
Paths below are in `packages/climate-ref-ilamb/src/climate_ref_ilamb/`.

## Add an entry

Land diagnostics go in `configure/ilamb.yaml` and ocean ones in `configure/iomb.yaml`.

```yaml
mrro-LORA:
  version: 3
  sources:
    mrro:
      obs_source: obs4ref
      source_id: LORA-1-0
      variable_id: mrro
      grid_label: gn
      version: v20250902
  variable_cmap: Blues
```

- Name the entry `<variable>-<dataset>`.
  The slug is the name in lower case, e.g. `mrro-lora`.
- `sources` names the variable and its reference data.
- `version` defaults to `2`. Bump it when the results change.
- `test_source_id` sets the model for the test cases. It defaults to `CanESM5`.
- Every other key is passed straight to ilamb3. See the ilamb3 documentation.

## Reference data

A `sources` value is either a set of facets or a file path.

- Facets find an obs4MIPs dataset, with obs4REF as a fallback.
  `obs_source` says where the test cases fetch it from: `obs4ref` or `obs4mips`.
- A path such as `ilamb/tas/CRU4.02/tas.nc` is a file in `dataset_registry/ilamb.txt`.
  Use this only for data not in obs4MIPs or obs4REF, and ask a maintainer to upload it.

## Tests

The test cases are made from the entry, so there is nothing to write.
Fetch and run them as in [Adding a test case](testing-diagnostics.md#adding-a-test-case):

```bash
ref test-cases fetch --provider ilamb --diagnostic mrro-lora
ref test-cases run --provider ilamb --diagnostic mrro-lora
```

The solver's executions are also compared with a fixture in `tests/unit/test_solve_regression/`.
Create or update it with:

```bash
uv run pytest packages/climate-ref-ilamb/tests/unit/test_solve_regression.py --force-regen
```

Check the new fixture is not empty.
An empty fixture means your data is not in `tests/test-data/esgf-catalog/`.
Add it with `scripts/generate_esgf_catalog.py`.
