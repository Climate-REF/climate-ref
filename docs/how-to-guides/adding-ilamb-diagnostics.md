# Adding an ILAMB diagnostic

ILAMB diagnostics are written in YAML, not Python.
Each entry in a configuration file becomes one diagnostic,
which runs a single [ilamb3](https://github.com/rubisco-sfa/ilamb3) analysis block in the REF's own Python environment.

Read [Adding a diagnostic](adding_custom_diagnostics.md) first,
for how diagnostics run and how they are tested.
Paths are relative to `packages/climate-ref-ilamb/src/climate_ref_ilamb/`.

## Configuration files

There is one file per collection:

- `configure/ilamb.yaml` holds the land diagnostics.
- `configure/iomb.yaml` holds the ocean diagnostics.

`realm` and the optional `region_masks` at the top of a file apply to every entry in it.
Every other top level key is a diagnostic.

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

The key is the diagnostic's name.
The slug is the key in lower case, with spaces replaced by hyphens.
The text between the first and second `-` becomes the `reference_source_id` of every value,
so name entries `<variable>-<dataset>`.

Each entry becomes an `ILAMBStandard` diagnostic (`standard.py`).
It reads these keys itself:

| Key | Meaning |
| --- | --- |
| `sources` | The variable to analyse and its reference data. See [Reference data](#reference-data). |
| `version` | The diagnostic version. Bump it when the results change. |
| `test_source_id` | The model used for the test cases. Defaults to `CanESM5`. Set it when CanESM5 does not have the variable. |
| `analysis_variable` | The variable to analyse, when it is not the first key of `sources`. |

Every other key is passed to ilamb3 unchanged,
for example `alternate_vars`, `related_vars`, `relationships`, `transforms`, `analyses` and `variable_cmap`.
See the ilamb3 documentation for what they do.
A custom transform must be registered in `ilamb3.transform.ALL_TRANSFORMS`.
`standard.py` registers the REF's own transforms.

## Model data

The data requirements are built from the entry.
The diagnostic asks for monthly `historical`, `esm-hist` or `land-hist` data
for the analysed variable and its `alternate_vars`, `related_vars` and `relationships`,
from CMIP6 or CMIP7.
Each execution covers one model, experiment, member and grid.
Cell areas and land or ocean fractions are added as supplementary datasets.

## Reference data

Each `sources` value is either a set of facets or a registry key.

A set of facets requests an obs4MIPs dataset with an obs4REF fallback, like any other diagnostic.
`obs_source` is not a facet.
It says where the test cases fetch the data from: `obs4ref` or `obs4mips`.
With `obs4mips`, leave out `version`,
because ESGF version facets have no `v` prefix and would not match.

A string is a file in a dataset registry.

```yaml
gpp-WECANN:
  sources:
    gpp:
      obs_source: obs4ref
      source_id: WECANN-1-0
      variable_id: gpp
      grid_label: gn
      version: v20250902
  relationships:
    pr: obs4REF/NOAA-NCEI/GPCP-2-3/mon/pr/gn/v20210727/pr_mon_GPCP-2-3_PCMDI_gn_197901-201907.nc
    tas: ilamb/tas/CRU4.02/tas.nc
```

Keys starting with `obs4REF/` come from the obs4REF registry.
Keys starting with `ilamb/` come from `dataset_registry/ilamb.txt`,
which holds the few datasets not yet published to obs4MIPs or obs4REF.
These files are fetched by `ref providers setup` and read directly, not ingested.

To add a reference dataset, publish it to obs4REF and use a set of facets if you can.
Otherwise it has to be uploaded to the REF data server and added to `ilamb.txt`
as a line of `<key> sha1:<checksum>`.
Ask a maintainer to upload it.

## Regions

Land entries are evaluated over the global and tropical regions from the `ilamb-regions` masks,
which `region_masks` at the top of `ilamb.yaml` selects.
The ocean collection sets no masks.

## Tests

The test cases are generated from the entry, so there is nothing to write.
Each diagnostic gets a `cmip6` case and a `cmip7` case
with `historical` data from `test_source_id` and the reference data named in `sources`.
Fetch, run and commit them as in [Adding a test case](testing-diagnostics.md#adding-a-test-case):

```bash
ref test-cases fetch --provider ilamb --diagnostic mrro-lora
ref test-cases run --provider ilamb --diagnostic mrro-lora
```

`tests/unit/test_solve_regression.py` also compares the solver's executions
with a fixture per diagnostic in `tests/unit/test_solve_regression/`.
Generate it for a new entry, or after changing one:

```bash
uv run pytest packages/climate-ref-ilamb/tests/unit/test_solve_regression.py --force-regen
```

Check that the new fixture is not empty.
These tests solve against the parquet catalogs in `tests/test-data/esgf-catalog/`,
and a diagnostic whose data is not in them gets an empty fixture that still passes.
Regenerate the catalogs with `scripts/generate_esgf_catalog.py` if your variables are missing.
