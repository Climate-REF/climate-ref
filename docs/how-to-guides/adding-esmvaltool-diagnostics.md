# Adding an ESMValTool diagnostic

An ESMValTool diagnostic in the REF wraps an ESMValTool recipe.
The REF fills the recipe with the datasets of one execution,
runs `esmvaltool` in the provider's conda environment,
and turns the provenance ESMValTool writes into a REF result.

Read [Adding a diagnostic](adding_custom_diagnostics.md) first.
This page covers only what is specific to ESMValTool.
Paths are relative to `packages/climate-ref-esmvaltool/src/climate_ref_esmvaltool/`.

## The diagnostic class

Subclass `ESMValToolDiagnostic` from `diagnostics/base.py`.
On top of the usual attributes it needs `base_recipe`, the recipe to start from,
and an `update_recipe` method.
This is `GlobalMeanTimeseries` from `diagnostics/example.py`, without its CMIP7 requirement and test cases:

```python
class GlobalMeanTimeseries(ESMValToolDiagnostic):
    name = "Global Mean Timeseries"
    slug = "global-mean-timeseries"
    base_recipe = "examples/recipe_python.yml"

    data_requirements = (
        (
            DataRequirement(
                source_type=SourceDatasetType.CMIP6,
                filters=(FacetFilter(facets={"variable_id": ("tas",)}),),
                group_by=("source_id", "experiment_id", "member_id", "table_id", "variable_id", "grid_label"),
                constraints=(
                    RequireContiguousTimerange(group_by=("instance_id",)),
                    AddSupplementaryDataset.from_defaults("areacella", SourceDatasetType.CMIP6),
                ),
            ),
        ),
    )
    facets = ()
    series = (
        SeriesDefinition(
            file_pattern="work/timeseries/script1/*.nc",
            dimensions={"statistic": "annual mean", "variable_id": "tas", "region": "global"},
            values_name="tas",
            index_name="time",
            attributes=[],
        ),
    )
    files = (
        FileDefinition(
            file_pattern="plots/timeseries/script1/png/*.png",
            dimensions={"statistic": "annual mean", "variable_id": "tas", "region": "global"},
        ),
    )

    @staticmethod
    def update_recipe(recipe: Recipe, input_files: dict[SourceDatasetType, pandas.DataFrame]) -> None:
        recipe["datasets"].clear()
        recipe["diagnostics"].pop("map")
        variables = recipe["diagnostics"]["timeseries"]["variables"]
        variables.clear()

        recipe_variables = dataframe_to_recipe(input_files[get_cmip_source_type(input_files)])
        recipe_variables = {k: v for k, v in recipe_variables.items() if k != "areacella"}
        for variable in recipe_variables.values():
            variable["preprocessor"] = "annual_mean_global"
            variable["caption"] = "Annual global mean {long_name} according to {dataset}."
        variables.update(recipe_variables)
```

Import the new class in `diagnostics/__init__.py` and add it to `__all__`.
Every class in `__all__` is registered on the provider.

## Recipes

Recipes are not copied into the package.
They are downloaded from the ESMValTool repository at the commit pinned in `recipe.py`,
the same commit the conda environment installs ESMValTool from.
`recipes.txt` lists each recipe path with its sha256 checksum.

To use a new recipe, it must exist upstream at the pinned commit.
Add its path and checksum to `recipes.txt` and set `base_recipe` to the path.
Moving the pinned commit can change any recipe,
so every checksum must be checked and every recipe fixture regenerated.

`write_recipe` loads the base recipe, calls `update_recipe` and `reduce_recipe_for_regression_fixture`,
then rewrites the MIP names for CMIP7 data and writes `recipe.yml`.
`reduce_recipe_for_regression_fixture` does nothing by default.
Override it to shrink a recipe whose test case would give a large baseline,
and only change the recipe when `definition.key` starts with `"test-"`.
`update_recipe` edits the recipe in place.
It usually replaces `datasets`, removes the diagnostics it does not need,
and fills the variables with the datasets from `input_files`.

Helpers in `recipe.py`:

| Helper | Use |
| --- | --- |
| `dataframe_to_recipe(files)` | Turns a dataset table into `{short_name: {"additional_datasets": [...]}}`. |
| `get_cmip_source_type(input_files)` | Picks CMIP7 when present, otherwise CMIP6. Lives in `diagnostics/base.py`. |
| `get_child_and_parent_dataset(...)` | Pairs an experiment with its parent run, as the TCR, ECS and ZEC diagnostics need. |
| `as_timerange`, `as_facets` | Convert a dataset table row into recipe values. |

Supplementary datasets such as `areacella` arrive in `input_files` too.
Remove them from the variables, as the example does.

A diagnostic that supports both CMIP6 and CMIP7 gets either in `input_files`,
so `update_recipe` must handle both.
`tcr.py` shows how, for example by dropping `branding_suffix` for CMIP6.

## Results

`build_execution_result` is written once in the base class, so a diagnostic does not override it.
It reads the `diagnostic_provenance.yml` files ESMValTool writes
and puts every listed plot into the output bundle as a plot and every other file as data,
with the ESMValTool caption as its name.

Each file gets the dimensions of the first `files` or `series` entry whose `file_pattern` matches it.
Patterns are relative to the ESMValTool session directory, so they start with `work/` or `plots/`.
A pattern may use the group's selector names, eg. `{source_id}`.

Each `SeriesDefinition` reads one series from the netCDF files that match its pattern.
`values_name` and `index_name` name the variables to read,
and `sel` picks a slice first.
String values in `sel` are formatted with the group's selectors, eg. `{"dataset": "{source_id}"}`.
A series counts as a reference series only when its `dimensions` include `reference_source_id`.

Scalar metrics come from `format_result`.
The default returns an empty metric bundle.
Override it to read the values ESMValTool wrote and fill the bundle.
From `tcr.py`:

```python
@staticmethod
def format_result(
    result_dir: Path,
    execution_dataset: ExecutionDatasetCollection,
    metric_args: MetricBundleArgs,
    output_args: OutputBundleArgs,
) -> tuple[CMECMetric, CMECOutput]:
    tcr_ds = xarray.open_dataset(result_dir / "work" / "tcr" / "calculate" / "tcr.nc")
    tcr = float(fillvalues_to_nan(tcr_ds["tcr"].values)[0])

    metric_args[MetricCV.DIMENSIONS.value] = {
        "json_structure": ["region", "metric"],
        "region": {"global": {}},
        "metric": {"tcr": {}},
    }
    metric_args[MetricCV.RESULTS.value] = {"global": {"tcr": tcr}}

    return CMECMetric.model_validate(metric_args), CMECOutput.model_validate(output_args)
```

The base class then prepends the group's selectors to the metric bundle,
so they must be listed in `facets`.

## Reference data

There are two ways to get observations.

- If the data is in obs4MIPs or obs4REF, add an obs4MIPs requirement with an obs4REF fallback
  in the same tuple as the model requirement.
  `cloud_radiative_effects.py` does this.
- If ESMValTool needs its own OBS, OBS6 or native6 data,
  add a `DataRequirement` for `SourceDatasetType.ESMValToolReference`.
  These files are listed in `dataset_registry/data.txt`,
  fetched by `ref providers setup` and ingested into the database.
  `update_recipe` must still write the matching OBS dataset entries into the recipe,
  and they must agree with the requirement.
  `sea_ice_area_basic.py` is the example to copy.
  Its test cases fetch the data with a `RegistryRequest` that uses `parse_registry_key`.

## Tests

On top of a test case and its regression baseline, two fixtures need generating for a new diagnostic.

- `tests/unit/diagnostics/test_recipes.py` compares the written recipe
  with `tests/unit/diagnostics/recipes/recipe_<slug>_<source types>.yml`.
- `tests/unit/test_solve_regression.py` compares the solver's executions
  with `tests/unit/test_solve_regression/test_solve_regression_<slug>_.yml`.

Hyphens in the slug become underscores in these names,
eg. `recipe_cloud_radiative_effects_cmip6_obs4mips.yml`.

Generate both by running the tests once with `--force-regen`, then check and commit the new files:

```bash
uv run pytest packages/climate-ref-esmvaltool/tests/unit --force-regen
```

Check that the new fixture is not empty.
These tests solve against the parquet catalogs in `tests/test-data/esgf-catalog/`,
and a diagnostic whose data is not in them gets an empty fixture that still passes.
Regenerate the catalogs with `scripts/generate_esgf_catalog.py` if your variables are missing.

Changing a recipe or a data requirement changes these fixtures,
so regenerate them and review the diff.
