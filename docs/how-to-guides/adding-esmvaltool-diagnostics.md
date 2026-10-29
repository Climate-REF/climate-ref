# Adding an ESMValTool diagnostic

An ESMValTool diagnostic fills an ESMValTool recipe with the datasets of one execution,
runs it, and reads the results back.
Read [Adding a diagnostic](adding_custom_diagnostics.md) first.
Paths below are in `packages/climate-ref-esmvaltool/src/climate_ref_esmvaltool/`.

## The diagnostic class

Subclass `ESMValToolDiagnostic` from `diagnostics/base.py`.
On top of the usual attributes, set `base_recipe` and write `update_recipe`.
`GlobalMeanTimeseries` in `diagnostics/example.py` is the simplest one to copy:

```python
class GlobalMeanTimeseries(ESMValToolDiagnostic):
    name = "Global Mean Timeseries"
    slug = "global-mean-timeseries"
    base_recipe = "examples/recipe_python.yml"
    data_requirements = ...

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

`update_recipe` edits the recipe in place.
It usually swaps in the datasets from `input_files` and removes the parts it does not need.
Supplementary datasets such as `areacella` arrive in `input_files` too, so drop them from the variables,
as the example does.
If the diagnostic takes CMIP6 and CMIP7, handle both. `tcr.py` shows how.

To register the class, import it in `diagnostics/__init__.py` and add its name to `__all__`.

## Recipes

Recipes are downloaded from ESMValTool at the commit pinned in `recipe.py`.
To use a new recipe, add its path and sha256 checksum to `recipes.txt`.

## Results

The base class builds the results from the files ESMValTool writes,
so do not override `build_execution_result`.

- `files` and `series` give each output file its dimensions.
  Their `file_pattern`s start with `work/` or `plots/`.
- A series is a reference series when its `dimensions` include `reference_source_id`.
- For scalar values, override `format_result`. `tcr.py` is a short example.

## Reference data

- Data in obs4MIPs or obs4REF: add an obs4MIPs requirement with an obs4REF fallback,
  as in `cloud_radiative_effects.py`.
- ESMValTool's own OBS data: add a requirement for `SourceDatasetType.ESMValToolReference`
  and list the files in `dataset_registry/data.txt`.
  `sea_ice_area_basic.py` shows how.

## Tests

Besides the test case, two fixtures are compared on every test run:
the written recipe, in `tests/unit/diagnostics/recipes/`,
and the solver's executions, in `tests/unit/test_solve_regression/`.
Create or update them with:

```bash
uv run pytest packages/climate-ref-esmvaltool/tests/unit --force-regen
```

Check the new solve fixture is not empty.
An empty fixture means your data is not in `tests/test-data/esgf-catalog/`.
Add it with `scripts/generate_esgf_catalog.py`.
