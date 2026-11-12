import pandas as pd
from climate_ref_esmvaltool import provider
from climate_ref_esmvaltool.diagnostics.climate_at_global_warming_levels import ClimateAtGlobalWarmingLevels

from climate_ref.solver import solve_executions
from climate_ref_core.datasets import SourceDatasetType

BRANDING = {
    "pr": "tavg-u-hxy-u",
    "tas": "tavg-h2m-hxy-u",
    "areacella": "ti-u-hxy-u",
}


def _row(experiment_id: str, variable_id: str, start: str | None, end: str | None, **overrides) -> dict:
    frequency = "fx" if variable_id == "areacella" else "mon"
    branding = overrides.pop("branding", BRANDING[variable_id])
    row = {
        "instance_id": (
            f"CMIP7.ScenarioMIP.UKNCSP.UKESM1-3-LL.{experiment_id}.r1i1p1f1."
            f"glb.{frequency}.{variable_id}.{branding}.g110.v20260101"
        ),
        "source_id": "UKESM1-3-LL",
        "activity_id": "ScenarioMIP",
        "institution_id": "UKNCSP",
        "realm": "atmos",
        "experiment_id": experiment_id,
        "variant_label": "r1i1p1f1",
        "grid_label": "g110",
        "variable_id": variable_id,
        "branded_variable": f"{variable_id}_{branding}",
        "branding_suffix": branding,
        "frequency": frequency,
        "region": "glb",
        "start_time": pd.Timestamp(start) if start else pd.NaT,
        "end_time": pd.Timestamp(end) if end else pd.NaT,
        "version": "v20260101",
        "path": f"{experiment_id}_{variable_id}_{branding}.nc",
    }
    row.update(overrides)
    return row


def _catalog(*pairs: tuple[str, str]) -> pd.DataFrame:
    rows = []
    for scenario, parent in pairs:
        for variable_id in ("pr", "tas"):
            rows.append(_row(parent, variable_id, "1850-01-16", "2021-12-16"))
            rows.append(_row(scenario, variable_id, "2022-01-16", "2100-12-16"))
        rows.append(_row(parent, "areacella", None, None))
    # A second tas branding in the parent must not be pulled in alongside the monthly mean.
    rows.append(_row("historical", "tas", "1850-01-16", "2021-12-16", branding="tmaxavg-h2m-hxy-u"))
    return pd.DataFrame(rows)


def _experiments(execution) -> set[str]:
    datasets = execution.datasets[SourceDatasetType.CMIP7].datasets
    return set(datasets[datasets["variable_id"] != "areacella"]["experiment_id"])


def _diagnostic() -> ClimateAtGlobalWarmingLevels:
    return next(d for d in provider.diagnostics() if d.slug == "climate-at-global-warming-levels")


def test_cmip7_concentration_and_emissions_driven_scenarios_solve_separately() -> None:
    """Pair each scenario with its own parent, never mixing the two forcing types."""
    diagnostic = _diagnostic()
    catalog = _catalog(("scen7-m", "historical"), ("esm-scen7-m", "esm-hist"))

    executions = list(solve_executions({SourceDatasetType.CMIP7: catalog}, diagnostic, diagnostic.provider))

    experiments = {frozenset(_experiments(execution)) for execution in executions}
    assert experiments == {
        frozenset({"scen7-m", "historical"}),
        frozenset({"esm-scen7-m", "esm-hist"}),
    }
    for execution in executions:
        datasets = execution.datasets[SourceDatasetType.CMIP7].datasets
        assert "tas_tmaxavg-h2m-hxy-u" not in set(datasets["branded_variable"])


def test_cmip7_scenario_without_its_parent_does_not_solve() -> None:
    """An emissions-driven scenario cannot borrow the concentration-driven historical run."""
    diagnostic = _diagnostic()
    catalog = _catalog(("scen7-m", "historical"), ("esm-scen7-m", "esm-hist"))
    catalog = catalog[catalog["experiment_id"] != "esm-hist"]

    executions = list(solve_executions({SourceDatasetType.CMIP7: catalog}, diagnostic, diagnostic.provider))

    assert [_experiments(execution) for execution in executions] == [{"scen7-m", "historical"}]


def test_update_recipe_drops_parent_only_datasets() -> None:
    """Keep only scenario datasets at the top level for both forcing types."""
    rows = [
        _row(experiment_id, variable_id, start, end)
        for experiment_id, start, end in (
            ("esm-hist", "1850-01-16", "2021-12-16"),
            ("esm-scen7-m", "2022-01-16", "2100-12-16"),
        )
        for variable_id in ("pr", "tas")
    ]
    recipe = {
        "diagnostics": {
            name: {"additional_datasets": [], "variables": {}}
            for name in ("calculate_gwl_exceedance_years", "gwl_mean_plots_tas", "gwl_mean_plots_pr")
        }
    }

    ClimateAtGlobalWarmingLevels.update_recipe(recipe, {SourceDatasetType.CMIP7: pd.DataFrame(rows)})

    assert [dataset["exp"] for dataset in recipe["datasets"]] == [["esm-hist", "esm-scen7-m"]]
    assert "branding_suffix" not in recipe["datasets"][0]
    brandings = {
        variable["short_name"]: variable["branding_suffix"]
        for diagnostic in recipe["diagnostics"].values()
        for variable in diagnostic["variables"].values()
    }
    assert brandings == {"pr": "tavg-u-hxy-u", "tas": "tavg-h2m-hxy-u"}
