import pandas as pd
import pytest
from climate_ref_esmvaltool.diagnostics import tcre

from climate_ref_core.datasets import SourceDatasetType


def test_update_recipe_uses_cmip7_esm_flat10_emissions(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use emissions from the same CO2-increase experiment as temperature."""
    monkeypatch.setattr(
        tcre,
        "get_child_and_parent_dataset",
        lambda *_args, **_kwargs: (
            {"exp": "esm-flat10", "timerange": "1850/1959"},
            {"exp": "esm-piControl", "timerange": "1850/1959"},
        ),
    )
    monkeypatch.setattr(
        tcre,
        "dataframe_to_recipe",
        lambda *_args, **_kwargs: {
            "fco2antt": {
                "additional_datasets": [
                    {"exp": "esm-flat10", "timerange": "1850/1999"},
                ],
            },
        },
    )

    recipe = {"diagnostics": {"tcre": {}}}
    tcre.TransientClimateResponseEmissions.update_recipe(
        recipe,
        {SourceDatasetType.CMIP7: pd.DataFrame({"variable_id": ["tas", "fco2antt"]})},
    )

    variables = recipe["diagnostics"]["tcre"]["variables"]
    assert variables["tas"]["additional_datasets"][0]["exp"] == "esm-flat10"
    assert variables["fco2antt"]["additional_datasets"][0]["exp"] == "esm-flat10"
    assert variables["fco2antt"]["additional_datasets"][0]["timerange"] == "1850/1959"
