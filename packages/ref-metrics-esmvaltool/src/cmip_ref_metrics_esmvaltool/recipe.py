from __future__ import annotations

import importlib.resources
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pooch
from ruamel.yaml import YAML

from cmip_ref_metrics_esmvaltool.types import Recipe

if TYPE_CHECKING:
    import pandas as pd

yaml = YAML()

FACETS = {
    "CMIP6": {
        "activity": "activity_id",
        "dataset": "source_id",
        "ensemble": "member_id",
        "institute": "institution_id",
        "exp": "experiment_id",
        "grid": "grid_label",
        "mip": "table_id",
        "short_name": "variable_id",
    },
}


def as_isodate(timestamp: pd.Timestamp) -> str:
    """Format a timestamp as an ISO 8601 datetime.

    For example, '2014-12-16 12:00:00' will be formatted as '20141216T120000'.

    Parameters
    ----------
    timestamp
        The timestamp to format.

    """
    return str(timestamp).replace(" ", "T").replace("-", "").replace(":", "")


def as_timerange(group: pd.DataFrame) -> str | None:
    """Format the timeranges from a dataframe as an ESMValTool timerange.

    Parameters
    ----------
    group
        The dataframe describing a single dataset.

    Returns
    -------
        A timerange.
    """
    start_times = group.start_time.dropna()
    if start_times.empty:
        return None
    end_times = group.end_time.dropna()
    if end_times.empty:
        return None  # pragma: no cover
    return f"{as_isodate(start_times.min())}/{as_isodate(end_times.max())}"


def as_facets(
    group: pd.DataFrame,
) -> dict[str, Any]:
    """Convert a group from the datasets dataframe to ESMValTool facets.

    Parameters
    ----------
    group:
        A group of datasets representing a single instance_id.

    Returns
    -------
        A :obj:`dict` containing facet-value pairs.

    """
    facets = {}
    first_row = group.iloc[0]
    project = first_row.instance_id.split(".", 2)[0]
    facets["project"] = project
    for esmvaltool_name, ref_name in FACETS[project].items():
        facets[esmvaltool_name] = getattr(first_row, ref_name)
    timerange = as_timerange(group)
    if timerange is not None:
        facets["timerange"] = timerange
    return facets


def dataframe_to_recipe(files: pd.DataFrame) -> dict[str, Any]:
    """Convert the datasets dataframe to a recipe "variables" section.

    Parameters
    ----------
    files
        The pandas dataframe describing the input files.

    Returns
    -------
        A "variables" section that can be used in an ESMValTool recipe.
    """
    variables: dict[str, Any] = {}
    # TODO: refine to make it possible to combine historical and scenario runs.
    for _, group in files.groupby("instance_id"):
        facets = as_facets(group)
        short_name = facets.pop("short_name")
        if short_name not in variables:
            variables[short_name] = {"additional_datasets": []}
        variables[short_name]["additional_datasets"].append(facets)
    return variables


_ESMVALTOOL_VERSION = "2.11.0"

_RECIPES = pooch.create(
    path=pooch.os_cache("cmip_ref_metrics_esmvaltool"),
    base_url="https://raw.githubusercontent.com/ESMValGroup/ESMValTool/refs/tags/v{version}/esmvaltool/recipes/",
    version=_ESMVALTOOL_VERSION,
    env="REF_METRICS_ESMVALTOOL_DATA_DIR",
)
_RECIPES.load_registry(importlib.resources.open_binary("cmip_ref_metrics_esmvaltool", "recipes.txt"))


def load_recipe(recipe: str) -> Recipe:
    """Load a recipe.

    Parameters
    ----------
    recipe
        The name of an ESMValTool recipe.

    Returns
    -------
        The loaded recipe.
    """
    filename = _RECIPES.fetch(recipe)
    return yaml.load(Path(filename).read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def prepare_climate_data(datasets: pd.DataFrame, climate_data_dir: Path) -> None:
    """Symlink the input files from the Pandas dataframe into a directory tree.

    This ensures that ESMValTool can find the data and only uses the
    requested data.

    Parameters
    ----------
    datasets
        The pandas dataframe describing the input datasets.
    climate_data_dir
        The directory where ESMValTool should look for input data.
    """
    for row in datasets.itertuples():
        if not isinstance(row.instance_id, str):  # pragma: no branch
            msg = f"Invalid instance_id encountered in {row}"
            raise ValueError(msg)
        if not isinstance(row.path, str):  # pragma: no branch
            msg = f"Invalid path encountered in {row}"
            raise ValueError(msg)
        tgt = climate_data_dir.joinpath(*row.instance_id.split(".")) / Path(row.path).name
        tgt.parent.mkdir(parents=True, exist_ok=True)
        tgt.symlink_to(row.path)
