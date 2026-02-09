from __future__ import annotations

import importlib.resources
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pooch
import yaml

from climate_ref_esmvaltool.types import Recipe

if TYPE_CHECKING:
    import pandas as pd


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
    "CMIP7": {
        "activity": "activity_id",
        "branding_suffix": "branding_suffix",
        "dataset": "source_id",
        "ensemble": "variant_label",
        "exp": "experiment_id",
        "frequency": "frequency",
        "grid": "grid_label",
        "institute": "institution_id",
        "mip": "realm",
        "region": "region",
        "short_name": "variable_id",
    },
    "obs4MIPs": {
        "dataset": "source_id",
        "frequency": "frequency",
        "grid": "grid_label",
        "institute": "institution_id",
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
    # TODO: apply some rounding to avoid problems?
    # https://github.com/ESMValGroup/ESMValCore/issues/2048
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
    project = group.iloc[0].instance_id.split(".", 2)[0]
    facets["project"] = project
    for esmvaltool_name, ref_name in FACETS[project].items():
        values = group[ref_name].unique().tolist()
        facets[esmvaltool_name] = values if len(values) > 1 else values[0]
    timerange = as_timerange(group)
    if timerange is not None:
        facets["timerange"] = timerange
    return facets


def dataframe_to_recipe(
    files: pd.DataFrame,
    group_by: tuple[str, ...] = ("instance_id",),
    equalize_timerange: bool = False,
) -> dict[str, Any]:
    """Convert the datasets dataframe to a recipe "variables" section.

    Parameters
    ----------
    files
        The pandas dataframe describing the input files.
    group_by
        The columns to group the input files by.
    equalize_timerange
        If True, use the timerange that is covered by all datasets.

    Returns
    -------
        A "variables" section that can be used in an ESMValTool recipe.
    """
    variables: dict[str, Any] = {}
    for _, group in files.groupby(list(group_by)):
        facets = as_facets(group)
        short_name = facets.pop("short_name")
        if short_name not in variables:
            variables[short_name] = {"additional_datasets": []}
        variables[short_name]["additional_datasets"].append(facets)

    if equalize_timerange:
        # Select a timerange covered by all datasets.
        start_times, end_times = [], []
        for variable in variables.values():
            for dataset in variable["additional_datasets"]:
                if "timerange" in dataset:
                    start, end = dataset["timerange"].split("/")
                    start_times.append(start)
                    end_times.append(end)
        timerange = f"{max(start_times)}/{min(end_times)}"
        for variable in variables.values():
            for dataset in variable["additional_datasets"]:
                if "timerange" in dataset:
                    dataset["timerange"] = timerange

    return variables


_ESMVALTOOL_COMMIT = "f5214c9242725fe9a4c3628f304917c7434b361d"
_ESMVALTOOL_VERSION = f"2.13.0.dev65+g{_ESMVALTOOL_COMMIT[:9]}"
_ESMVALTOOL_URL = f"git+https://github.com/ESMValGroup/ESMValTool.git@{_ESMVALTOOL_COMMIT}"

_ESMVALCORE_COMMIT = "71ab2196b602fe19409919265e12886ba09ba0f5"
_ESMVALCORE_URL = f"git+https://github.com/ESMValGroup/ESMValCore.git@{_ESMVALCORE_COMMIT}"

_RECIPES = pooch.create(
    path=pooch.os_cache("climate_ref_esmvaltool"),
    # TODO: use a released version
    # base_url="https://raw.githubusercontent.com/ESMValGroup/ESMValTool/refs/tags/v{version}/esmvaltool/recipes/",
    # version=_ESMVALTOOL_VERSION,
    base_url=f"https://raw.githubusercontent.com/ESMValGroup/ESMValTool/{_ESMVALTOOL_COMMIT}/esmvaltool/recipes/",
    env="REF_METRICS_ESMVALTOOL_DATA_DIR",
    retry_if_failed=10,
)
with importlib.resources.files("climate_ref_esmvaltool").joinpath("recipes.txt").open("rb") as _buffer:
    _RECIPES.load_registry(_buffer)


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

    def normalize(obj: Any) -> Any:
        # Ensure objects in the recipe are not shared.
        if isinstance(obj, dict):
            return {k: normalize(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [normalize(item) for item in obj]
        return obj

    return normalize(yaml.safe_load(Path(filename).read_text(encoding="utf-8")))  # type: ignore[no-any-return]


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
        if row.instance_id.startswith("obs4MIPs."):
            version = row.instance_id.split(".")[-1]
            subdirs: list[str] = ["obs4MIPs", row.source_id, version]  # type: ignore[list-item]
        elif row.instance_id.startswith("CMIP7."):
            subdirs = row.instance_id.split(".")
        else:
            subdirs = row.instance_id.split(".")
        tgt = climate_data_dir.joinpath(*subdirs) / Path(row.path).name
        tgt.parent.mkdir(parents=True, exist_ok=True)
        if tgt.is_symlink() or tgt.exists():
            tgt.unlink()
        tgt.symlink_to(row.path)
