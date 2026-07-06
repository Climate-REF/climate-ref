"""
Adapter for ESMValTool reference (observational/reanalysis) datasets.

ESMValTool reference data is *not* CMOR/obs4MIPs compliant, so metadata cannot be read
from global attributes the way :mod:`climate_ref.datasets.obs4mips` does. Instead it is
parsed from the ESMValCore DRS path and filename templates that ESMValTool itself uses to
locate the data at run time (see ``climate_ref_esmvaltool.diagnostics.base``):

* ``OBS`` / ``OBS6`` (metadata from the filename):
  ``OBS/Tier{tier}/{dataset}/{project}_{dataset}_{type}_{version}_{mip}_{short_name}_{timerange}.nc``
* ``native6`` (metadata from the directory; raw non-CMOR filename):
  ``native6/Tier{tier}/{dataset}/{version}/{frequency}/{short_name}/*.nc``
* ``obs4MIPs``: ``obs4MIPs/{dataset}/{version}/{short_name}_*.nc``
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from climate_ref.datasets.base import DatasetAdapter
from climate_ref.datasets.catalog_builder import build_catalog
from climate_ref.datasets.utils import build_instance_id, parse_cftime_dates, parse_drs_daterange
from climate_ref.models.dataset import Dataset, ESMValToolReferenceDataset

# Top-level ESMValTool reference project directories (relative to the ``ESMValTool`` data root).
# ``OBS6`` data lives under the ``OBS`` directory; the project is recovered from the filename.
_PROJECT_ANCHORS = ("OBS", "native6", "obs4MIPs")

_SLUG_PREFIX = "esmvaltool-reference"

# Metadata columns (in order) that make up the dataset ``instance_id`` slug.
_INSTANCE_ID_FACETS = ("project", "source_id", "table_id", "variable_id", "version")


def _tier_from_segment(segment: str) -> int | None:
    """Parse ``Tier2`` -> ``2``; return ``None`` if the segment is not a tier."""
    if segment.startswith("Tier") and segment[4:].isdigit():
        return int(segment[4:])
    return None


def _parse_obs(rel: tuple[str, ...], filename: str) -> dict[str, Any]:
    # rel == ("OBS", "Tier{n}", "{dataset}", ..., filename)
    tier = _tier_from_segment(rel[1])
    dataset = rel[2]
    stem = filename[:-3] if filename.endswith(".nc") else filename
    tokens = stem.split("_")
    # {project}_{dataset}_{type}_{version}_{mip}_{short_name}[_{timerange}]
    if len(tokens) < 6:  # noqa: PLR2004
        raise ValueError(f"unexpected OBS filename structure: {filename}")
    project, _, data_type, version, mip, short_name = tokens[:6]
    # The timerange is the trailing token; use ``tokens[-1]`` (matching ``_parse_obs4mips``)
    # so an unexpected extra segment does not silently drop the date range.
    timerange = tokens[-1] if len(tokens) > 6 else None  # noqa: PLR2004
    start_time, end_time = parse_drs_daterange(timerange) if timerange else (None, None)
    return {
        "project": project,
        "source_id": dataset,
        "variable_id": short_name,
        "table_id": mip,
        "version": version,
        "data_type": data_type,
        "tier": tier,
        "start_time": start_time,
        "end_time": end_time,
    }


def _parse_native6(rel: tuple[str, ...]) -> dict[str, Any]:
    # rel == ("native6", "Tier{n}", "{dataset}", "{version}", "{frequency}", "{short_name}", filename)
    if len(rel) < 7:  # noqa: PLR2004
        raise ValueError(f"unexpected native6 path structure: {'/'.join(rel)}")
    tier = _tier_from_segment(rel[1])
    dataset, version, frequency, short_name = rel[2], rel[3], rel[4], rel[5]
    return {
        "project": "native6",
        "source_id": dataset,
        "variable_id": short_name,
        "table_id": frequency,
        "version": version,
        "data_type": None,
        "tier": tier,
        # native6 filenames are raw (non-CMOR) and carry no reliable DRS date range.
        "start_time": None,
        "end_time": None,
    }


def _parse_obs4mips(rel: tuple[str, ...], filename: str) -> dict[str, Any]:
    # rel == ("obs4MIPs", "{dataset}", "{version}", filename)
    if len(rel) < 4:  # noqa: PLR2004
        raise ValueError(f"unexpected obs4MIPs path structure: {'/'.join(rel)}")
    dataset, version = rel[1], rel[2]
    stem = filename[:-3] if filename.endswith(".nc") else filename
    tokens = stem.split("_")
    if not tokens[0]:
        raise ValueError(f"unexpected obs4MIPs filename structure: {filename}")
    short_name = tokens[0]
    timerange = tokens[-1] if len(tokens) > 1 else None
    start_time, end_time = parse_drs_daterange(timerange) if timerange else (None, None)
    return {
        "project": "obs4MIPs",
        "source_id": dataset,
        "variable_id": short_name,
        # obs4MIPs reference files here are monthly; use "mon" as the non-null grouping key.
        "table_id": "mon",
        "version": version,
        "data_type": None,
        "tier": None,
        "start_time": start_time,
        "end_time": end_time,
    }


def parse_esmvaltool_reference(file: str, **kwargs: Any) -> dict[str, Any]:
    """
    Parse a single ESMValTool reference file into a metadata record.

    Dispatches on the top-level project directory (``OBS``/``native6``/``obs4MIPs``) and
    reads metadata from the path/filename rather than the file contents, because the data
    is not CMOR compliant.
    """
    try:
        path = Path(file)
        parts = path.parts

        anchor_idx = next((i for i, part in enumerate(parts) if part in _PROJECT_ANCHORS), None)
        if anchor_idx is None:
            raise ValueError(
                f"{file} is not under a known ESMValTool reference project ({', '.join(_PROJECT_ANCHORS)})"
            )
        rel = parts[anchor_idx:]
        anchor = rel[0]

        if anchor == "OBS":
            info = _parse_obs(rel, path.name)
        elif anchor == "native6":
            info = _parse_native6(rel)
        else:
            info = _parse_obs4mips(rel, path.name)

        info["path"] = str(file)
        info["long_name"] = None
        info["units"] = None
        return info
    except (ValueError, IndexError) as err:
        logger.warning(str(err))
        return {"INVALID_ASSET": file, "TRACEBACK": str(err)}
    except Exception:
        logger.warning(traceback.format_exc())
        return {"INVALID_ASSET": file, "TRACEBACK": traceback.format_exc()}


class ESMValToolReferenceDatasetAdapter(DatasetAdapter):
    """
    Adapter for ESMValTool reference datasets.

    See the module docstring for the layout conventions this adapter understands.
    """

    dataset_cls: type[Dataset] = ESMValToolReferenceDataset
    slug_column = "instance_id"

    dataset_specific_metadata = (
        "project",
        "source_id",
        "variable_id",
        "table_id",
        "version",
        "data_type",
        "tier",
        "long_name",
        "units",
        "finalised",
        slug_column,
    )

    file_specific_metadata = ("start_time", "end_time", "path")
    version_metadata = "version"
    dataset_id_metadata = (
        "project",
        "source_id",
        "table_id",
        "variable_id",
    )

    def __init__(self, n_jobs: int = 1):
        self.n_jobs = n_jobs

    def find_local_datasets(self, file_or_directory: Path) -> pd.DataFrame:
        """
        Generate a data catalog from the specified file or directory.

        Each dataset may contain multiple files (rows). The unique dataset identifier is
        the ``instance_id`` slug in :attr:`slug_column`.
        """
        datasets = build_catalog(
            paths=[str(file_or_directory)],
            parsing_func=parse_esmvaltool_reference,
            include_patterns=["*.nc"],
            depth=10,
            n_jobs=self.n_jobs,
        )
        if datasets.empty:
            logger.error("No datasets found")
            raise ValueError("No ESMValTool reference datasets found")

        datasets["start_time"] = parse_cftime_dates(datasets["start_time"])
        datasets["end_time"] = parse_cftime_dates(datasets["end_time"])
        datasets["finalised"] = True
        return build_instance_id(datasets, list(_INSTANCE_ID_FACETS), prefix=_SLUG_PREFIX)
