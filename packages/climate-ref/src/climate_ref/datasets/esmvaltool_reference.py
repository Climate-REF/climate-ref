"""
Adapter for ESMValTool reference (observational/reanalysis) datasets.

ESMValTool reference data is *not* CMOR/obs4MIPs compliant, so metadata cannot be read
from global attributes the way :mod:`climate_ref.datasets.obs4mips` does. Instead it is
parsed from the ESMValCore DRS path and filename templates by
:mod:`climate_ref_core.esmvaltool_reference`, which describes the layouts.
"""

from __future__ import annotations

import traceback
from pathlib import Path
from typing import Any

import pandas as pd
from loguru import logger

from climate_ref.datasets.base import DatasetAdapter
from climate_ref.datasets.catalog_builder import build_catalog
from climate_ref.datasets.utils import (
    build_instance_id,
    parse_cftime_dates,
    parse_drs_daterange,
)
from climate_ref.models.dataset import Dataset, ESMValToolReferenceDataset
from climate_ref_core.esmvaltool_reference import parse_reference_path

_SLUG_PREFIX = "esmvaltool-reference"

# Metadata columns (in order) that make up the dataset ``instance_id`` slug.
_INSTANCE_ID_FACETS = ("project", "source_id", "frequency", "variable_id", "version")


def parse_esmvaltool_reference(file: str, **kwargs: Any) -> dict[str, Any]:
    """
    Parse a single ESMValTool reference file into a metadata record.

    Metadata comes from the path rather than the file contents,
    because the data is not CMOR compliant.
    """
    try:
        info = parse_reference_path(file)._asdict()
        timerange = info.pop("timerange")

        info["start_time"], info["end_time"] = parse_drs_daterange(timerange) if timerange else (None, None)
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
        "frequency",
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
        "frequency",
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
