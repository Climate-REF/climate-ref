"""
Base classes and protocols for ESGF data requests.

This module provides the infrastructure for fetching datasets from ESGF
using the intake-esgf package.
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import pandas as pd
from intake_esgf import ESGFCatalog

if TYPE_CHECKING:
    import xarray as xr


@runtime_checkable
class ESGFRequest(Protocol):
    """
    Protocol for ESGF dataset requests.

    Implementations provide the logic for searching ESGF and generating
    output paths for downloaded datasets.
    """

    slug: str
    """Unique identifier for this request."""

    source_type: str
    """Type of dataset (e.g., 'CMIP6', 'obs4MIPs')."""

    time_span: tuple[str, str] | None
    """Optional time range to filter datasets (start, end)."""

    def fetch_datasets(self) -> pd.DataFrame:
        """
        Fetch dataset metadata from ESGF.

        Returns
        -------
        pd.DataFrame
            DataFrame containing dataset metadata and file paths.
            Must contain at minimum:
            - key: A unique identifier for the dataset
            - files: A list of files for the dataset
        """
        ...

    def generate_output_path(
        self, metadata: pd.Series[Any], ds: xr.Dataset, ds_filename: pathlib.Path
    ) -> pathlib.Path:
        """
        Generate the local output path for a dataset.

        Parameters
        ----------
        metadata
            Row from the DataFrame returned by fetch_datasets
        ds
            Loaded xarray dataset
        ds_filename
            Original filename of the dataset

        Returns
        -------
        pathlib.Path
            Relative path where the dataset should be stored
        """
        ...


def _deduplicate_datasets(datasets: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate a dataset collection.

    Uses the metadata from the first dataset in each group,
    but expands the time range to the min/max timespan of the group.

    Parameters
    ----------
    datasets
        The dataset collection

    Returns
    -------
    pd.DataFrame
        The deduplicated dataset collection spanning the times requested
    """

    def _deduplicate_group(group: pd.DataFrame) -> pd.DataFrame:
        first = group.iloc[[0]].copy()
        if "time_start" in first.columns:
            first["time_start"] = group["time_start"].min()
        if "time_end" in first.columns:
            first["time_end"] = group["time_end"].max()
        return first

    result: pd.DataFrame = (
        datasets.groupby("key")
        .apply(_deduplicate_group, include_groups=False)  # type: ignore[call-overload]
        .reset_index()
    )
    return result


class IntakeESGFMixin:
    """
    Mixin that fetches datasets from ESGF using intake-esgf.

    Subclasses must define:
    - facets: dict[str, str | tuple[str, ...]]
    - remove_ensembles: bool
    - time_span: tuple[str, str] | None
    """

    facets: dict[str, str | tuple[str, ...]]
    remove_ensembles: bool
    time_span: tuple[str, str] | None

    def fetch_datasets(self) -> pd.DataFrame:
        """Fetch dataset metadata from ESGF."""
        facets: dict[str, Any] = dict(self.facets)
        if self.time_span:
            facets["file_start"] = self.time_span[0]
            facets["file_end"] = self.time_span[1]

        cat = ESGFCatalog()  # type: ignore[no-untyped-call]
        cat.search(**facets)

        if self.remove_ensembles:
            cat.remove_ensembles()

        path_dict = cat.to_path_dict(prefer_streaming=False, minimal_keys=False, quiet=True)
        if cat.df is None or cat.df.empty:
            raise ValueError("No datasets found for the given ESGF request")
        merged_df = cat.df.merge(pd.Series(path_dict, name="files"), left_on="key", right_index=True)

        if self.time_span:
            merged_df["time_start"] = self.time_span[0]
            merged_df["time_end"] = self.time_span[1]

        return _deduplicate_datasets(merged_df)
