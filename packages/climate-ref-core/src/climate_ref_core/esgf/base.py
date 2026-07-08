"""
Base classes and protocols for ESGF data requests.

This module provides the infrastructure for fetching datasets from ESGF
using the intake-esgf package.
"""

import os
import sys
from typing import Any, Protocol, runtime_checkable

import intake_esgf.base
import intake_esgf.catalog
import pandas as pd
from intake_esgf import ESGFCatalog
from tqdm import tqdm


class _EnvAwareTqdm(tqdm):  # type: ignore[type-arg]  # not subscriptable at runtime
    """
    A tqdm that can always be silenced by the environment.

    intake-esgf passes ``disable=`` explicitly on every progress bar, which defeats tqdm's
    own ``TQDM_DISABLE`` handling and its off-when-not-a-terminal default. Worse, its
    ``quiet`` flag never reaches ``parallel_download``, so per-file download bars render
    even when the caller asked for silence. Each bar emits an update ~10 times a second
    and downloads run one bar per thread, which buries non-interactive logs.

    Only ever tighten the caller's choice: a bar the caller already disabled stays
    disabled, and one it enabled is suppressed when ``TQDM_DISABLE`` is set or stderr is
    not a terminal.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        suppress = bool(os.environ.get("TQDM_DISABLE")) or not sys.stderr.isatty()
        kwargs["disable"] = bool(kwargs.get("disable")) or suppress
        super().__init__(*args, **kwargs)


# Both modules bind `tqdm` into their own namespace, so rebind each.
intake_esgf.base.tqdm = _EnvAwareTqdm  # type: ignore[assignment,attr-defined]
intake_esgf.catalog.tqdm = _EnvAwareTqdm  # type: ignore[assignment,attr-defined]


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

        # Convert tuples to lists for intake-esgf compatibility
        for key, value in facets.items():
            if isinstance(value, tuple):
                facets[key] = list(value)

        # intake-esgf assigns Python lists into individual DataFrame cells, which only works for object dtype.
        # Pandas >= 3.0 defaults strings to the pyarrow-backed dtype, where that assignment raises,
        # so pin the legacy object-string behaviour for the intake-esgf interaction.
        with pd.option_context("future.infer_string", False):
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
