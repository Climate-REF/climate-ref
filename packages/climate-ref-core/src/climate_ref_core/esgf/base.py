"""
Base classes and protocols for ESGF data requests.

This module provides the infrastructure for fetching datasets from ESGF
using the intake-esgf package.
"""

import copy
from collections.abc import Mapping, Sequence
from typing import Any, ClassVar, Protocol, Self, cast, runtime_checkable

import pandas as pd
from intake_esgf import ESGFCatalog
from intake_esgf.exceptions import NoSearchResults
from loguru import logger

from climate_ref_core.exceptions import DatasetResolutionError


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

    def pin_to_datasets(self, datasets: Sequence[Mapping[str, Any]]) -> "ESGFRequest":
        """
        Return a copy of this request that limits its resolution to the given datasets.

        Parameters
        ----------
        datasets
            The catalog records of the datasets recorded for this request's source type.

        Returns
        -------
        :
            The pinned request, or this request unchanged when the records do not carry
            what the pin needs. Pinning is all-or-nothing, so a pinned request never
            silently resolves a subset of what was recorded.
        """
        ...


def facet_pins_from_datasets(
    datasets: Sequence[Mapping[str, Any]],
    fields: Mapping[str, str],
    slug: str,
) -> tuple[dict[str, str], ...] | None:
    """
    Read one facet set per recorded dataset, for pins that a search cannot ask for by id.

    Parameters
    ----------
    datasets
        The catalog records to pin to.
    fields
        The record field each search facet is spelled as, keyed by facet name.
    slug
        Slug of the request being pinned, for the warning when a record falls short.

    Returns
    -------
    :
        The facet sets, deduplicated because a dataset is recorded once per file it
        holds, or ``None`` if any record is missing one of the facets.
    """
    pins = []
    for dataset in datasets:
        pin = {
            facet: str(dataset[field])
            for facet, field in fields.items()
            if dataset.get(field) not in (None, "")
        }
        if len(pin) != len(fields):
            logger.warning(
                "Recorded datasets are missing facets needed to identify them; "
                f"resolving request {slug} from its declared facets instead"
            )
            return None
        pins.append(pin)

    if not pins:
        return None

    unique = dict.fromkeys(tuple(sorted(pin.items())) for pin in pins)
    return tuple(dict(facets) for facets in unique)


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

    Subclasses must define ``source_type`` and ``project``.

    A request can also be pinned to the datasets an existing test case catalog records,
    see :meth:`pin_to_datasets`.
    """

    slug: str
    project: str
    """ESGF project to search, always sent because intake-esgf's default varies by release."""
    facets: dict[str, str | tuple[str, ...]]
    remove_ensembles: bool
    time_span: tuple[str, str] | None

    pinned_instance_ids: tuple[str, ...] | None = None
    """Ids of the datasets this request is pinned to."""

    facet_renames: ClassVar[Mapping[str, str]] = {}
    """REF facet names that the ESGF index spells differently."""

    def __init__(
        self,
        slug: str,
        facets: dict[str, Any],
        remove_ensembles: bool = False,
        time_span: tuple[str, str] | None = None,
    ):
        """
        Initialize an ESGF request.

        Parameters
        ----------
        slug
            Unique identifier for this request
        facets
            ESGF search facets (e.g., source_id, variable_id, experiment_id)
        remove_ensembles
            If True, keep only one ensemble member per model
        time_span
            Optional time range filter (start, end) in YYYY-MM format
        """
        self.slug = slug
        self.facets = facets
        self.remove_ensembles = remove_ensembles
        self.time_span = time_span

    def pin_to_datasets(self, datasets: Sequence[Mapping[str, Any]]) -> Self:
        """
        Return a copy of this request that only resolves the given datasets.

        The ids a catalog records are the ones ESGF publishes the datasets under, so they
        are used in the search instead of the facets if possible.

        Parameters
        ----------
        datasets
            The catalog records of the datasets recorded for this request's source type.

        Returns
        -------
        :
            The pinned request, or this request unchanged if any record has no id.
            Pinning is all-or-nothing, so a pinned request never silently resolves a
            subset of what was recorded.
        """
        instance_ids = [dataset.get("instance_id") for dataset in datasets]
        if not instance_ids or None in instance_ids:
            logger.warning(
                f"Recorded datasets have no instance_id; "
                f"resolving request {self.slug} from its declared facets instead"
            )
            return self

        pinned = copy.copy(self)
        # A dataset is recorded once per file it holds
        pinned.pinned_instance_ids = tuple(dict.fromkeys(cast(list[str], instance_ids)))
        return pinned

    def _pinned_facets(self, instance_ids: tuple[str, ...]) -> dict[str, Any]:
        """Search facets that find the pinned datasets."""
        return {"instance_id": list(instance_ids)}

    def _select_pinned(self, results: pd.DataFrame) -> pd.DataFrame:
        """Narrow the search results for a pinned request down to the pinned datasets."""
        return results

    def fetch_datasets(self) -> pd.DataFrame:
        """Fetch dataset metadata from ESGF."""
        facets: dict[str, Any] = {
            "project": self.project,
            **{self.facet_renames.get(key, key): value for key, value in self.facets.items()},
        }

        if self.pinned_instance_ids:
            # Ask for the pinned datasets by id as the declared facets may now
            # resolve to something else. The project is kept because it selects how the
            # index is read rather than which datasets match.
            logger.info(f"Resolving {len(self.pinned_instance_ids)} pinned datasets for {self.slug}")
            facets = {
                "project": facets["project"],
                **self._pinned_facets(self.pinned_instance_ids),
                # A pin holds even once a newer version of the dataset has been published
                "latest": [True, False],
            }

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
            try:
                cat.search(**facets)
            except NoSearchResults:
                msg = f"ESGF search returned no results for facets: {facets}"
                raise DatasetResolutionError(msg) from None

            if self.remove_ensembles and not self.pinned_instance_ids:
                # Pinned ids already name one ensemble member each
                cat.remove_ensembles()

            if cat.df is None or cat.df.empty:
                raise DatasetResolutionError(f"No datasets found for the ESGF request {self.slug}")
            if self.pinned_instance_ids:
                cat.df = self._select_pinned(cat.df)

            path_dict = cat.to_path_dict(prefer_streaming=False, minimal_keys=False, quiet=True)
            merged_df = cat.df.merge(pd.Series(path_dict, name="files"), left_on="key", right_index=True)
            if self.pinned_instance_ids and len(merged_df) < len(cat.df):
                missing = sorted(set(cat.df["key"]) - set(path_dict))
                raise DatasetResolutionError(f"Could not retrieve pinned datasets for {self.slug}: {missing}")

            if self.time_span:
                merged_df["time_start"] = self.time_span[0]
                merged_df["time_end"] = self.time_span[1]

            return _deduplicate_datasets(merged_df)
