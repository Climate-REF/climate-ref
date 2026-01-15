"""
Registry-based dataset request implementation.

This module provides request classes for fetching datasets from pooch registries
(e.g., pmp-climatology) rather than ESGF.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

import pandas as pd
from loguru import logger

from climate_ref_core.dataset_registry import dataset_registry_manager

# Number of path parts in PMP climatology registry keys
_PMP_CLIMATOLOGY_PATH_PARTS = 5
# Number of path parts in obs4REF registry keys
_OBS4REF_PATH_PARTS = 8


def _parse_obs4ref_key(key: str) -> dict[str, Any]:
    """
    Parse an obs4REF registry key to extract metadata.

    Keys follow the pattern:
    obs4REF/{institution_id}/{source_id}/{frequency}/{variable_id}/{grid_label}/{version}/{filename}

    Where filename is:
    {variable_id}_{frequency}_{source_id}_{inst_short}_{grid_label}_{time_range}.nc

    Parameters
    ----------
    key
        The registry key (path) to parse

    Returns
    -------
        Dictionary with parsed metadata, or empty dict if parsing fails
    """
    # Example: obs4REF/MOHC/HadISST-1-1/mon/ts/gn/v20250415/ts_mon_HadISST-1-1_PCMDI_gn_187001-202501.nc
    parts = key.split("/")
    if len(parts) != _OBS4REF_PATH_PARTS:
        logger.debug(f"Unexpected obs4REF key format (expected 8 parts): {key}")
        return {}

    _, institution_id, _source_id, _frequency, _variable_id, _grid_label, version, filename = parts

    # Parse filename: {var}_{freq}_{source_id}_{inst_short}_{grid}_{time_range}.nc
    # Handle source_ids with hyphens (e.g., "HadISST-1-1", "GPCP-Monthly-3-2")
    filename_pattern = re.compile(
        r"^(?P<variable_id>[a-zA-Z0-9]+)_"
        r"(?P<frequency>[a-z]+)_"
        r"(?P<source_id>[A-Za-z0-9-]+)_"
        r"(?P<institution_short>[A-Za-z0-9-]+)_"
        r"(?P<grid_label>[a-zA-Z]+)_"
        r"(?P<time_range>\d+-\d+)\.nc$"
    )

    match = filename_pattern.match(filename)
    if not match:
        logger.debug(f"obs4REF filename doesn't match expected pattern: {filename}")
        return {}

    metadata = match.groupdict()

    # Add path-derived metadata (can override filename metadata for consistency)
    metadata["institution_id"] = institution_id
    metadata["version"] = version

    # Parse time range (format: YYYYMM-YYYYMM)
    time_parts = metadata["time_range"].split("-")
    if len(time_parts) == 2:  # noqa: PLR2004
        metadata["time_start"] = time_parts[0]
        metadata["time_end"] = time_parts[1]

    # Add the full key for reference
    metadata["key"] = key

    return metadata


def _parse_pmp_climatology_key(key: str) -> dict[str, Any]:
    """
    Parse a PMP climatology registry key to extract metadata.

    Keys follow the pattern:
    PMP_obs4MIPsClims/{variable_id}/{grid_label}/{version}/{filename}

    Where filename is:
    {variable_id}_mon_{source_id}_{institution_id}_{grid_label}_{time_range}_AC_{version}_{resolution}.nc

    Parameters
    ----------
    key
        The registry key (path) to parse

    Returns
    -------
        Dictionary with parsed metadata, or empty dict if parsing fails
    """
    # Example: PMP_obs4MIPsClims/psl/gr/v20250224/
    #          psl_mon_ERA-5_PCMDI_gr_198101-200412_AC_v20250224_2.5x2.5.nc
    parts = key.split("/")
    if len(parts) != _PMP_CLIMATOLOGY_PATH_PARTS:
        logger.debug(f"Unexpected key format (expected 5 parts): {key}")
        return {}

    _, _variable_id_dir, _grid_label, _version, filename = parts

    # Parse filename: {var}_mon_{source_id}_{inst_id}_{grid}_{time}_AC_{ver}_{res}.nc
    # Handle source_ids with hyphens (e.g., "ERA-5", "GPCP-Monthly-3-2")
    filename_pattern = re.compile(
        r"^(?P<variable_id>[a-z]+)_mon_"
        r"(?P<source_id>[A-Za-z0-9-]+)_"
        r"(?P<institution_id>[A-Za-z0-9]+)_"
        r"(?P<grid_label>[a-z]+)_"
        r"(?P<time_range>\d+-\d+)_AC_"
        r"(?P<version>v\d+)_"
        r"(?P<resolution>.+)\.nc$"
    )

    match = filename_pattern.match(filename)
    if not match:
        logger.debug(f"Filename doesn't match expected pattern: {filename}")
        return {}

    metadata = match.groupdict()

    # Parse time range (format: YYYYMM-YYYYMM)
    time_parts = metadata["time_range"].split("-")
    if len(time_parts) == 2:  # noqa: PLR2004
        metadata["time_start"] = time_parts[0]
        metadata["time_end"] = time_parts[1]

    # Add the full key for reference
    metadata["key"] = key

    return metadata


def _matches_facets(
    metadata: dict[str, Any],
    facets: dict[str, str | tuple[str, ...]],
) -> bool:
    """
    Check if metadata matches all provided facets.

    Parameters
    ----------
    metadata
        Parsed metadata dictionary
    facets
        Facets to match against. Values can be strings or tuples of strings.

    Returns
    -------
        True if all facets match
    """
    for facet_name, facet_value in facets.items():
        if facet_name not in metadata:
            return False

        # Normalize to tuple for comparison
        allowed_values = (facet_value,) if isinstance(facet_value, str) else facet_value

        if metadata[facet_name] not in allowed_values:
            return False

    return True


class RegistryRequest:
    """
    Request for data from a pooch registry (e.g., pmp-climatology).

    These data are fetched from a pooch registry rather than ESGF.
    This is useful for pre-processed datasets like PMP climatologies
    that are hosted externally but not on ESGF.

    Parameters
    ----------
    slug
        Unique identifier for this request
    registry_name
        Name of the registry to fetch from (e.g., "pmp-climatology")
    facets
        Facets to filter datasets (e.g., {"variable_id": "psl", "source_id": "ERA-5"})
    source_type
        Type of dataset source (default: "PMPClimatology")
    time_span
        Optional time range filter (not used for registry filtering, but required for protocol)

    Example
    -------
    ```python
    request = RegistryRequest(
        slug="era5-psl",
        registry_name="pmp-climatology",
        facets={"variable_id": "psl", "source_id": "ERA-5"},
    )
    df = request.fetch_datasets()
    ```
    """

    def __init__(
        self,
        slug: str,
        registry_name: str,
        facets: dict[str, str | tuple[str, ...]],
        source_type: str = "PMPClimatology",
        time_span: tuple[str, str] | None = None,
    ) -> None:
        self.slug = slug
        self.registry_name = registry_name
        self.facets = facets
        self.source_type = source_type
        self.time_span = time_span

    def __repr__(self) -> str:
        return (
            f"RegistryRequest(slug={self.slug!r}, registry_name={self.registry_name!r}, "
            f"facets={self.facets!r}, source_type={self.source_type!r}, time_span={self.time_span!r})"
        )

    def _get_parser(self) -> Callable[[str], dict[str, Any]]:
        """Get the appropriate parser function based on registry name."""
        if self.registry_name == "pmp-climatology":
            return _parse_pmp_climatology_key
        elif self.registry_name == "obs4ref":
            return _parse_obs4ref_key
        else:
            # Default to obs4ref parser as fallback
            logger.warning(f"Unknown registry '{self.registry_name}', using obs4ref parser")
            return _parse_obs4ref_key

    def fetch_datasets(self) -> pd.DataFrame:
        """
        Fetch matching datasets from the registry.

        Returns
        -------
            DataFrame containing dataset metadata and file paths.
            Each row represents one file, with columns for metadata
            and a 'files' column containing a list with the file path.
        """
        logger.info(f"Fetching from registry '{self.registry_name}' for request: {self.slug}")

        try:
            registry = dataset_registry_manager[self.registry_name]
        except KeyError:
            raise ValueError(
                f"Registry '{self.registry_name}' not found. "
                f"Available registries: {list(dataset_registry_manager.keys())}"
            )

        parser = self._get_parser()
        matching_rows: list[dict[str, Any]] = []

        for key in registry.registry.keys():
            # Parse metadata from the registry key
            metadata = parser(key)
            if not metadata:
                continue

            # Check if it matches the requested facets
            if not _matches_facets(metadata, self.facets):
                continue

            # Fetch the file (downloads if not cached)
            try:
                file_path = registry.fetch(key)
                logger.debug(f"Fetched: {key} -> {file_path}")
            except Exception as e:
                logger.warning(f"Failed to fetch {key}: {e}")
                continue

            # Build row compatible with ESGFFetcher expectations
            row = {
                **metadata,
                "files": [file_path],
                "path": file_path,
            }
            matching_rows.append(row)

        if not matching_rows:
            logger.warning(f"No datasets found matching facets: {self.facets}")
            return pd.DataFrame()

        result = pd.DataFrame(matching_rows)
        logger.info(f"Found {len(result)} datasets matching request: {self.slug}")

        return result
