"""
CMIP7 dataset request implementation.

Since CMIP7 data is not yet available on ESGF, this module provides
a request class that fetches CMIP6 data and converts it to CMIP7 format.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import pandas as pd
import platformdirs
import xarray as xr
from loguru import logger

from climate_ref_core.cmip6_to_cmip7 import (
    _get_dreq_entry,
    convert_cmip6_dataset,
    create_cmip7_filename,
)
from climate_ref_core.esgf.cmip6 import CMIP6Request


def _get_cmip7_cache_dir() -> Path:
    """Get the cache directory for converted CMIP7 files."""
    # Use platform-appropriate cache directory for climate-ref
    # This avoids polluting the intake-esgf cache with converted files
    cache_dir = Path(platformdirs.user_cache_dir("climate-ref")) / "cmip7-converted"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _convert_file_to_cmip7(cmip6_path: Path, cmip7_facets: dict[str, Any]) -> Path:
    """
    Convert a CMIP6 file to CMIP7 format.

    Parameters
    ----------
    cmip6_path
        Path to the CMIP6 NetCDF file
    cmip7_facets
        CMIP7 facets for the output path

    Returns
    -------
    Path
        Path to the converted CMIP7 file
    """
    cache_dir = _get_cmip7_cache_dir()

    # Enrich facets with DReq metadata (branding_suffix, region) for filename construction
    table_id = cmip7_facets.get("table_id")
    variable_id = cmip7_facets.get("variable_id")
    if table_id and variable_id and "branding_suffix" not in cmip7_facets:
        try:
            entry = _get_dreq_entry(table_id, variable_id)
            cmip7_facets = {
                **cmip7_facets,
                "branding_suffix": entry.branding_suffix,
                "region": entry.region,
                "out_name": entry.out_name,
            }
        except KeyError:
            logger.debug(f"No DReq entry for {table_id}.{variable_id}, using facets as-is")

    # Build CMIP7 DRS path
    # CMIP7 DRS: {activity_id}/{institution_id}/{source_id}/{experiment_id}/
    #            {variant_label}/{frequency}/{variable_id}/{grid_label}/{version}
    # Ensure all facet values are strings (some may be integers from metadata)
    drs_path = cache_dir / Path(
        str(cmip7_facets.get("activity_id", "CMIP")),
        str(cmip7_facets.get("institution_id", "unknown")),
        str(cmip7_facets.get("source_id", "unknown")),
        str(cmip7_facets.get("experiment_id", "historical")),
        str(cmip7_facets.get("variant_label", "r1i1p1f1")),
        str(cmip7_facets.get("frequency", "mon")),
        str(cmip7_facets.get("variable_id", "tas")),
        str(cmip7_facets.get("grid_label", "gn")),
        str(cmip7_facets.get("version", "v1")),
    )
    # Create output filename and check cache before opening the source file
    output_file = drs_path / create_cmip7_filename(cmip7_facets)

    if output_file.exists():
        logger.debug(f"Using cached CMIP7 file: {output_file}")
        return output_file

    drs_path.mkdir(parents=True, exist_ok=True)

    # Convert the file
    logger.info(f"Converting to CMIP7: {cmip6_path.name}")
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    with xr.open_dataset(cmip6_path, decode_times=time_coder) as ds:
        ds_cmip7 = convert_cmip6_dataset(ds)
        try:
            ds_cmip7.to_netcdf(output_file)
        except PermissionError:
            # If we can't write but file exists (race condition or permission issue), use it
            if output_file.exists():
                logger.debug(f"Using existing CMIP7 file (could not overwrite): {output_file}")
                return output_file
            # Clear the cache directory hint for the user
            logger.error(f"Permission denied writing to {output_file}")
            logger.error(f"Try clearing the cache: rm -rf {_get_cmip7_cache_dir()}")
            raise

    return output_file


class CMIP7Request:
    """
    Represents a CMIP7 dataset request.

    Since CMIP7 data is not yet available on ESGF, this class fetches
    CMIP6 data and converts it to CMIP7 format using convert_cmip6_dataset().

    The facets use CMIP7 naming conventions (e.g., variant_label instead of member_id).
    """

    source_type = "CMIP7"

    # Map CMIP7 facets to CMIP6 facets
    facet_mapping: ClassVar[dict[str, str]] = {
        "variant_label": "member_id",
    }

    available_facets = (
        "activity_id",
        "institution_id",
        "source_id",
        "experiment_id",
        "variant_label",  # CMIP7 name for member_id
        "variable_id",
        "grid_label",
        "frequency",
        "table_id",  # Used for mapping to CMIP6
        "version",
    )

    def __init__(
        self,
        slug: str,
        facets: dict[str, Any],
        remove_ensembles: bool = False,
        time_span: tuple[str, str] | None = None,
    ):
        """
        Initialize a CMIP7 request.

        Parameters
        ----------
        slug
            Unique identifier for this request
        facets
            CMIP7 search facets (e.g., source_id, variable_id, variant_label)
        remove_ensembles
            If True, keep only one ensemble member per model
        time_span
            Optional time range filter (start, end) in YYYY-MM format
        """
        self.slug = slug
        self.facets = facets
        self.remove_ensembles = remove_ensembles
        self.time_span = time_span

        # Store CMIP7 facets
        self._cmip7_facets = dict(facets)

        # Create corresponding CMIP6 facets
        self._cmip6_facets = self._convert_to_cmip6_facets(facets)

    def _convert_to_cmip6_facets(self, cmip7_facets: dict[str, Any]) -> dict[str, Any]:
        """Convert CMIP7 facets to CMIP6 facets for fetching."""
        cmip6_facets = {}
        for key, value in cmip7_facets.items():
            # Map CMIP7 facet names to CMIP6
            cmip6_key = self.facet_mapping.get(key, key)
            cmip6_facets[cmip6_key] = value
        return cmip6_facets

    def _convert_to_cmip7_metadata(self, cmip6_row: dict[str, Any]) -> dict[str, Any]:
        """Convert CMIP6 metadata to CMIP7 format."""
        cmip7_row = dict(cmip6_row)

        # Map member_id to variant_label
        if "member_id" in cmip7_row:
            cmip7_row["variant_label"] = cmip7_row.pop("member_id")

        # Add CMIP7-specific metadata
        cmip7_row["mip_era"] = "CMIP7"

        # Map table_id to frequency if not present
        if "frequency" not in cmip7_row and "table_id" in cmip7_row:
            table_to_freq = {
                "Amon": "mon",
                "day": "day",
                "fx": "fx",
                "Oyr": "yr",
                "Omon": "mon",
            }
            cmip7_row["frequency"] = table_to_freq.get(cmip7_row["table_id"], "mon")

        return cmip7_row

    def fetch_datasets(self) -> pd.DataFrame:
        """
        Fetch CMIP6 datasets and convert them to CMIP7 format.

        Returns
        -------
        pd.DataFrame
            DataFrame containing CMIP7 dataset metadata and file paths.
        """
        # Create a CMIP6 request with converted facets
        cmip6_request = CMIP6Request(
            slug=f"{self.slug}-cmip6-source",
            facets=self._cmip6_facets,
            remove_ensembles=self.remove_ensembles,
            time_span=self.time_span,
        )

        # Fetch CMIP6 datasets
        cmip6_df = cmip6_request.fetch_datasets()

        if cmip6_df.empty:
            return cmip6_df

        # Convert each file and update metadata
        converted_rows = []
        for _, row in cmip6_df.iterrows():
            row_dict: dict[str, Any] = {str(k): v for k, v in row.to_dict().items()}

            # Get file paths and convert them
            files = row_dict.get("files", [])
            converted_files = []

            # Build CMIP7 facets for this row
            cmip7_facets = self._convert_to_cmip7_metadata(row_dict)

            for file_path in files:
                cmip6_path = Path(file_path)
                if cmip6_path.exists():
                    try:
                        cmip7_path = _convert_file_to_cmip7(cmip6_path, cmip7_facets)
                        converted_files.append(str(cmip7_path))
                    except Exception as e:
                        logger.warning(f"Failed to convert {cmip6_path.name}: {e}")
                        continue
                else:
                    logger.warning(f"CMIP6 file not found: {file_path}")

            if converted_files:
                cmip7_row = self._convert_to_cmip7_metadata(row_dict)
                cmip7_row["files"] = converted_files
                converted_rows.append(cmip7_row)

        if not converted_rows:
            logger.warning(f"No files converted for request: {self.slug}")
            return pd.DataFrame()

        return pd.DataFrame(converted_rows)

    def __repr__(self) -> str:
        return f"CMIP7Request(slug={self.slug!r}, facets={self.facets!r})"
