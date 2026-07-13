"""
CMIP7 dataset request implementation.

Since CMIP7 data is not yet available on ESGF, this module provides
a request class that fetches CMIP6 data and converts it to CMIP7 format.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import pandas as pd
import xarray as xr
from loguru import logger

from climate_ref_core.cmip6_to_cmip7 import (
    convert_cmip6_dataset,
    create_cmip7_filename,
    create_cmip7_path,
    format_cmip7_time_range,
    get_dreq_entry,
    get_frequency_from_table,
    shift_time_axis_end,
    suppress_bounds_coordinates,
)
from climate_ref_core.dataset_registry import resolve_cache_dir
from climate_ref_core.esgf.cmip6 import CMIP6Request


def _get_cmip7_cache_dir() -> Path:
    """Get the cache directory for converted CMIP7 files."""
    # Kept out of the intake-esgf cache so converted files don't mix with downloads.
    cache_dir = resolve_cache_dir("cmip7-converted")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def _convert_file_to_cmip7(
    cmip6_path: Path,
    cmip7_facets: dict[str, Any],
    extend_historical_to: tuple[int, int] | None = None,
) -> Path:
    """
    Convert a CMIP6 file to CMIP7 format.

    Parameters
    ----------
    cmip6_path
        Path to the CMIP6 NetCDF file
    cmip7_facets
        CMIP7 facets for the output path
    extend_historical_to
        Opt-in ``(end_year, end_month)`` passed to :func:`convert_cmip6_dataset`
        to relabel the time axis so historical coverage reaches that month.
        Defaults to ``None`` (time axis untouched).

    Returns
    -------
    Path
        Path to the converted CMIP7 file
    """
    cache_dir = _get_cmip7_cache_dir()

    # CMIP6 activity_id can contain multiple activities separated by spaces
    # (e.g. "C4MIP CDRMIP"). Use only the first activity for the DRS path.
    activity_id = str(cmip7_facets.get("activity_id", "CMIP")).split()[0]
    version = str(cmip7_facets.get("version", "v1"))

    # Build CMIP7 DRS path using the standard MIP-DRS7 path builder.
    # Provide defaults for fields that may not be in facets.
    path_facets = {
        "drs_specs": "MIP-DRS7",
        "mip_era": "CMIP7",
        "institution_id": "unknown",
        **cmip7_facets,
        "activity_id": activity_id,
    }
    drs_path = cache_dir / create_cmip7_path(path_facets, version)

    drs_path.mkdir(parents=True, exist_ok=True)

    # Convert the file and derive the time range from the dataset
    logger.info(f"Converting to CMIP7: {cmip6_path.name}")

    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    with xr.open_dataset(cmip6_path, decode_times=time_coder) as ds:
        # When fabricating extended historical coverage, relabel the time axis first
        # so both the filename time range and the written data reflect the new dates.
        source_ds = ds
        if extend_historical_to is not None:
            end_year, end_month = extend_historical_to
            source_ds = shift_time_axis_end(ds, end_year=end_year, end_month=end_month)

        frequency = str(cmip7_facets.get("frequency", "mon"))
        time_range = format_cmip7_time_range(source_ds, frequency)
        output_file = drs_path / create_cmip7_filename(cmip7_facets, time_range=time_range)

        if output_file.exists():
            logger.debug(f"Using cached CMIP7 file: {output_file}")
            return output_file

        ds_cmip7 = convert_cmip6_dataset(source_ds)

        # Ensure version and sanitized activity_id are in the file attributes
        # so that parse_cmip7_file can extract them for instance_id construction
        ds_cmip7.attrs["version"] = version
        ds_cmip7.attrs["activity_id"] = activity_id

        try:
            logger.info(f"Writing translated CMIP7 file: {output_file}")
            suppress_bounds_coordinates(ds_cmip7)

            # Apply lossy compression - these are converted files used for
            # verification only, so precision loss is acceptable.
            # gpp magnitudes (~1e-8 kg m-2 s-1) sit far below the fixed
            # least_significant_digit=3 (~1e-3) precision floor, which would round the entire field to zero.
            # Keep gpp lossless; other variables tolerate it.
            encoding: dict[str, dict[str, Any]] = {}
            for var in ds_cmip7.data_vars:
                var_encoding: dict[str, Any] = {"zlib": True, "complevel": 5}

                lossless_variables = {"gpp"}
                if str(var) not in lossless_variables:
                    var_encoding["least_significant_digit"] = 3
                encoding[str(var)] = var_encoding

            ds_cmip7.to_netcdf(output_file, encoding=encoding)
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

    # CMIP7-only facets that should not be passed to CMIP6 ESGF searches
    cmip7_only_facets: ClassVar[set[str]] = {
        "branded_variable",
        "region",
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
        "region",
        "branded_variable",
    )

    def __init__(
        self,
        slug: str,
        facets: dict[str, Any],
        remove_ensembles: bool = False,
        time_span: tuple[str, str] | None = None,
        extend_historical_to: tuple[int, int] | None = None,
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
        extend_historical_to
            Opt-in ``(end_year, end_month)``. When set, each converted CMIP7 file has
            its time axis relabelled so historical coverage ends on that month, letting
            us fabricate CMIP7 data for years without real CMIP6 source data (e.g. the
            fire diagnostic's 2002-2021 window). Defaults to ``None`` (time axis
            untouched), so other CMIP7 conversions are unchanged.
        """
        self.slug = slug
        self.facets = facets
        self.remove_ensembles = remove_ensembles
        self.time_span = time_span
        self.extend_historical_to = extend_historical_to

        # Create corresponding CMIP6 facets
        self._cmip6_facets = self._convert_to_cmip6_facets(facets)

    def _convert_to_cmip6_facets(self, cmip7_facets: dict[str, Any]) -> dict[str, Any]:
        """Convert CMIP7 facets to CMIP6 facets for fetching."""
        cmip6_facets = {}
        for key, value in cmip7_facets.items():
            # Skip CMIP7-only facets that don't exist in CMIP6 ESGF
            if key in self.cmip7_only_facets:
                continue
            # Map CMIP7 facet names to CMIP6
            cmip6_key = self.facet_mapping.get(key, key)
            cmip6_facets[cmip6_key] = value
        return cmip6_facets

    def _convert_to_cmip7_metadata(self, cmip6_row: dict[str, Any]) -> dict[str, Any]:
        """Convert a subset of CMIP6 metadata to CMIP7 format.

        This is the single location for DReq enrichment: it updates
        ``variable_id`` and adds ``region``, ``branding_suffix``, and
        ``branded_variable`` from the Data Request when available.
        """
        cmip7_row = dict(cmip6_row)

        # Map member_id to variant_label
        if "member_id" in cmip7_row:
            cmip7_row["variant_label"] = cmip7_row.pop("member_id")

        # Add CMIP7-specific metadata
        cmip7_row["mip_era"] = "CMIP7"

        # CMIP6 activity_id can contain multiple activities separated by spaces
        # (e.g. "C4MIP CDRMIP"). Use only the first activity for CMIP7.
        if "activity_id" in cmip7_row and " " in str(cmip7_row["activity_id"]):
            cmip7_row["activity_id"] = str(cmip7_row["activity_id"]).split()[0]

        # Map table_id to frequency if not present
        if "frequency" not in cmip7_row and "table_id" in cmip7_row:
            cmip7_row["frequency"] = get_frequency_from_table(cmip7_row["table_id"])

        # Enrich with DReq metadata
        table_id = cmip7_row.get("table_id")
        variable_id = cmip7_row.get("variable_id")
        if table_id and variable_id:
            try:
                entry = get_dreq_entry(table_id, variable_id)
                cmip7_row["region"] = entry.region
                cmip7_row["variable_id"] = entry.variable_id
                cmip7_row["branding_suffix"] = entry.branding_suffix
                cmip7_row["branded_variable"] = entry.branded_variable
            except KeyError:
                logger.error(
                    f"No DReq entry for {table_id}.{variable_id}, region/branding_suffix will not be set"
                )

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
        # The returned DataFrame does not need to have the complete set of CMIP7 metadata
        # Datasets will be re-read during DB ingestion during the tests
        converted_rows = []
        for _, row in cmip6_df.iterrows():
            row_dict: dict[str, Any] = {str(k): v for k, v in row.to_dict().items()}

            # Build CMIP7 facets for this row (includes DReq enrichment)
            cmip7_row = self._convert_to_cmip7_metadata(row_dict)

            # Get file paths and convert them
            files = row_dict.get("files", [])
            converted_files = []
            for file_path in files:
                cmip6_path = Path(file_path)
                if cmip6_path.exists():
                    try:
                        cmip7_path = _convert_file_to_cmip7(
                            cmip6_path, cmip7_row, extend_historical_to=self.extend_historical_to
                        )
                        converted_files.append(str(cmip7_path))
                    except Exception as e:
                        logger.exception(f"Failed to convert {cmip6_path.name}: {e}")
                        continue
                else:
                    logger.warning(f"CMIP6 file not found: {file_path}")

            if converted_files:
                cmip7_row["files"] = converted_files
                converted_rows.append(cmip7_row)

        if not converted_rows:
            logger.warning(f"No files converted for request: {self.slug}")
            return pd.DataFrame()

        return pd.DataFrame(converted_rows)

    def __repr__(self) -> str:
        return f"CMIP7Request(slug={self.slug!r}, facets={self.facets!r})"
