"""
CMIP7 dataset request implementation.

This module provides a CMIP7Request class that wraps CMIP6 data from ESGF
and converts it to CMIP7 format for testing diagnostics with CMIP7-style data
before actual CMIP7 data becomes available on ESGF.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import xarray as xr
from loguru import logger

from climate_ref_core.cmip6_to_cmip7 import (
    convert_cmip6_dataset,
    create_cmip7_instance_id,
    create_cmip7_path,
    get_branding_suffix,
    get_frequency_from_table,
    get_realm_from_table,
)
from climate_ref_core.esgf.cmip6 import CMIP6Request

if TYPE_CHECKING:
    import xarray as xr


def _convert_file(
    cmip6_path: Path,
    output_dir: Path,
    rename_variables: bool = False,
) -> tuple[Path, dict[str, Any]] | None:
    """
    Convert a single CMIP6 file to CMIP7 format.

    Parameters
    ----------
    cmip6_path
        Path to the CMIP6 file
    output_dir
        Root directory for converted CMIP7 files
    rename_variables
        If True, rename variables to CMIP7 branded format

    Returns
    -------
    tuple[Path, dict[str, Any]] | None
        Tuple of (cmip7_path, cmip7_attrs) or None if conversion failed
    """
    try:
        ds: xr.Dataset = xr.open_dataset(cmip6_path)
    except Exception as e:
        logger.warning(f"Failed to open {cmip6_path}: {e}")
        return None

    try:
        ds_cmip7 = convert_cmip6_dataset(ds, rename_variables=rename_variables)

        # Build output path using CMIP7 DRS
        cmip7_subpath = create_cmip7_path(ds_cmip7.attrs)
        cmip7_dir = output_dir / cmip7_subpath
        cmip7_dir.mkdir(parents=True, exist_ok=True)

        # Use original filename
        cmip7_path = cmip7_dir / cmip6_path.name

        # Only write if file doesn't already exist
        if not cmip7_path.exists():
            ds_cmip7.to_netcdf(cmip7_path)
            logger.debug(f"Wrote CMIP7 file: {cmip7_path}")
        else:
            logger.debug(f"CMIP7 file already exists: {cmip7_path}")

        return cmip7_path, dict(ds_cmip7.attrs)

    except Exception as e:
        logger.warning(f"Failed to convert {cmip6_path}: {e}")
        return None
    finally:
        ds.close()


class CMIP7Request:
    """
    CMIP7 dataset request that wraps CMIP6 data with conversion.

    Fetches CMIP6 data from ESGF and converts to CMIP7 format.
    Converted files are written to disk with CMIP7 directory structure.

    Parameters
    ----------
    slug
        Unique identifier for this request
    facets
        ESGF search facets (e.g., source_id, variable_id, experiment_id)
    output_dir
        Directory to write converted CMIP7 files.
        Defaults to ~/.cache/climate-ref/cmip7
    remove_ensembles
        If True, keep only one ensemble member per model
    time_span
        Optional time range filter (start, end) in YYYY-MM format
    rename_variables
        If True, rename variables to CMIP7 branded format (e.g., tas -> tas_tavg-h2m-hxy-u)

    Examples
    --------
    >>> from climate_ref_core.esgf import CMIP7Request
    >>> request = CMIP7Request(
    ...     slug="tas-historical",
    ...     facets={
    ...         "source_id": "ACCESS-ESM1-5",
    ...         "experiment_id": "historical",
    ...         "variable_id": "tas",
    ...         "member_id": "r1i1p1f1",
    ...         "table_id": "Amon",
    ...     },
    ...     time_span=("2000-01", "2014-12"),
    ... )
    >>> df = request.fetch_datasets()
    """

    source_type = "CMIP7"

    def __init__(  # noqa: PLR0913
        self,
        slug: str,
        facets: dict[str, Any],
        output_dir: Path | None = None,
        remove_ensembles: bool = False,
        time_span: tuple[str, str] | None = None,
        rename_variables: bool = False,
    ):
        self.slug = slug
        self.facets = facets
        self.output_dir = output_dir or Path.home() / ".cache" / "climate-ref" / "cmip7"
        self.rename_variables = rename_variables
        self.time_span = time_span

        # Internal CMIP6 request for actual fetching
        self._cmip6_request = CMIP6Request(
            slug=f"{slug}_cmip6_source",
            facets=facets,
            remove_ensembles=remove_ensembles,
            time_span=time_span,
        )

    def __repr__(self) -> str:
        return f"CMIP7Request(slug={self.slug!r}, facets={self.facets!r})"

    def fetch_datasets(self) -> pd.DataFrame:
        """
        Fetch CMIP6 data from ESGF and convert to CMIP7 format.

        Returns
        -------
        pd.DataFrame
            DataFrame containing CMIP7 metadata and file paths.
            Contains columns: key, files, path, source_type, and all CMIP7 metadata fields.
        """
        # Fetch CMIP6 data
        cmip6_df = self._cmip6_request.fetch_datasets()

        if cmip6_df.empty:
            logger.warning(f"No CMIP6 datasets found for {self.slug}")
            return cmip6_df

        converted_rows = []

        for _, row in cmip6_df.iterrows():
            files_list = row.get("files", [])
            if not files_list:
                continue

            cmip7_files = []
            cmip7_attrs: dict[str, Any] | None = None

            for _file_path in files_list:
                # Convert each CMIP6 file to CMIP7-like
                file_path = Path(_file_path)
                if not file_path.exists():
                    logger.warning(f"CMIP6 file does not exist: {file_path}")
                    continue

                result = _convert_file(file_path, self.output_dir, self.rename_variables)
                if result is not None:
                    cmip7_path, attrs = result
                    cmip7_files.append(str(cmip7_path))
                    # Use attributes from first successful conversion
                    if cmip7_attrs is None:
                        cmip7_attrs = attrs

            if cmip7_files and cmip7_attrs is not None:
                converted_row = self._build_cmip7_metadata(row, cmip7_files, cmip7_attrs)
                converted_rows.append(converted_row)

        if not converted_rows:
            logger.warning(f"No CMIP7 files converted for {self.slug}")
            return pd.DataFrame()

        return pd.DataFrame(converted_rows)

    def _build_cmip7_metadata(
        self,
        cmip6_row: pd.Series,
        cmip7_files: list[str],
        cmip7_attrs: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Build CMIP7 metadata row from CMIP6 source and converted attributes.

        Parameters
        ----------
        cmip6_row
            Original CMIP6 metadata row
        cmip7_files
            List of converted CMIP7 file paths
        cmip7_attrs
            Attributes from the converted CMIP7 dataset

        Returns
        -------
        dict[str, Any]
            CMIP7 metadata dictionary
        """
        table_id = cmip6_row.get("table_id", "Amon")
        variable_id = cmip6_row.get("variable_id", "tas")
        branding_suffix = get_branding_suffix(variable_id)

        # Start with CMIP6 row as base
        result: dict[str, Any] = cmip6_row.to_dict()

        # Override with CMIP7-specific values
        result.update(
            {
                "path": cmip7_files[0],  # Primary path
                "files": cmip7_files,
                "mip_era": "CMIP7",
                "table_id": get_realm_from_table(table_id),
                "frequency": get_frequency_from_table(table_id),
                "region": cmip7_attrs.get("region", "GLB"),
                "branding_suffix": str(branding_suffix),
                "archive_id": cmip7_attrs.get("archive_id", "WCRP"),
                "host_collection": cmip7_attrs.get("host_collection", "CMIP7"),
                "drs_specs": cmip7_attrs.get("drs_specs", "MIP-DRS7"),
                "cv_version": cmip7_attrs.get("cv_version", "7.0.0.0"),
                "temporal_label": cmip7_attrs.get("temporal_label", branding_suffix.temporal_label),
                "vertical_label": cmip7_attrs.get("vertical_label", branding_suffix.vertical_label),
                "horizontal_label": cmip7_attrs.get("horizontal_label", branding_suffix.horizontal_label),
                "area_label": cmip7_attrs.get("area_label", branding_suffix.area_label),
            }
        )

        # Generate CMIP7 instance_id
        result["key"] = create_cmip7_instance_id(result)

        return result
