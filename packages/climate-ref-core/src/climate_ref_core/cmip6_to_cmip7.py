"""
CMIP6 to CMIP7 format converter for testing.

This module provides utilities to convert CMIP6 xarray datasets to CMIP7 format.
This is useful for testing diagnostics with CMIP7-style data before actual CMIP7 data becomes available.

Based on the CMIP7 CMOR tables specification:
https://github.com/WCRP-CMIP/cmip7-cmor-tables
and the example notebook:
https://github.com/WCRP-CMIP/cmip7-cmor-tables/blob/main/Simple_recmorise_cmip6-cmip7.ipynb

Key differences between CMIP6 and CMIP7:
- Variable naming: CMIP7 uses branded names in the DRS like `tas_tavg-h2m-hxy-u` instead of `tas`
- New attributes: mip_era, region, archive_id, host_collection, branding_suffix, etc.
- Directory structure: Different DRS (MIP-DRS7)
- table_id: Uses realm names instead of CMOR table names (atmos vs Amon)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import xarray as xr


# CMIP6 table_id to CMIP7 realm mapping
TABLE_TO_REALM = {
    "Amon": "atmos",
    "Omon": "ocean",
    "Lmon": "land",
    "LImon": "landIce",
    "SImon": "seaIce",
    "AERmon": "aerosol",
    "Oday": "ocean",
    "day": "atmos",
    "Aday": "atmos",
    "Eday": "atmos",
    "CFday": "atmos",
    "3hr": "atmos",
    "6hrLev": "atmos",
    "6hrPlev": "atmos",
    "6hrPlevPt": "atmos",
    "fx": "atmos",  # Fixed fields default to atmos
    "Ofx": "ocean",
    "Efx": "atmos",
    "Lfx": "land",
}

# CMIP6 frequency values (table_id prefix patterns)
FREQUENCY_MAP = {
    "mon": "mon",
    "day": "day",
    "3hr": "3hr",
    "6hr": "6hr",
    "1hr": "1hr",
    "yr": "yr",
    "fx": "fx",
    "subhr": "subhr",
}


@dataclass
class BrandingSuffix:
    """
    CMIP7 branding suffix components.

    Format: <temporal_label>-<vertical_label>-<horizontal_label>-<area_label>
    Example: tavg-h2m-hxy-u
    """

    temporal_label: str = "tavg"  # tavg, tpt, tmax, tmin, tsum, tclm, ti
    vertical_label: str = "u"  # h2m, h10m, u (unspecified), p19, etc.
    horizontal_label: str = "hxy"  # hxy (gridded), hm (mean), hy (zonal), etc.
    area_label: str = "u"  # u (unmasked), lnd, sea, si, etc.

    def __str__(self) -> str:
        return f"{self.temporal_label}-{self.vertical_label}-{self.horizontal_label}-{self.area_label}"


# Common variable to branding suffix mappings
# These are based on typical CMIP6 variable definitions
VARIABLE_BRANDING: dict[str, BrandingSuffix] = {
    # Atmosphere 2D variables
    "tas": BrandingSuffix("tavg", "h2m", "hxy", "u"),
    "tasmax": BrandingSuffix("tmax", "h2m", "hxy", "u"),
    "tasmin": BrandingSuffix("tmin", "h2m", "hxy", "u"),
    "pr": BrandingSuffix("tavg", "u", "hxy", "u"),
    "psl": BrandingSuffix("tavg", "u", "hxy", "u"),
    "ps": BrandingSuffix("tavg", "u", "hxy", "u"),
    "uas": BrandingSuffix("tavg", "h10m", "hxy", "u"),
    "vas": BrandingSuffix("tavg", "h10m", "hxy", "u"),
    "sfcWind": BrandingSuffix("tavg", "h10m", "hxy", "u"),
    "hurs": BrandingSuffix("tavg", "h2m", "hxy", "u"),
    "huss": BrandingSuffix("tavg", "h2m", "hxy", "u"),
    "clt": BrandingSuffix("tavg", "u", "hxy", "u"),
    "rsds": BrandingSuffix("tavg", "u", "hxy", "u"),
    "rsus": BrandingSuffix("tavg", "u", "hxy", "u"),
    "rlds": BrandingSuffix("tavg", "u", "hxy", "u"),
    "rlus": BrandingSuffix("tavg", "u", "hxy", "u"),
    "rsdt": BrandingSuffix("tavg", "u", "hxy", "u"),
    "rsut": BrandingSuffix("tavg", "u", "hxy", "u"),
    "rlut": BrandingSuffix("tavg", "u", "hxy", "u"),
    "evspsbl": BrandingSuffix("tavg", "u", "hxy", "u"),
    "tauu": BrandingSuffix("tavg", "u", "hxy", "u"),
    "tauv": BrandingSuffix("tavg", "u", "hxy", "u"),
    # Ocean 2D variables
    "tos": BrandingSuffix("tavg", "d0m", "hxy", "sea"),
    "sos": BrandingSuffix("tavg", "d0m", "hxy", "sea"),
    "zos": BrandingSuffix("tavg", "u", "hxy", "sea"),
    "mlotst": BrandingSuffix("tavg", "u", "hxy", "sea"),
    # Sea ice variables
    "siconc": BrandingSuffix("tavg", "u", "hxy", "u"),
    "sithick": BrandingSuffix("tavg", "u", "hxy", "si"),
    "sisnthick": BrandingSuffix("tavg", "u", "hxy", "si"),
    # Land variables
    "mrso": BrandingSuffix("tavg", "u", "hxy", "lnd"),
    "mrsos": BrandingSuffix("tavg", "d10cm", "hxy", "lnd"),
    "mrro": BrandingSuffix("tavg", "u", "hxy", "lnd"),
    "snw": BrandingSuffix("tavg", "u", "hxy", "lnd"),
    "lai": BrandingSuffix("tavg", "u", "hxy", "lnd"),
    "gpp": BrandingSuffix("tavg", "u", "hxy", "lnd"),
    "npp": BrandingSuffix("tavg", "u", "hxy", "lnd"),
    "nbp": BrandingSuffix("tavg", "u", "hxy", "lnd"),
    "cVeg": BrandingSuffix("tavg", "u", "hxy", "lnd"),
    "cSoil": BrandingSuffix("tavg", "u", "hxy", "lnd"),
    "treeFrac": BrandingSuffix("tavg", "u", "hxy", "lnd"),
    "vegFrac": BrandingSuffix("tavg", "u", "hxy", "lnd"),
    # Fixed fields
    "areacella": BrandingSuffix("ti", "u", "hxy", "u"),
    "areacello": BrandingSuffix("ti", "u", "hxy", "u"),
    "sftlf": BrandingSuffix("ti", "u", "hxy", "u"),
    "sftof": BrandingSuffix("ti", "u", "hxy", "u"),
    "orog": BrandingSuffix("ti", "u", "hxy", "u"),
}


def get_branding_suffix(variable_id: str, cell_methods: str | None = None) -> BrandingSuffix:
    """
    Determine the CMIP7 branding suffix for a variable.

    Parameters
    ----------
    variable_id
        The CMIP6 variable ID (e.g., "tas", "pr")
    cell_methods
        Optional cell_methods attribute to help determine temporal/spatial operations

    Returns
    -------
    BrandingSuffix
        The branding suffix components
    """
    # Use predefined mapping if available
    if variable_id in VARIABLE_BRANDING:
        return VARIABLE_BRANDING[variable_id]

    # Fallback: infer from variable name patterns
    suffix = BrandingSuffix()

    # Check for max/min in variable name
    if variable_id.endswith("max") or (cell_methods and "maximum" in cell_methods):
        suffix = BrandingSuffix(temporal_label="tmax")
    elif variable_id.endswith("min") or (cell_methods and "minimum" in cell_methods):
        suffix = BrandingSuffix(temporal_label="tmin")

    return suffix


def get_cmip7_variable_name(variable_id: str, branding: BrandingSuffix | None = None) -> str:
    """
    Convert a CMIP6 variable name to CMIP7 branded format.

    Parameters
    ----------
    variable_id
        The CMIP6 variable ID (e.g., "tas")
    branding
        Optional branding suffix; if None, determined automatically

    Returns
    -------
    str
        The CMIP7 variable name (e.g., "tas_tavg-h2m-hxy-u")
    """
    if branding is None:
        branding = get_branding_suffix(variable_id)
    return f"{variable_id}_{branding}"


def get_frequency_from_table(table_id: str) -> str:  # noqa: PLR0911
    """
    Extract frequency from CMIP6 table_id.

    Parameters
    ----------
    table_id
        CMIP6 table identifier (e.g., "Amon", "Oday", "3hr")

    Returns
    -------
    str
        Frequency string (e.g., "mon", "day", "3hr")
    """
    # Check common patterns
    if "mon" in table_id.lower():
        return "mon"
    elif "day" in table_id.lower():
        return "day"
    elif "yr" in table_id.lower():
        return "yr"
    elif "hr" in table_id.lower():
        # Extract hour value
        match = re.search(r"(\d+)hr", table_id.lower())
        if match:
            return f"{match.group(1)}hr"
        return "1hr"
    elif table_id.lower().startswith("fx") or table_id.lower().endswith("fx"):
        return "fx"

    return "mon"  # Default


def get_realm_from_table(table_id: str) -> str:
    """
    Convert CMIP6 table_id to CMIP7 realm.

    Parameters
    ----------
    table_id
        CMIP6 table identifier (e.g., "Amon", "Omon")

    Returns
    -------
    str
        CMIP7 realm (e.g., "atmos", "ocean")
    """
    return TABLE_TO_REALM.get(table_id, "atmos")


@dataclass
class CMIP7Metadata:
    """
    CMIP7 metadata attributes for conversion.

    This captures the additional/modified attributes needed for CMIP7 format.
    """

    # Required new attributes
    mip_era: str = "CMIP7"
    region: str = "GLB"
    archive_id: str = "WCRP"
    host_collection: str = "CMIP7"
    drs_specs: str = "MIP-DRS7"
    cv_version: str = "7.0.0.0"

    # Label attributes (derived from branding_suffix)
    temporal_label: str = "tavg"
    vertical_label: str = "u"
    horizontal_label: str = "hxy"
    area_label: str = "u"

    # Derived attributes
    branding_suffix: str = field(init=False)

    def __post_init__(self) -> None:
        self.branding_suffix = (
            f"{self.temporal_label}-{self.vertical_label}-{self.horizontal_label}-{self.area_label}"
        )

    @classmethod
    def from_branding(cls, branding: BrandingSuffix, **kwargs: Any) -> CMIP7Metadata:
        """Create metadata from a BrandingSuffix."""
        return cls(
            temporal_label=branding.temporal_label,
            vertical_label=branding.vertical_label,
            horizontal_label=branding.horizontal_label,
            area_label=branding.area_label,
            **kwargs,
        )


def convert_cmip6_to_cmip7_attrs(
    cmip6_attrs: dict[str, Any],
    variable_id: str | None = None,
    branding: BrandingSuffix | None = None,
) -> dict[str, Any]:
    """
    Convert CMIP6 global attributes to CMIP7 format.

    Parameters
    ----------
    cmip6_attrs
        Dictionary of CMIP6 global attributes
    variable_id
        Variable ID for determining branding suffix
    branding
        Optional explicit branding suffix

    Returns
    -------
    dict
        Dictionary of CMIP7 global attributes
    """
    # Start with a copy of existing attributes
    attrs = dict(cmip6_attrs)

    # Determine variable_id if not provided
    if variable_id is None:
        variable_id = attrs.get("variable_id", "unknown")

    # Get branding suffix
    if branding is None:
        branding = get_branding_suffix(variable_id, attrs.get("cell_methods"))

    # Create CMIP7 metadata
    cmip7_meta = CMIP7Metadata.from_branding(branding)

    # Update mip_era
    attrs["mip_era"] = cmip7_meta.mip_era
    attrs["parent_mip_era"] = attrs.get("parent_mip_era", "CMIP6")

    # Add new required attributes
    attrs["region"] = cmip7_meta.region
    attrs["archive_id"] = cmip7_meta.archive_id
    attrs["host_collection"] = cmip7_meta.host_collection
    attrs["drs_specs"] = cmip7_meta.drs_specs
    attrs["cv_version"] = cmip7_meta.cv_version

    # Add label attributes
    attrs["temporal_label"] = cmip7_meta.temporal_label
    attrs["vertical_label"] = cmip7_meta.vertical_label
    attrs["horizontal_label"] = cmip7_meta.horizontal_label
    attrs["area_label"] = cmip7_meta.area_label
    attrs["branding_suffix"] = cmip7_meta.branding_suffix

    # Convert table_id to realm-based
    if "table_id" in attrs:
        old_table_id = attrs["table_id"]
        attrs["table_id"] = get_realm_from_table(old_table_id)
        # Also update frequency if not present
        if "frequency" not in attrs:
            attrs["frequency"] = get_frequency_from_table(old_table_id)

    # Update Conventions
    attrs["Conventions"] = "CF-1.12 CMIP-7.0"

    return attrs


def convert_cmip6_dataset(
    ds: xr.Dataset,
    rename_variables: bool = True,
    inplace: bool = False,
) -> xr.Dataset:
    """
    Convert a CMIP6 xarray Dataset to CMIP7 format in-memory.

    This function modifies the dataset attributes and optionally renames
    variables to use CMIP7 branded names.

    Parameters
    ----------
    ds
        The CMIP6 xarray Dataset to convert
    rename_variables
        If True, rename data variables to CMIP7 branded format
    inplace
        If True, modify the dataset in place; otherwise return a copy

    Returns
    -------
    xr.Dataset
        The converted CMIP7-style dataset

    Examples
    --------
    >>> import xarray as xr
    >>> from climate_ref_core.cmip6_to_cmip7 import convert_cmip6_dataset
    >>> ds = xr.open_dataset("tas_Amon_ACCESS-ESM1-5_historical_r1i1p1f1_gn_185001-201412.nc")
    >>> ds_cmip7 = convert_cmip6_dataset(ds)
    >>> print(ds_cmip7.attrs["mip_era"])
    CMIP7
    >>> print(list(ds_cmip7.data_vars))
    ['tas_tavg-h2m-hxy-u']
    """
    if not inplace:
        ds = ds.copy(deep=False)

    # Determine the primary variable (skip coordinates/bounds)
    data_vars = [str(v) for v in ds.data_vars if not str(v).endswith("_bnds") and v not in ds.coords]

    # Convert global attributes
    variable_id = ds.attrs.get("variable_id")
    if variable_id is None and data_vars:
        variable_id = data_vars[0]

    branding = get_branding_suffix(variable_id) if variable_id else None
    ds.attrs = convert_cmip6_to_cmip7_attrs(ds.attrs, variable_id=variable_id, branding=branding)

    # Rename variables if requested
    if rename_variables and variable_id and branding:
        rename_map: dict[str, str] = {}
        for var in data_vars:
            var_branding = get_branding_suffix(var)
            new_name = get_cmip7_variable_name(var, var_branding)
            if var != new_name:
                rename_map[var] = new_name
        if rename_map:
            ds = ds.rename(rename_map)

    return ds


def create_cmip7_instance_id(attrs: dict[str, Any]) -> str:
    """
    Create a CMIP7 instance_id from attributes.

    The CMIP7 instance_id follows the MIP-DRS7 specification:
    <mip_era>.<activity_id>.<institution_id>.<source_id>.<experiment_id>.
    <variant_label>.<region>.<frequency>.<variable_id>.<branding_suffix>.<grid_label>.<version>

    Parameters
    ----------
    attrs
        Dictionary containing CMIP7 attributes

    Returns
    -------
    str
        The CMIP7 instance_id
    """
    components = [
        attrs.get("mip_era", "CMIP7"),
        attrs.get("activity_id", "CMIP"),
        attrs.get("institution_id", ""),
        attrs.get("source_id", ""),
        attrs.get("experiment_id", ""),
        attrs.get("variant_label", ""),
        attrs.get("region", "GLB"),
        attrs.get("frequency", "mon"),
        attrs.get("variable_id", ""),
        attrs.get("branding_suffix", ""),
        attrs.get("grid_label", "gn"),
        attrs.get("version", "v1"),
    ]
    return ".".join(str(c) for c in components)


def create_cmip7_path(attrs: dict[str, Any]) -> str:
    """
    Create a CMIP7 directory path from attributes.

    The CMIP7 path follows the MIP-DRS7 specification:
    <drs_specs>/<mip_era>/<activity_id>/<institution_id>/<source_id>/<experiment_id>/
    <variant_label>/<region>/<frequency>/<variable_id>/<branding_suffix>/<grid_label>/<version>

    Parameters
    ----------
    attrs
        Dictionary containing CMIP7 attributes

    Returns
    -------
    str
        The CMIP7 directory path
    """
    components = [
        attrs.get("drs_specs", "MIP-DRS7"),
        attrs.get("mip_era", "CMIP7"),
        attrs.get("activity_id", "CMIP"),
        attrs.get("institution_id", ""),
        attrs.get("source_id", ""),
        attrs.get("experiment_id", ""),
        attrs.get("variant_label", ""),
        attrs.get("region", "GLB"),
        attrs.get("frequency", "mon"),
        attrs.get("variable_id", ""),
        attrs.get("branding_suffix", ""),
        attrs.get("grid_label", "gn"),
        attrs.get("version", "v1"),
    ]
    return "/".join(str(c) for c in components)
