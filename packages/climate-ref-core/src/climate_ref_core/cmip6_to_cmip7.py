"""
CMIP6 to CMIP7 format converter.

This module provides utilities to convert CMIP6 xarray datasets to CMIP7 format,
following the CMIP7 Global Attributes V1.0 specification (DOI: 10.5281/zenodo.17250297).


Key differences between CMIP6 and CMIP7
---------------------------------------
- Variable naming: CMIP7 uses branded names like `tas_tavg-h2m-hxy-u` instead of `tas`
- Branding suffix: `<temporal>-<vertical>-<horizontal>-<area>` labels (e.g., `tavg-h2m-hxy-u`)
- Variant indices: Changed from integers to prefixed strings (1 -> "r1", "i1", "p1", "f1")
- New mandatory attributes: license_id
- table_id: Uses realm names instead of CMOR table names (atmos vs Amon)
- Directory structure: MIP-DRS7 specification
- Filename format: Includes branding suffix, region, and grid_label
- Removed CMIP6 attributes: further_info_url, grid, member_id, sub_experiment, sub_experiment_id

References
----------
- CMIP7 Global Attributes V1.0: https://doi.org/10.5281/zenodo.17250297
- CMIP7 CVs: https://github.com/WCRP-CMIP/CMIP7_CVs
- CMIP7 Guidance: https://wcrp-cmip.github.io/cmip7-guidance/
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
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
}

# CMIP6-only attributes that should be removed when converting to CMIP7
# These are not part of the CMIP7 Global Attributes specification (V1.0)
# These may be included in output, but they won't be checked
CMIP6_ONLY_ATTRIBUTES = {
    "further_info_url",  # CMIP6-specific URL format, replaced by different mechanism in CMIP7
    "grid",  # Replaced by grid_label in CMIP7
    "member_id",  # Redundant with variant_label, not in CMIP7 spec
    "sub_experiment",  # Not in CMIP7 spec
    "sub_experiment_id",  # Not in CMIP7 spec
    "table_id",  # Not in CMIP7 spec
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


def convert_variant_index(value: int | str, prefix: str) -> str:
    """
    Convert CMIP6 numeric variant index to CMIP7 string format.

    In CMIP6, indices like realization_index were integers (e.g., 1).
    In CMIP7, they are strings with a prefix (e.g., "r1").

    Parameters
    ----------
    value
        The index value (int or str)
    prefix
        The prefix to use ("r", "i", "p", or "f")

    Returns
    -------
    str
        The CMIP7 format index (e.g., "r1", "i1", "p1", "f1")
    """
    if isinstance(value, int):
        return f"{prefix}{value}"
    elif isinstance(value, str):
        # Already has prefix
        if value.startswith(prefix):
            return value
        # Try to extract numeric part
        try:
            return f"{prefix}{int(value)}"
        except ValueError:
            return f"{prefix}{value}"

    return f"{prefix}1"  # type: ignore


@dataclass
class CMIP7Metadata:
    """
    CMIP7 metadata attributes for conversion.

    This captures the additional/modified attributes needed for CMIP7 format.
    Based on CMIP7 Global Attributes V1.0 (DOI: 10.5281/zenodo.17250297).
    """

    # Required new attributes
    mip_era: str = "CMIP7"
    region: str = "glb"
    drs_specs: str = "MIP-DRS7"
    data_specs_version: str = "MIP-DS7.1.0.0"
    product: str = "model-output"
    license_id: str = "CC-BY-4.0"

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

    Based on CMIP7 Global Attributes V1.0 (DOI: 10.5281/zenodo.17250297).

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

    # New/updated CMIP7 attributes
    attrs["region"] = cmip7_meta.region
    attrs["drs_specs"] = cmip7_meta.drs_specs
    attrs["data_specs_version"] = cmip7_meta.data_specs_version
    attrs["product"] = cmip7_meta.product
    attrs["license_id"] = cmip7_meta.license_id

    # Add tracking_id with CMIP7 handle prefix
    attrs["tracking_id"] = f"hdl:21.14107/{uuid.uuid4()}"

    # Add creation_date in ISO format
    attrs["creation_date"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Add label attributes
    attrs["temporal_label"] = cmip7_meta.temporal_label
    attrs["vertical_label"] = cmip7_meta.vertical_label
    attrs["horizontal_label"] = cmip7_meta.horizontal_label
    attrs["area_label"] = cmip7_meta.area_label
    attrs["branding_suffix"] = cmip7_meta.branding_suffix

    # Add branded_variable (required in CMIP7)
    attrs["branded_variable"] = f"{variable_id}_{cmip7_meta.branding_suffix}"

    # Convert variant indices from CMIP6 integer to CMIP7 string format
    if "realization_index" in attrs:
        attrs["realization_index"] = convert_variant_index(attrs["realization_index"], "r")
    if "initialization_index" in attrs:
        attrs["initialization_index"] = convert_variant_index(attrs["initialization_index"], "i")
    if "physics_index" in attrs:
        attrs["physics_index"] = convert_variant_index(attrs["physics_index"], "p")
    if "forcing_index" in attrs:
        attrs["forcing_index"] = convert_variant_index(attrs["forcing_index"], "f")

    # Rebuild variant_label from converted indices
    r = attrs.get("realization_index", "r1")
    i = attrs.get("initialization_index", "i1")
    p = attrs.get("physics_index", "p1")
    f = attrs.get("forcing_index", "f1")
    attrs["variant_label"] = f"{r}{i}{p}{f}"

    # Convert table_id to realm-based and set realm attribute
    if "table_id" in attrs:
        old_table_id = attrs["table_id"]
        realm = get_realm_from_table(old_table_id)
        attrs["realm"] = realm
        # Also update frequency if not present
        if "frequency" not in attrs:
            attrs["frequency"] = get_frequency_from_table(old_table_id)
        # Store legacy CMIP6 compound name for reference (optional but recommended)
        attrs["cmip6_compound_name"] = f"{old_table_id}.{variable_id}"

    # Update Conventions (CF version only, per CMIP7 spec)
    attrs["Conventions"] = "CF-1.12"

    # Remove CMIP6-only attributes that are not in CMIP7 spec
    for attr in CMIP6_ONLY_ATTRIBUTES:
        attrs.pop(attr, None)

    return attrs


def convert_cmip6_dataset(
    ds: xr.Dataset,
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
    inplace
        If True, modify the dataset in place; otherwise return a copy

    Returns
    -------
    xr.Dataset
        The converted CMIP7-style dataset
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

    return ds


def create_cmip7_filename(
    attrs: dict[str, Any],
    time_range: str | None = None,
) -> str:
    """
    Create a CMIP7 filename from attributes.

    The CMIP7 filename follows the MIP-DRS7 specification (V1.0):
    <variable_id>_<branding_suffix>_<frequency>_<region>_<grid_label>_<source_id>_<experiment_id>_<variant_label>[_<timeRangeDD>].nc

    Parameters
    ----------
    attrs
        Dictionary containing CMIP7 attributes
    time_range
        Optional time range string (e.g., "190001-190912").
        Format depends on frequency: "YYYY" for yearly, "YYYYMM" for monthly, "YYYYMMDD" for daily.
        Omit for fixed/time-independent variables.

    Returns
    -------
    str
        The CMIP7 filename

    Examples
    --------
    >>> attrs = {
    ...     "variable_id": "tas",
    ...     "branding_suffix": "tavg-h2m-hxy-u",
    ...     "frequency": "mon",
    ...     "region": "glb",
    ...     "grid_label": "g13s",
    ...     "source_id": "CanESM6-MR",
    ...     "experiment_id": "historical",
    ...     "variant_label": "r2i1p1f1",
    ... }
    >>> create_cmip7_filename(attrs, "190001-190912")
    'tas_tavg-h2m-hxy-u_mon_glb_g13s_CanESM6-MR_historical_r2i1p1f1_190001-190912.nc'
    """
    components = [
        attrs.get("variable_id", ""),
        attrs.get("branding_suffix", ""),
        attrs.get("frequency", "mon"),
        attrs.get("region", "glb"),
        attrs.get("grid_label", "gn"),
        attrs.get("source_id", ""),
        attrs.get("experiment_id", ""),
        attrs.get("variant_label", ""),
    ]

    filename = "_".join(str(c) for c in components)

    # Add time range if provided (omit for fixed/time-independent variables)
    if time_range:
        filename = f"{filename}_{time_range}"

    return f"{filename}.nc"


def create_cmip7_path(attrs: dict[str, Any], version: str | None = None) -> str:
    """
    Create a CMIP7 directory path from attributes.

    The CMIP7 path follows the MIP-DRS7 specification (V1.0):
    <drs_specs>/<mip_era>/<activity_id>/<institution_id>/<source_id>/<experiment_id>/
    <variant_label>/<region>/<frequency>/<variable_id>/<branding_suffix>/<grid_label>/<version>

    Parameters
    ----------
    attrs
        Dictionary containing CMIP7 attributes
    version
        Optional version string (e.g., "v20250622"). If not provided, uses attrs["version"]
        or defaults to "v1".

    Returns
    -------
    str
        The CMIP7 directory path

    Examples
    --------
    >>> attrs = {
    ...     "drs_specs": "MIP-DRS7",
    ...     "mip_era": "CMIP7",
    ...     "activity_id": "CMIP",
    ...     "institution_id": "CCCma",
    ...     "source_id": "CanESM6-MR",
    ...     "experiment_id": "historical",
    ...     "variant_label": "r2i1p1f1",
    ...     "region": "glb",
    ...     "frequency": "mon",
    ...     "variable_id": "tas",
    ...     "branding_suffix": "tavg-h2m-hxy-u",
    ...     "grid_label": "g13s",
    ... }
    >>> create_cmip7_path(attrs, "v20250622")
    'MIP-DRS7/CMIP7/CMIP/CCCma/CanESM6-MR/historical/r2i1p1f1/glb/mon/tas/tavg-h2m-hxy-u/g13s/v20250622'
    """
    version_str = version or attrs.get("version", "v1")

    components = [
        attrs.get("drs_specs", "MIP-DRS7"),
        attrs.get("mip_era", "CMIP7"),
        attrs.get("activity_id", "CMIP"),
        attrs.get("institution_id", ""),
        attrs.get("source_id", ""),
        attrs.get("experiment_id", ""),
        attrs.get("variant_label", ""),
        attrs.get("region", "glb"),
        attrs.get("frequency", "mon"),
        attrs.get("variable_id", ""),
        attrs.get("branding_suffix", ""),
        attrs.get("grid_label", "gn"),
        version_str,
    ]
    return "/".join(str(c) for c in components)
