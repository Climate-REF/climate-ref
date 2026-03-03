"""
CMIP6 to CMIP7 format converter.

This module provides utilities to convert CMIP6 xarray datasets to CMIP7 format,
following the CMIP7 Global Attributes V1.0 specification (DOI: 10.5281/zenodo.17250297).

Variable branding mappings are sourced from the CMIP7 Data Request (DReq v1.2.2.3) via a bundled JSON subset.
Regenerate with:

    python scripts/extract-data-request-mappings.py

The variable_id and table_id attributes are used to map the CMIP6 variable to its CMIP7 equivalent,
and to determine the appropriate branding suffix and realm for the CMIP7 format.
Grid information is not converted so the grid_labels may not be valid for CMIP7.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from importlib import resources
from typing import TYPE_CHECKING, Any

import attrs
import cftime
import pandas as pd

if TYPE_CHECKING:
    import xarray as xr


def suppress_bounds_coordinates(ds: xr.Dataset) -> xr.Dataset:
    """
    Prevent xarray from adding spurious ``coordinates`` attributes to bounds variables.

    When a dataset contains scalar coordinates (e.g. ``height``),
    xarray's CF encoder adds a ``coordinates`` attribute to *every* variable,
    including bounds variables like ``time_bnds``.
    Call this on a dataset before :meth:`xarray.Dataset.to_netcdf`.

    Parameters
    ----------
    ds
        The dataset about to be written.  Modified in place.

    Returns
    -------
    xr.Dataset
        The same dataset, for chaining.
    """
    for name in ds:
        if str(name).endswith("_bnds"):
            # Setting encoding["coordinates"] to None tells xarray's CF encoder
            # to skip auto-adding a coordinates attribute for this variable.
            ds[name].encoding["coordinates"] = None
    return ds


@attrs.frozen
class DReqVariableMapping:
    """
    A single CMIP6-to-CMIP7 variable mapping from the Data Request.

    Each instance represents one row in the DReq Variables JSON file,
    capturing the CMIP6 compound name, its CMIP7 equivalent, and
    the branding/realm metadata needed for format conversion.
    """

    table_id: str
    variable_id: str
    cmip6_compound_name: str
    cmip7_compound_name: str
    branded_variable: str
    branding_suffix: str
    temporal_label: str
    vertical_label: str
    horizontal_label: str
    area_label: str
    realm: str
    region: str
    frequency: str

    def to_dict(self) -> dict[str, str]:
        """Serialise to a plain dict (for JSON output)."""
        return attrs.asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DReqVariableMapping:
        """Deserialise from a plain dict (e.g. loaded from JSON)."""
        return cls(**{a.name: data[a.name] for a in attrs.fields(cls)})


def _load_dreq_mappings() -> dict[str, DReqVariableMapping]:
    """
    Load CMIP6-to-CMIP7 variable mappings from bundled DReq JSON.

    Returns
    -------
    dict
        Mapping from CMIP6 compound name (e.g. ``"Amon.tas"``) to
        :class:`DReqVariableMapping`.
    """
    data_files = resources.files("climate_ref_core") / "data" / "cmip6_cmip7_variable_map.json"
    raw: dict[str, Any] = json.loads(data_files.read_text(encoding="utf-8"))
    variables: dict[str, dict[str, Any]] = raw.get("variables", {})
    return {key: DReqVariableMapping.from_dict(entry) for key, entry in variables.items()}


_DREQ_VARIABLES: dict[str, DReqVariableMapping] = {}


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

    temporal_label: str = "tavg"  # tavg, tpt, tmax, tmin, tsum, tclm, ti, tmaxavg, tminavg
    vertical_label: str = "u"  # h2m, h10m, u (unspecified), p19, al, etc.
    horizontal_label: str = "hxy"  # hxy (gridded), hm (mean), hy (zonal), etc.
    area_label: str = "u"  # u (unmasked), lnd, sea, si, air, etc.

    def __str__(self) -> str:
        return f"{self.temporal_label}-{self.vertical_label}-{self.horizontal_label}-{self.area_label}"


def get_dreq_entry(table_id: str, variable_id: str) -> DReqVariableMapping:
    """
    Look up a variable in the Data Request by compound name.

    Parameters
    ----------
    table_id
        CMIP6 table identifier (e.g., "Amon")
    variable_id
        CMIP6 variable ID (e.g., "tas")

    Returns
    -------
    DReqVariableMapping
        The DReq variable entry.

    Raises
    ------
    KeyError
        If the compound name is not found in the Data Request mappings.
    """
    cmip6_compound = f"{table_id}.{variable_id}"

    # Lazy load the DReq mappings on first access
    if len(_DREQ_VARIABLES) == 0:
        _DREQ_VARIABLES.update(_load_dreq_mappings())

    entry = _DREQ_VARIABLES.get(cmip6_compound)
    if entry is None:
        raise KeyError(
            f"Variable '{cmip6_compound}' not found in Data Request mappings. "
            f"Add it to INCLUDED_VARIABLES in scripts/extract-data-request-mappings.py and regenerate."
        )
    return entry


def get_branding_suffix(table_id: str, variable_id: str) -> BrandingSuffix:
    """
    Determine the CMIP7 branding suffix for a variable.

    Parameters
    ----------
    table_id
        CMIP6 table ID (e.g., "Amon", "Omon")
    variable_id
        CMIP6 variable ID (e.g., "tas", "pr")

    Returns
    -------
    BrandingSuffix
        The branding suffix components

    Raises
    ------
    KeyError
        If the variable is not found in the Data Request mappings.
    """
    entry = get_dreq_entry(table_id, variable_id)
    return BrandingSuffix(
        temporal_label=entry.temporal_label,
        vertical_label=entry.vertical_label,
        horizontal_label=entry.horizontal_label,
        area_label=entry.area_label,
    )


def get_cmip7_compound_name(table_id: str, variable_id: str) -> str:
    """
    Get the full CMIP7 compound name for a CMIP6 variable.

    The CMIP7 compound name has the format:
    ``<realm>.<variable_id>.<branding_suffix>.<frequency>.<region>``

    Parameters
    ----------
    table_id
        CMIP6 table identifier (e.g., "Amon")
    variable_id
        CMIP6 variable ID (e.g., "tas")

    Returns
    -------
    str
        The CMIP7 compound name.

    Raises
    ------
    KeyError
        If the variable is not found in the Data Request mappings.
    """
    entry = get_dreq_entry(table_id, variable_id)
    return entry.cmip7_compound_name


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


def get_realm(table_id: str, variable_id: str) -> str:
    """
    Get the CMIP7 realm for a CMIP6 variable.

    Parameters
    ----------
    table_id
        CMIP6 table identifier (e.g., "Amon", "Omon")
    variable_id
        CMIP6 variable ID (e.g., "tas", "tos")

    Returns
    -------
    str
        CMIP7 realm (e.g., "atmos", "ocean", "land")

    Raises
    ------
    KeyError
        If the variable is not found in the Data Request mappings.
    """
    entry = get_dreq_entry(table_id, variable_id)
    return entry.realm


def parse_variant_label(variant_label: str) -> dict[str, int]:
    """
    Parse a CMIP6 variant label into its component indexes.

    Parameters
    ----------
    variant_label
        Variant label string (e.g., "r1i1p1f1", "r3i2p4f5")

    Returns
    -------
    dict
        Dictionary with keys ``realization_index``, ``initialization_index``,
        ``physics_index``, ``forcing_index`` mapped to their integer values.

    Raises
    ------
    ValueError
        If the variant label does not match the expected pattern.
    """
    match = re.match(r"r(\d+)i(\d+)p(\d+)f(\d+)", variant_label)
    if not match:
        raise ValueError(f"Cannot parse variant label: {variant_label!r}")
    return {
        "realization_index": int(match.group(1)),
        "initialization_index": int(match.group(2)),
        "physics_index": int(match.group(3)),
        "forcing_index": int(match.group(4)),
    }


def convert_variant_index(value: int | str, prefix: str) -> str:
    """
    Convert CMIP6 numeric variant index to CMIP7 string format.

    In CMIP6, indices like realization_index were integers (e.g., 1).
    In CMIP7, they are strings with a prefix (e.g., "r1").

    Parameters
    ----------
    value
        The index value (int, numpy integer, or str)
    prefix
        The prefix to use ("r", "i", "p", or "f")

    Returns
    -------
    str
        The CMIP7 format index (e.g., "r1", "i1", "p1", "f1")
    """
    if isinstance(value, str):
        # Already has prefix
        if value.startswith(prefix):
            return value
        # Try to extract numeric part
        try:
            return f"{prefix}{int(value)}"
        except ValueError:
            return f"{prefix}{value}"

    try:
        return f"{prefix}{int(value)}"
    except (TypeError, ValueError):
        return f"{prefix}1"


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


def convert_cmip6_to_cmip7_attrs(
    cmip6_attrs: dict[str, Any],
) -> dict[str, Any]:
    """
    Convert CMIP6 global attributes to CMIP7 format.

    Based on CMIP7 Global Attributes V1.0 (DOI: 10.5281/zenodo.17250297).

    Parameters
    ----------
    cmip6_attrs
        Dictionary of CMIP6 global attributes. Must contain ``table_id``.

    Returns
    -------
    dict
        Dictionary of CMIP7 global attributes
    """
    # Start with a copy of existing attributes
    attrs = dict(cmip6_attrs)

    # Determine table_id and variable_id for branding lookup
    variable_id = attrs["variable_id"]
    table_id = attrs["table_id"]

    # Look up the information from the data request for this variable
    dreq_entry = get_dreq_entry(table_id, variable_id)

    # Static CMIP7 metadata
    cmip7_meta = CMIP7Metadata()

    # The variable id CMIP7 may differ from CMIP6 (tasmax -> tas)
    attrs["variable_id"] = dreq_entry.variable_id

    # Update mip_era
    attrs["mip_era"] = cmip7_meta.mip_era
    attrs["parent_mip_era"] = attrs.get("parent_mip_era", "CMIP6")

    # New/updated CMIP7 attributes
    attrs["region"] = dreq_entry.region
    attrs["realm"] = dreq_entry.realm
    attrs["frequency"] = dreq_entry.frequency

    attrs["drs_specs"] = CMIP7Metadata.drs_specs
    attrs["data_specs_version"] = CMIP7Metadata.data_specs_version
    attrs["product"] = CMIP7Metadata.product
    attrs["license_id"] = CMIP7Metadata.license_id

    # Add label attributes
    attrs["temporal_label"] = dreq_entry.temporal_label
    attrs["vertical_label"] = dreq_entry.vertical_label
    attrs["horizontal_label"] = dreq_entry.horizontal_label
    attrs["area_label"] = dreq_entry.area_label
    attrs["branding_suffix"] = dreq_entry.branding_suffix

    # Add branded_variable (required in CMIP7)
    attrs["branded_variable"] = dreq_entry.branded_variable

    # Convert variant indices from CMIP6 integer to CMIP7 string format.
    # If individual indexes are missing, parse them from variant_label
    parsed_indices: dict[str, int] = {}
    variant_label = attrs.get("variant_label")
    if variant_label:
        try:
            parsed_indices = parse_variant_label(variant_label)
        except ValueError:
            pass

    for index_name, prefix in [
        ("realization_index", "r"),
        ("initialization_index", "i"),
        ("physics_index", "p"),
        ("forcing_index", "f"),
    ]:
        raw = attrs.get(index_name, parsed_indices.get(index_name, 1))
        attrs[index_name] = convert_variant_index(raw, prefix)

    # Rebuild variant_label from converted indices
    attrs["variant_label"] = (
        f"{attrs['realization_index']}"
        f"{attrs['initialization_index']}"
        f"{attrs['physics_index']}"
        f"{attrs['forcing_index']}"
    )

    # Store legacy CMIP6 compound name for reference
    attrs["cmip6_compound_name"] = dreq_entry.cmip6_compound_name

    # Add tracking_id with CMIP7 handle prefix
    attrs["tracking_id"] = f"hdl:21.14107/{uuid.uuid4()}"

    # Add creation_date in ISO format
    attrs["creation_date"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

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

    This uses the cmip6 compound name (table_id.variable_id) to look up the appropriate CMIP7 branding suffix
    and realm from the Data Request mappings.
    It then updates the global attributes according to the CMIP7 Global Attributes.

    This doesn't modify the values or coordinates,
    but it does add a "cmip6_compound_name" attribute for reference.
    This may not be a fully valid CMIP7 dataset due to the grid names.

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

    variable_id = ds.attrs.get("variable_id")
    if variable_id is None and data_vars:
        variable_id = data_vars[0]
    if variable_id is None:  # pragma: no cover
        raise ValueError("Cannot determine variable_id for branding.")

    table_id = ds.attrs["table_id"]

    # Use the CMIP7 variable name
    dreq_entry = get_dreq_entry(table_id, variable_id)
    ds = ds.rename({variable_id: dreq_entry.variable_id})

    # Convert global attributes
    ds.attrs = convert_cmip6_to_cmip7_attrs(ds.attrs)

    return ds


def format_cmip7_time_range(ds: xr.Dataset, frequency: str) -> str | None:
    """
    Format a CMIP7 time range string from a dataset's time coordinate.

    Per the MIP-DRS7 spec, monthly data uses ``YYYYMM-YYYYMM`` format.
    Fixed-frequency (``"fx"``) data has no time range.

    Parameters
    ----------
    ds
        xarray Dataset with a ``"time"`` coordinate
    frequency
        Frequency string (e.g., ``"mon"``, ``"fx"``)

    Returns
    -------
    str or None
        Formatted time range string, or ``None`` for fixed-frequency data
        or datasets without a time coordinate.
    """
    if frequency == "fx" or "time" not in ds or len(ds["time"]) == 0:
        return None

    start = ds["time"].values[0]
    end = ds["time"].values[-1]

    def _strftime(t: Any) -> str:
        if isinstance(t, cftime.datetime):
            return str(t.strftime("%Y%m"))
        return str(pd.Timestamp(t).strftime("%Y%m"))

    return f"{_strftime(start)}-{_strftime(end)}"


def create_cmip7_filename(
    attrs: dict[str, Any],
    time_range: str | None = None,
) -> str:
    """
    Create a CMIP7 filename from attributes.

    The CMIP7 filename follows the MIP-DRS7 specification (V1.0):
    <variable_id>_<branding_suffix>_<frequency>_<region>_<grid_label>_<source_id>_<experiment_id>_<variant_label>[_<timeRangeDD>].nc

    Note that ``variable_id`` in CMIP7 may differ from CMIP6. For example,
    the CMIP6 "tasmax" variable maps to "tas" in CMIP7 because the "max" part
    is now captured in the branding suffix.

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
    filename_order = [
        "variable_id",
        "branding_suffix",
        "frequency",
        "region",
        "grid_label",
        "source_id",
        "experiment_id",
        "variant_label",
    ]

    filename = "_".join(str(attrs[c]) for c in filename_order)

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

    drs_order = [
        "drs_specs",
        "mip_era",
        "activity_id",
        "institution_id",
        "source_id",
        "experiment_id",
        "variant_label",
        "region",
        "frequency",
        "variable_id",
        "branding_suffix",
        "grid_label",
    ]

    components = [attrs[key] for key in drs_order]
    components.append(version_str)  # version goes at the end
    return "/".join(str(c) for c in components)
