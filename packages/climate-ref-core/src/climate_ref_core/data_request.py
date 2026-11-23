"""
Variables of the CMIP7 Data Request.

Maps the CMIP6 variables the REF uses (``<table_id>.<variable_id>``) to their CMIP7 counterparts:
branded variable, branding labels, realm, frequency and region.

The mappings are a subset of the CMIP7 Data Request (DReq v1.2.2.3) bundled as JSON.
Regenerate it with:

    python scripts/extract-data-request-mappings.py
"""

from __future__ import annotations

import json
from functools import cache
from importlib import resources
from typing import Any

import attrs


@attrs.frozen
class DReqVariableMapping:
    """
    A single CMIP6-to-CMIP7 variable mapping from the Data Request.

    Each instance represents one row in the DReq Variables JSON file,
    capturing the CMIP6 compound name, its CMIP7 equivalent,
    and its branding and realm.
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


@cache
def _load_dreq_mappings() -> dict[str, DReqVariableMapping]:
    """Load the bundled mappings, keyed by CMIP6 compound name (e.g. ``"Amon.tas"``)."""
    data_files = resources.files("climate_ref_core") / "data" / "cmip6_cmip7_variable_map.json"
    raw: dict[str, Any] = json.loads(data_files.read_text(encoding="utf-8"))
    variables: dict[str, dict[str, Any]] = raw.get("variables", {})
    return {key: DReqVariableMapping.from_dict(entry) for key, entry in variables.items()}


def dreq_entries() -> tuple[DReqVariableMapping, ...]:
    """
    Return every bundled Data Request entry.

    Returns
    -------
    :
        The entries, one per CMIP6 compound name
    """
    return tuple(_load_dreq_mappings().values())


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
    entry = _load_dreq_mappings().get(cmip6_compound)
    if entry is None:
        raise KeyError(
            f"Variable '{cmip6_compound}' not found in Data Request mappings. "
            f"Add it to INCLUDED_VARIABLES in scripts/extract-data-request-mappings.py and regenerate."
        )
    return entry
