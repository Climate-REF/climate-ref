"""
CMOR controlled vocabularies shared across source types.
"""

# CMIP6 CMOR table -> CMIP6 ``frequency`` CV value.
#
# A CMOR table name is a realm prefix plus a frequency suffix, but the reduction is not a plain
# suffix strip: ``Oclim`` is a monthly climatology (``monC``), ``E1hrClimMon`` is ``1hrCM``, the
# zonal-mean tables (``AERmonZ``, ``EmonZ``, ``EdayZ``, ``E6hrZ``) keep the frequency of their
# non-zonal counterpart, and the ``Pt`` (point-sampled) tables map to distinct ``*Pt``
# frequencies. So the mapping is enumerated rather than derived.
_MIP_TABLE_FREQUENCIES: dict[str, str] = {
    "3hr": "3hr",
    "6hrLev": "6hr",
    "6hrPlev": "6hr",
    "6hrPlevPt": "6hrPt",
    "AERday": "day",
    "AERhr": "1hr",
    "AERmon": "mon",
    "AERmonZ": "mon",
    "Amon": "mon",
    "CF3hr": "3hr",
    "CFday": "day",
    "CFmon": "mon",
    "CFsubhr": "subhrPt",
    "day": "day",
    "E1hr": "1hr",
    "E1hrClimMon": "1hrCM",
    "E3hr": "3hr",
    "E3hrPt": "3hrPt",
    "E6hrZ": "6hr",
    "Eday": "day",
    "EdayZ": "day",
    "Efx": "fx",
    "Emon": "mon",
    "EmonZ": "mon",
    "Esubhr": "subhrPt",
    "Eyr": "yr",
    "IfxAnt": "fx",
    "IfxGre": "fx",
    "ImonAnt": "mon",
    "ImonGre": "mon",
    "IyrAnt": "yr",
    "IyrGre": "yr",
    "LImon": "mon",
    "Lmon": "mon",
    "OIclim": "monC",
    "OIday": "day",
    "OImon": "mon",
    "Oclim": "monC",
    "Oday": "day",
    "Odec": "dec",
    "Ofx": "fx",
    "Omon": "mon",
    "Oyr": "yr",
    "SIday": "day",
    "SImon": "mon",
    "fx": "fx",
}

# The CMIP6 ``frequency`` CV. Values already in this set pass through ``frequency_from_mip_table``
# untouched, which is what lets a layout that stores a frequency rather than a table
# use the same call site.
_FREQUENCIES: frozenset[str] = frozenset(
    {
        "1hr",
        "1hrCM",
        "1hrPt",
        "3hr",
        "3hrPt",
        "6hr",
        "6hrPt",
        "day",
        "dec",
        "fx",
        "mon",
        "monC",
        "monPt",
        "subhrPt",
        "yr",
        "yrPt",
    }
)


def frequency_from_mip_table(value: str) -> str:
    """
    Reduce a CMOR MIP table name to its CMIP6 ``frequency`` CV value.

    Some datasets record ``frequency`` and not the MIP table.
    This maps a table (e.g. ``Amon`` -> ``mon``) onto the same axis.

    A value that is already a valid frequency is returned unchanged,
    so a caller reading a layout that stores ``mon`` or ``day`` can use this without branching.
    ``day`` and ``fx`` are both a table name and a frequency, and map to themselves either way.

    Parameters
    ----------
    value
        A CMOR MIP table name (``Amon``, ``Omon``, ``SIday``) or an existing frequency (``mon``).

    Returns
    -------
    :
        The corresponding CMIP6 frequency.

    Raises
    ------
    ValueError
        If the value is neither a known MIP table nor a known frequency. Failing loudly is
        deliberate: silently defaulting would let a mis-parsed path collapse two datasets that
        differ only by frequency onto one ``instance_id``.
    """
    if value in _MIP_TABLE_FREQUENCIES:
        return _MIP_TABLE_FREQUENCIES[value]
    if value in _FREQUENCIES:
        return value
    raise ValueError(f"Unknown MIP table or frequency: {value!r}")
