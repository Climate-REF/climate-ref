"""
Reading the keys of the ESMValTool reference data registry.

A :class:`~climate_ref_core.esgf.RegistryRequest` for this registry is built with
:func:`parse_registry_key`, so that a request for reference data filters on the facets
the ingest catalog records rather than on a second spelling of them.
"""

from typing import Any

from loguru import logger

from climate_ref_core.esmvaltool_reference import parse_reference_path


def parse_registry_key(key: str) -> dict[str, Any]:
    """
    Parse a registry key into the facets its path carries.

    A key is the file's DRS path relative to the data root,
    so it reads the same way the ingest adapter reads the downloaded file.

    Parameters
    ----------
    key
        The registry key (path) to parse.

    Returns
    -------
    :
        The facets the key carries, plus the key itself.
        An empty dict if the key is not a reference file, which skips it.
    """
    # Example: ESMValTool/OBS/Tier2/OSI-450-nh/OBS_OSI-450-nh_reanaly_v3_OImon_sic_197901-197912.nc
    try:
        facets = parse_reference_path(key)
    except ValueError as err:
        logger.debug(f"Unexpected ESMValTool reference key format: {key} ({err})")
        return {}

    metadata = facets._asdict()
    metadata["key"] = key
    return metadata
