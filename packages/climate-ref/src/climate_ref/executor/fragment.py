"""
Helpers for allocating non-colliding output fragment paths.
"""

import datetime


def allocate_output_fragment(base_fragment: str) -> str:
    """
    Return a unique output fragment by appending a UTC timestamp.

    The returned fragment is ``{base_fragment}_{YYYYMMDDTHHMMSS}``, which is
    practically unique without needing any database or filesystem lookups.

    Parameters
    ----------
    base_fragment
        The natural fragment, e.g. ``provider/diagnostic/dataset_hash``

    Returns
    -------
    :
        A new fragment with a timestamp suffix
    """
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    return f"{base_fragment}_{now.strftime('%Y%m%dT%H%M%S')}"
