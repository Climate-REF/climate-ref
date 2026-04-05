"""
Helpers for allocating non-colliding output fragment paths.
"""

import datetime
from pathlib import Path


def allocate_output_fragment(base_fragment: str, results_dir: Path) -> str:
    """
    Return a unique output fragment by appending a UTC timestamp.

    The returned fragment is ``{base_fragment}_{YYYYMMDDTHHMMSSffffff}``, which is
    practically unique without needing any database or filesystem lookups.
    Microsecond resolution avoids collisions from rapid successive calls.

    Parameters
    ----------
    base_fragment
        The natural fragment, e.g. ``provider/diagnostic/dataset_hash``
    results_dir
        The results root directory. Used to verify the allocated fragment
        does not already exist on disk.

    Returns
    -------
    :
        A new fragment with a timestamp suffix

    Raises
    ------
    FileExistsError
        If the computed output directory already exists under *results_dir*
    """
    now = datetime.datetime.now(tz=datetime.timezone.utc)
    fragment = f"{base_fragment}_{now.strftime('%Y%m%dT%H%M%S%f')}"

    target = results_dir / fragment
    if target.exists():
        raise FileExistsError(
            f"Output directory already exists: {target}. Cannot allocate fragment '{fragment}'."
        )

    return fragment
