"""
Helpers for allocating non-colliding output fragment paths.
"""

import pathlib


def allocate_output_fragment(
    base_fragment: str,
    existing_fragments: set[str],
    results_dir: pathlib.Path,
) -> str:
    """
    Return a non-colliding output fragment path.

    If *base_fragment* is not already used (neither in *existing_fragments* nor
    on disk under *results_dir*), it is returned unchanged.  Otherwise ``_v2``,
    ``_v3``, ... suffixes are tried until a free slot is found.

    Parameters
    ----------
    base_fragment
        The natural fragment, e.g. ``provider/diagnostic/dataset_hash``
    existing_fragments
        Set of ``output_fragment`` values already recorded in the database
        for the relevant execution group
    results_dir
        Root results directory; used to check for orphaned directories on disk

    Returns
    -------
    :
        A fragment string guaranteed not to collide with *existing_fragments*
        or any directory under *results_dir*
    """
    if base_fragment not in existing_fragments and not (results_dir / base_fragment).exists():
        return base_fragment

    version = 2
    while True:
        candidate = f"{base_fragment}_v{version}"
        if candidate not in existing_fragments and not (results_dir / candidate).exists():
            return candidate
        version += 1
