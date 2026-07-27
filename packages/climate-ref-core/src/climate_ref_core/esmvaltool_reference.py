"""
Path conventions for ESMValTool reference (observational/reanalysis) data.

ESMValCore locates this data from its own DRS directory templates rather than by
``instance_id``, so a reference file is identified by where its DRS path begins.
"""

import re
from pathlib import Path

# DRS shape of each project, counted from the anchor: the fewest and most path
# components it produces, and whether a ``TierN`` directory sits directly under the
# anchor. ``native6`` and ``obs4MIPs`` are fixed depth. ``OBS`` allows extra
# directories between the dataset and the file, so it has no upper bound.
_PROJECT_LAYOUTS = {
    "OBS": (3, None, True),
    "native6": (7, 7, True),
    "obs4MIPs": (4, 4, False),
}

PROJECT_ANCHORS = tuple(_PROJECT_LAYOUTS)
"""
Top-level project directories, relative to the ``ESMValTool`` data root, that begin a DRS path.

``OBS6`` data lives under the ``OBS`` directory, so the anchor for both is ``OBS``.
The project itself is recovered from the filename.
"""

_TIER_RE = re.compile(r"^Tier\d+$")


def _is_plausible_drs(rel: tuple[str, ...]) -> bool:
    """Report whether ``rel`` looks like a DRS path for the project it starts with."""
    min_parts, max_parts, tiered = _PROJECT_LAYOUTS[rel[0]]
    if len(rel) < min_parts or (max_parts is not None and len(rel) > max_parts):
        return False
    # Only the tiered projects put a ``TierN`` directory directly under the anchor,
    # so its presence tells the two layouts apart.
    return tiered == bool(_TIER_RE.match(rel[1]))


def drs_relative_parts(path: str | Path) -> tuple[str, ...]:
    """
    Split a reference file path into its DRS-relative components.

    A data root may itself contain a directory named after a project, and a directory
    inside the tree may in turn be named after another project, so neither the first nor
    the last matching component is reliable on its own. The rightmost candidate whose
    remaining structure fits its project is used, falling back to the leftmost candidate
    when none fits.

    Parameters
    ----------
    path
        Path to a reference file, absolute or relative.

    Returns
    -------
    :
        The path components from the project anchor onward,
        so ``.../ESMValTool/OBS/Tier2/CERES-EBAF/x.nc`` becomes
        ``("OBS", "Tier2", "CERES-EBAF", "x.nc")``.

    Raises
    ------
    ValueError
        If no component of ``path`` is one of :data:`PROJECT_ANCHORS`.
    """
    parts = Path(path).parts
    candidates = [index for index, part in enumerate(parts) if part in PROJECT_ANCHORS]

    if not candidates:
        raise ValueError(
            f"{path} is not under a known ESMValTool reference project ({', '.join(PROJECT_ANCHORS)})"
        )

    for index in reversed(candidates):
        if _is_plausible_drs(parts[index:]):
            return parts[index:]

    return parts[candidates[0] :]
