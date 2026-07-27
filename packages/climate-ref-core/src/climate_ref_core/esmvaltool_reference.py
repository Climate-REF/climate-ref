"""
Path conventions for ESMValTool reference (observational/reanalysis) data.

ESMValTool reference data is located by ESMValCore from its own DRS directory templates
rather than by ``instance_id``, so both the ingest adapter and the diagnostic runner need
to know where the DRS-relative part of a path begins. That knowledge lives here so the two
cannot drift apart.
"""

import re
from pathlib import Path

PROJECT_ANCHORS = ("OBS", "native6", "obs4MIPs")
"""
Top-level project directories, relative to the ``ESMValTool`` data root, that begin a DRS path.

``OBS6`` data lives under the ``OBS`` directory, so the anchor for both is ``OBS``.
The project itself is recovered from the filename.
"""

_TIER_RE = re.compile(r"^Tier\d+$")

# Shortest DRS path each project can produce, counted from the anchor.
_OBS_MIN_PARTS = 3
_OBS4MIPS_MIN_PARTS = 4
_NATIVE6_MIN_PARTS = 7


def _is_plausible_drs(rel: tuple[str, ...]) -> bool:
    """Report whether ``rel`` looks like a DRS path for the project it starts with."""
    if rel[0] == "obs4MIPs":
        return len(rel) >= _OBS4MIPS_MIN_PARTS

    minimum = _NATIVE6_MIN_PARTS if rel[0] == "native6" else _OBS_MIN_PARTS
    return len(rel) >= minimum and bool(_TIER_RE.match(rel[1]))


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
