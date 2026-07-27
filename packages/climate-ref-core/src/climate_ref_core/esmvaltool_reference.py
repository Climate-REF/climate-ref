"""
Path conventions for ESMValTool reference (observational/reanalysis) data.

ESMValCore locates this data from its own DRS directory templates rather than by
``instance_id``, so a reference file is identified by where its DRS path begins.
"""

import re
from pathlib import Path
from typing import NamedTuple


class _Layout(NamedTuple):
    """DRS shape of one project, counted from its anchor."""

    min_parts: int
    max_parts: int | None
    """``None`` where the project allows extra directories above the file."""
    tiered: bool
    """Whether a ``TierN`` directory sits directly under the anchor."""


_PROJECT_LAYOUTS = {
    "OBS": _Layout(min_parts=3, max_parts=None, tiered=True),
    "native6": _Layout(min_parts=7, max_parts=7, tiered=True),
    "obs4MIPs": _Layout(min_parts=4, max_parts=4, tiered=False),
}

PROJECT_ANCHORS = tuple(_PROJECT_LAYOUTS)
"""
Top-level project directories, relative to the ``ESMValTool`` data root, that begin a DRS path.

``OBS6`` data lives under the ``OBS`` directory, so the anchor for both is ``OBS``.
The project itself is recovered from the filename.
"""

_TIER_RE = re.compile(r"^Tier\d+$")


def matches_project_layout(rel: tuple[str, ...]) -> bool:
    """
    Report whether DRS-relative components fit the layout of the project they start with.

    Parameters
    ----------
    rel
        Path components from the project anchor onward, as returned by
        :func:`drs_relative_parts`.

    Returns
    -------
    :
        Whether ``rel`` has a usable depth and tier directory for its project.
    """
    layout = _PROJECT_LAYOUTS[rel[0]]

    # Every layout is at least three components deep, so this also guards ``rel[1]`` below.
    if len(rel) < layout.min_parts:
        return False
    if layout.max_parts is not None and len(rel) > layout.max_parts:
        return False

    return layout.tiered == bool(_TIER_RE.match(rel[1]))


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
        if matches_project_layout(parts[index:]):
            return parts[index:]

    return parts[candidates[0] :]
