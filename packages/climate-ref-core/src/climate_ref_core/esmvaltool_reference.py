"""
Path conventions for ESMValTool reference (observational/reanalysis) data.

ESMValCore locates this data from its own DRS directory templates rather than by ``instance_id``,
so a reference file is identified by where its DRS path begins.
"""

from pathlib import Path
from typing import NamedTuple


class _Layout(NamedTuple):
    """DRS shape of one project, counted from its anchor."""

    parts: int
    """The number of path components the project's template produces."""
    allow_extra: bool
    """Whether extra directories may sit between the dataset and the file."""
    tiered: bool
    """Whether a ``TierN`` directory sits directly under the anchor."""


_PROJECT_LAYOUTS = {
    "OBS": _Layout(parts=3, allow_extra=True, tiered=True),
    "native6": _Layout(parts=7, allow_extra=False, tiered=True),
    "obs4MIPs": _Layout(parts=4, allow_extra=False, tiered=False),
}

PROJECT_ANCHORS = tuple(_PROJECT_LAYOUTS)
"""
Top-level project directories, relative to the ``ESMValTool`` data root, that begin a DRS path.

``OBS6`` data lives under the ``OBS`` directory, so the anchor for both is ``OBS``.
The project itself is recovered from the filename.
"""


def tier_from_segment(segment: str) -> int | None:
    """
    Parse a ``TierN`` directory name into its tier number.

    Parameters
    ----------
    segment
        A single path component.

    Returns
    -------
    :
        The tier number, or ``None`` if ``segment`` does not name a tier.
    """
    if segment.startswith("Tier") and segment[4:].isdigit():
        return int(segment[4:])
    return None


def _fits_layout(rel: tuple[str, ...]) -> bool:
    """Report whether DRS-relative components fit the layout of the project they start with."""
    layout = _PROJECT_LAYOUTS[rel[0]]

    # Every layout is at least three components deep, so this also guards ``rel[1]`` below.
    if len(rel) < layout.parts:
        return False
    if not layout.allow_extra and len(rel) > layout.parts:
        return False

    return layout.tiered == (tier_from_segment(rel[1]) is not None)


def drs_relative_parts(path: str | Path) -> tuple[str, ...]:
    """
    Split a reference file path into its DRS-relative components.

    A data root may itself contain a directory named after a project, and a directory
    inside the tree may in turn be named after another project, so neither the first nor
    the last matching component is reliable on its own. The rightmost candidate whose
    remaining components fit its project's layout wins.

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
        If no component of ``path`` names a project,
        or if none of them begins a path that fits that project's layout.
    """
    parts = Path(path).parts
    candidates = [index for index, part in enumerate(parts) if part in PROJECT_ANCHORS]

    if not candidates:
        raise ValueError(
            f"{path} is not under a known ESMValTool reference project ({', '.join(PROJECT_ANCHORS)})"
        )

    for index in reversed(candidates):
        if _fits_layout(parts[index:]):
            return parts[index:]

    raise ValueError(f"unexpected {parts[candidates[0]]} path structure: {path}")
