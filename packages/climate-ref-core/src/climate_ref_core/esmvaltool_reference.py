"""
Path conventions for ESMValTool reference (observational/reanalysis) data.

ESMValTool reference data is located by ESMValCore from its own DRS directory templates
rather than by ``instance_id``, so both the ingest adapter and the diagnostic runner need
to know where the DRS-relative part of a path begins. That knowledge lives here so the two
cannot drift apart.
"""

from pathlib import Path

PROJECT_ANCHORS = ("OBS", "native6", "obs4MIPs")
"""
Top-level project directories, relative to the ``ESMValTool`` data root, that begin a DRS path.

``OBS6`` data lives under the ``OBS`` directory, so the anchor for both is ``OBS``.
The project itself is recovered from the filename.
"""


def relative_parts(path: str | Path) -> tuple[str, ...]:
    """
    Split a reference file path into its DRS-relative components.

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

    # Search from the right, so a data root that happens to contain a directory
    # named after a project does not truncate the path at the wrong place.
    for offset, part in enumerate(reversed(parts)):
        if part in PROJECT_ANCHORS:
            return parts[len(parts) - 1 - offset :]

    raise ValueError(
        f"{path} is not under a known ESMValTool reference project ({', '.join(PROJECT_ANCHORS)})"
    )
