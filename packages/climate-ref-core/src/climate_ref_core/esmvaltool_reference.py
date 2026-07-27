"""
Path conventions for ESMValTool reference (observational/reanalysis) data.

ESMValCore locates this data from its own DRS directory templates rather than by ``instance_id``,
so a reference file is identified by where its DRS path begins.

This data is not CMOR/obs4MIPs compliant, so its metadata cannot be read from global attributes.
It is parsed from the path and filename templates that ESMValTool itself uses to find the data:

* ``OBS`` / ``OBS6`` (metadata from the filename):
  ``OBS/Tier{tier}/{dataset}/{project}_{dataset}_{type}_{version}_{mip}_{short_name}_{timerange}.nc``
* ``native6`` (metadata from the directory, raw non-CMOR filename):
  ``native6/Tier{tier}/{dataset}/{version}/{frequency}/{short_name}/*.nc``
* ``obs4MIPs``: ``obs4MIPs/{dataset}/{version}/{short_name}_*.nc``

Both the ingest adapter and the registry fetcher parse through :func:`parse_reference_path`,
so a request cannot name a facet value that ingest would spell differently.
"""

from pathlib import Path
from typing import NamedTuple

from climate_ref_core.cmor import frequency_from_mip_table


class _Layout(NamedTuple):
    """DRS shape of one project, counted from its anchor."""

    parts: int
    """The number of path components the project's template produces."""
    allow_extra: bool
    """Whether extra directories may sit between the dataset and the file."""
    tiered: bool
    """Whether a ``TierN`` directory sits directly under the anchor."""


_PROJECT_LAYOUTS = {
    "OBS": _Layout(parts=4, allow_extra=True, tiered=True),
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

    # Every layout is at least four components deep, so this also guards ``rel[1]`` below.
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


# The shortest ``OBS`` filename that still carries every field of the template.
_OBS_FILENAME_TOKENS = 6


class ReferenceFacets(NamedTuple):
    """
    Metadata carried by the path of one ESMValTool reference file.

    ``timerange`` is the raw DRS token rather than a parsed date range,
    because it is not a facet anything selects on.
    """

    project: str
    source_id: str
    variable_id: str
    frequency: str
    version: str
    data_type: str | None
    tier: int | None
    timerange: str | None


def _filename_fields(filename: str) -> list[str]:
    """Split a CMOR-style filename into its underscore-separated fields."""
    stem = filename[:-3] if filename.endswith(".nc") else filename
    return stem.split("_")


def _parse_obs(rel: tuple[str, ...], filename: str) -> ReferenceFacets:
    # rel == ("OBS", "Tier{n}", "{dataset}", ..., filename)
    tokens = _filename_fields(filename)
    # {project}_{dataset}_{type}_{version}_{mip}_{short_name}[_{timerange}]
    if len(tokens) < _OBS_FILENAME_TOKENS:
        raise ValueError(f"unexpected OBS filename structure: {filename}")
    project, _, data_type, version, mip, short_name = tokens[:_OBS_FILENAME_TOKENS]
    # The timerange is the trailing token. Use ``tokens[-1]`` (matching ``_parse_obs4mips``)
    # so an unexpected extra segment does not silently drop the date range.
    return ReferenceFacets(
        project=project,
        source_id=rel[2],
        variable_id=short_name,
        frequency=frequency_from_mip_table(mip),
        version=version,
        data_type=data_type,
        tier=tier_from_segment(rel[1]),
        timerange=tokens[-1] if len(tokens) > _OBS_FILENAME_TOKENS else None,
    )


def _parse_native6(rel: tuple[str, ...]) -> ReferenceFacets:
    # rel == ("native6", "Tier{n}", "{dataset}", "{version}", "{frequency}", "{short_name}", filename)
    return ReferenceFacets(
        project="native6",
        source_id=rel[2],
        variable_id=rel[5],
        frequency=frequency_from_mip_table(rel[4]),
        version=rel[3],
        data_type=None,
        tier=tier_from_segment(rel[1]),
        # native6 filenames are raw (non-CMOR) and carry no reliable DRS date range.
        timerange=None,
    )


def _parse_obs4mips(rel: tuple[str, ...], filename: str) -> ReferenceFacets:
    # rel == ("obs4MIPs", "{dataset}", "{version}", filename)
    tokens = _filename_fields(filename)
    if not tokens[0]:
        raise ValueError(f"unexpected obs4MIPs filename structure: {filename}")
    return ReferenceFacets(
        project="obs4MIPs",
        source_id=rel[1],
        variable_id=tokens[0],
        # obs4MIPs reference files here are monthly, so "mon" is the non-null grouping key.
        frequency="mon",
        version=rel[2],
        data_type=None,
        tier=None,
        timerange=tokens[-1] if len(tokens) > 1 else None,
    )


def parse_reference_path(path: str | Path) -> ReferenceFacets:
    """
    Read the metadata a reference file's DRS path encodes.

    Parameters
    ----------
    path
        Path to a reference file, absolute or relative.
        It may equally be a registry key, which is the same path relative to the data root.

    Returns
    -------
    :
        The facets the path carries.

    Raises
    ------
    ValueError
        If the path does not fit any project layout,
        or if the filename does not fit the template of the project it sits under.
    """
    file = Path(path)
    rel = drs_relative_parts(file)
    anchor = rel[0]

    if anchor == "OBS":
        return _parse_obs(rel, file.name)
    if anchor == "native6":
        return _parse_native6(rel)
    return _parse_obs4mips(rel, file.name)
