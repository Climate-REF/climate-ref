"""
Declarative reference-dataset specifications for ESMValTool diagnostics.

ESMValTool diagnostics compare model output against reference (observational/reanalysis)
datasets that are not published as CMIP data. Historically each diagnostic hardcoded these
references as raw dictionaries inside ``update_recipe``. :class:`ESMValToolReferenceSpec`
makes them a declared, inspectable property of the diagnostic instead: a single source of
truth that feeds both recipe construction (:meth:`ESMValToolReferenceSpec.to_recipe_dataset`)
and dataset provenance.
"""

from __future__ import annotations

from typing import Any

from attrs import frozen


@frozen
class ESMValToolReferenceSpec:
    """
    A single reference dataset that an ESMValTool diagnostic depends on.

    The fields mirror the ESMValTool/ESMValCore dataset keys used in a recipe. Optional fields
    are omitted from the generated recipe entry when unset, matching how the references were
    previously written by hand.
    """

    project: str
    """ESMValCore project, e.g. ``OBS``, ``OBS6``, ``native6`` or ``obs4MIPs``."""

    dataset: str
    """Reference dataset name, e.g. ``OSI-450-nh``, ``HadCRUT5``, ``ERA5``."""

    mip: str | None = None
    """MIP table where applicable, e.g. ``OImon``, ``Amon``."""

    tier: int | None = None
    """ESMValTool data tier (accessibility)."""

    obs_type: str | None = None
    """ESMValTool observation ``type`` where applicable, e.g. ``reanaly``, ``sat``, ``ground``."""

    version: str | None = None
    """Dataset version, e.g. ``v3``, ``5.0.1.0-analysis``."""

    supplementary_variables: tuple[dict[str, str], ...] | None = None
    """Optional supplementary variables (e.g. cell measures) attached to the recipe entry."""

    def to_recipe_dataset(self) -> dict[str, Any]:
        """
        Build the ESMValTool recipe dataset entry for this reference.

        Keys are emitted in alphabetical order to match the historical hardcoded dictionaries,
        so the generated recipe YAML (dumped with ``sort_keys=False``) is byte-for-byte unchanged.
        """
        entry: dict[str, Any] = {"dataset": self.dataset}
        if self.mip is not None:
            entry["mip"] = self.mip
        entry["project"] = self.project
        if self.supplementary_variables is not None:
            entry["supplementary_variables"] = [dict(sv) for sv in self.supplementary_variables]
        if self.tier is not None:
            entry["tier"] = self.tier
        if self.obs_type is not None:
            entry["type"] = self.obs_type
        if self.version is not None:
            entry["version"] = self.version
        return entry
