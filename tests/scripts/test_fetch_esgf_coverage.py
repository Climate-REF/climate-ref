"""
Tests that `scripts/fetch-esgf.py` fetches the obs4MIPs reference data the diagnostics ask for.

The requests in that script are maintained by hand so they can drift from the providers' data requirements.
These tests check that the obs data requested is covered by a download.

No network access is performed and nothing is fetched.
"""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from climate_ref_core.diagnostics import DataRequirement
from climate_ref_core.source_types import SourceDatasetType

REPO_ROOT = Path(__file__).parents[2]
SCRIPT = REPO_ROOT / "scripts" / "fetch-esgf.py"
OBS4MIPS_CATALOG = REPO_ROOT / "tests" / "test-data" / "esgf-catalog" / "obs4mips_catalog.parquet"
OBS4REF_REGISTRY = (
    REPO_ROOT
    / "packages"
    / "climate-ref"
    / "src"
    / "climate_ref"
    / "dataset_registry"
    / "obs4ref_reference.txt"
)

# Number of path parts in an obs4REF registry key:
# obs4REF/{institution_id}/{source_id}/{frequency}/{variable_id}/{grid_label}/{version}/{filename}
_OBS4REF_KEY_PARTS = 8
_KEY_SOURCE_ID = 2
_KEY_VARIABLE_ID = 4


@pytest.fixture(scope="module")
def script():
    spec = importlib.util.spec_from_file_location("fetch_esgf", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def obs4mips_requirements() -> list[tuple[str, DataRequirement]]:
    """Every obs4MIPs-typed data requirement across the default providers, tagged with its slug."""
    climate_ref_esmvaltool = pytest.importorskip("climate_ref_esmvaltool")
    climate_ref_ilamb = pytest.importorskip("climate_ref_ilamb")
    climate_ref_pmp = pytest.importorskip("climate_ref_pmp")

    requirements = []
    for provider in (
        climate_ref_esmvaltool.provider,
        climate_ref_pmp.provider,
        climate_ref_ilamb.provider,
    ):
        for diagnostic in provider.diagnostics():
            for item in diagnostic.data_requirements:
                branch = item if isinstance(item, (list, tuple)) else [item]
                for requirement in branch:
                    if requirement.source_type is SourceDatasetType.obs4MIPs:
                        requirements.append((f"{provider.slug}/{diagnostic.slug}", requirement))
    assert requirements, "no obs4MIPs requirements found; the providers failed to load"
    return requirements


@pytest.fixture(scope="module")
def obs4ref_source_ids() -> set[str]:
    """The `source_id`s the obs4REF registry provides, which need no ESGF request."""
    source_ids = set()
    for raw_line in OBS4REF_REGISTRY.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()[0].split("/")
        if len(parts) == _OBS4REF_KEY_PARTS:
            source_ids.add(parts[_KEY_SOURCE_ID])
    return source_ids


@pytest.fixture(scope="module")
def esgf_catalog() -> pd.DataFrame:
    """
    The ESGF-published part of the committed obs4MIPs catalog snapshot.

    The snapshot is built from a local archive that also holds the obs4REF collection,
    so rows whose path sits under `obs4REF/` are excluded: they are provided by the registry
    rather than by a fetch from ESGF.
    """
    catalog = pd.read_parquet(OBS4MIPS_CATALOG)
    return catalog[~catalog["path"].str.contains("/obs4REF/")]


def _requested_pairs(script) -> dict[tuple[str, str], str]:
    """Map every (source_id, variable_id) the script requests to the id of the request."""
    pairs = {}
    for request in script.requests:
        if not isinstance(request, script.Obs4MIPsRequest):
            continue
        source_ids = request.facets["source_id"]
        variable_ids = request.facets["variable_id"]
        source_ids = (source_ids,) if isinstance(source_ids, str) else source_ids
        variable_ids = (variable_ids,) if isinstance(variable_ids, str) else variable_ids
        for source_id in source_ids:
            for variable_id in variable_ids:
                pairs[(source_id, variable_id)] = request.id
    return pairs


def _requirement_source_ids(requirement: DataRequirement) -> set[str]:
    source_ids: set[str] = set()
    for facet_filter in requirement.filters:
        value = facet_filter.facets.get("source_id")
        if value is None:
            continue
        source_ids.update((value,) if isinstance(value, str) else value)
    return source_ids


def test_every_required_source_is_obtainable(script, obs4mips_requirements, obs4ref_source_ids):
    """
    Every `source_id` a diagnostic asks for is either in the obs4REF registry or fetched from ESGF.

    This is the check that catches a new diagnostic naming a reference dataset that no one can obtain,
    which would otherwise surface only as a diagnostic that never runs.
    """
    requested = {source_id for source_id, _ in _requested_pairs(script)}

    unobtainable = {
        source_id: slug
        for slug, requirement in obs4mips_requirements
        for source_id in _requirement_source_ids(requirement)
        if source_id not in obs4ref_source_ids and source_id not in requested
    }

    assert not unobtainable, (
        "These source_ids are required by a diagnostic but are in neither the obs4REF registry "
        f"nor an Obs4MIPsRequest in {SCRIPT.name}: {unobtainable}"
    )


def test_every_selectable_dataset_is_fetched(script, obs4mips_requirements, esgf_catalog):
    """
    Every ESGF dataset a requirement can actually select is covered by a request.

    Resolving against the catalog rather than taking the cross product of the requirement's facets matters.
    ESGF intersects its facets,
    so a requirement naming four sources and eight variables does not need all thirty-two combinations,
    only the ones that exist.
    """
    requested = _requested_pairs(script)

    missing: dict[tuple[str, str], set[str]] = {}
    for slug, requirement in obs4mips_requirements:
        selected = requirement.apply_filters(esgf_catalog)
        for source_id, variable_id in zip(selected["source_id"], selected["variable_id"]):
            if (source_id, variable_id) not in requested:
                missing.setdefault((source_id, variable_id), set()).add(slug)

    assert not missing, (
        f"These obs4MIPs datasets are selected by a diagnostic but no request in {SCRIPT.name} "
        f"fetches them: { {pair: sorted(slugs) for pair, slugs in missing.items()} }"
    )


def test_no_request_is_stale(script, obs4mips_requirements):
    """
    Every requested `source_id` is still named by a requirement.

    Avoid silently downloads data that nothing will read.
    """
    required = {
        source_id
        for _, requirement in obs4mips_requirements
        for source_id in _requirement_source_ids(requirement)
    }

    stale = {
        source_id: request_id
        for (source_id, _), request_id in _requested_pairs(script).items()
        if source_id not in required
    }

    assert not stale, (
        f"These requests in {SCRIPT.name} fetch a source_id that no diagnostic requires: {stale}"
    )
