"""
CLI tool for fetching the required CMIP6 and Obs4MIPs datasets from ESGF.

This script can either run all predefined requests or a specific request by ID.
By default, only one ensemble member per model is fetched to reduce the total data volume.
This can be changed with the --no-remove-ensembles flag.

This fetches about 3TB of datasets into the default location for intake esgf.
This can be adjusted via `~/.config/intake-esgf/config.yaml`.
"""

import copy
import enum
import time
from typing import Any

import pandas as pd
import typer
from attrs import define
from intake_esgf.exceptions import (
    DatasetLoadError,
    GlobusTransferError,
    StalledDownload,
)
from loguru import logger

from climate_ref_core.esgf import CMIP6Request, Obs4MIPsRequest
from climate_ref_core.exceptions import DatasetResolutionError

TRANSIENT_ERRORS = (DatasetLoadError, StalledDownload, GlobusTransferError)
"""
Errors raised when the search succeeded but the data could not be retrieved.

ESGF data nodes intermittently return 5xx responses or stall mid-transfer,
so these are worth retrying before giving up.
A `DatasetResolutionError` is deliberately excluded:
it means the request matched nothing, which retrying cannot change.
"""

DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_RETRY_DELAY = 5.0
DEFAULT_FREQUENCIES = ["mon", "fx"]

Request = CMIP6Request | Obs4MIPsRequest


def cmip6(slug: str, facets: dict[str, Any]) -> CMIP6Request:
    """Request CMIP6 data, keeping one ensemble member per model."""
    return CMIP6Request(slug, {"frequency": DEFAULT_FREQUENCIES, **facets}, remove_ensembles=True)


def obs4mips(slug: str, facets: dict[str, Any]) -> Obs4MIPsRequest:
    """Request obs4MIPs reference data."""
    return Obs4MIPsRequest(slug, {"frequency": DEFAULT_FREQUENCIES, **facets})


# TODO use the data requirements from the diagnostics directly.
# Until then, `tests/scripts/test_fetch_esgf_coverage.py`
# checks the obs4MIPs requests below against those requirements,
# so a diagnostic cannot start asking for reference data that nothing here fetches.
requests: list[Request] = [
    cmip6(
        "esmvaltool-climate-at-global-warmings-levels",
        dict(
            variable_id=["pr", "tas"],
            experiment_id=["ssp126", "ssp245", "ssp370", "ssp585", "historical"],
            table_id="Amon",
        ),
    ),
    # ESMValTool Cloud radiative effects
    cmip6(
        "esmvaltool-cloud-radiative-effects",
        dict(
            variable_id=["rlut", "rlutcs", "rsut", "rsutcs"],
            experiment_id="historical",
            table_id="Amon",
        ),
    ),
    obs4mips(
        "esmvaltool-cloud-radiative-effects-obs4mips",
        dict(
            source_id="CERES-EBAF-4-2-1",
            variable_id=["rlut", "rlutcs", "rsut", "rsutcs"],
        ),
    ),
    # ESMValTool cloud scatterplots
    cmip6(
        "esmvaltool-cloud-scatterplots-cmip6",
        dict(
            variable_id=[
                "areacella",
                "cli",
                "clivi",
                "clt",
                "clwvi",
                "pr",
                "rlut",
                "rlutcs",
                "rsut",
                "rsutcs",
                "ta",
            ],
            experiment_id="historical",
            table_id="Amon",
        ),
    ),
    obs4mips(
        "esmvaltool-cloud-scatterplots-obs4mips",
        dict(
            source_id="ERA-5",
            variable_id="ta",
        ),
    ),
    # ESMValTool ECS data
    cmip6(
        "esmvaltool-ecs",
        dict(
            variable_id=["rlut", "rsdt", "rsut", "tas"],
            experiment_id=["abrupt-4xCO2", "piControl"],
            table_id="Amon",
        ),
    ),
    # ESMValTool ENSO data
    cmip6(
        "esmvaltool-enso",
        dict(
            variable_id=[
                "pr",
                "tos",
                "tauu",
            ],
            experiment_id=["historical"],
            table_id=("Amon", "Omon"),
        ),
    ),
    # ESMValTool fire data
    cmip6(
        "esmvaltool-fire",
        dict(
            variable_id=[
                "cVeg",
                "hurs",
                "pr",
                "sftlf",
                "tas",
                "tasmax",
                "treeFrac",
                "vegFrac",
            ],
            experiment_id=["historical"],
            table_id=("Amon", "Emon", "Lmon"),
        ),
    ),
    # ESMValTool Historical data
    cmip6(
        "esmvaltool-historical-cmip6",
        dict(
            variable_id=[
                "hus",
                "pr",
                "psl",
                "tas",
                "ua",
            ],
            experiment_id=["historical"],
            table_id="Amon",
        ),
    ),
    obs4mips(
        "esmvaltool-historical-obs4mips",
        dict(
            source_id="ERA-5",
            variable_id=[
                "psl",
                "tas",
                "ua",
            ],
        ),
    ),
    # ESMValTool TCR data
    cmip6(
        "esmvaltool-tcr",
        dict(
            variable_id=["tas"],
            experiment_id=["1pctCO2", "piControl"],
            table_id="Amon",
        ),
    ),
    # ESMValTool TCRE data
    cmip6(
        "esmvaltool-tcre",
        dict(
            variable_id=["fco2antt", "tas"],
            experiment_id=["esm-1pctCO2", "esm-piControl"],
            table_id="Amon",
        ),
    ),
    # ESMValTool ZEC data
    cmip6(
        "esmvaltool-zec",
        dict(
            variable_id=["areacella", "tas"],
            experiment_id=["1pctCO2", "esm-1pct-brch-1000PgC"],
            table_id="Amon",
        ),
    ),
    # ESMValTool Sea Ice Area Seasonal Cycle data
    cmip6(
        "esmvaltool-sea-ice-area-seasonal-cycle",
        dict(
            variable_id=["areacello", "siconc"],
            experiment_id=["historical"],
        ),
    ),
    # ESMValTool Ozone
    cmip6(
        "esmvaltool-ozone",
        dict(
            variable_id=["toz", "o3"],
            experiment_id=["historical"],
            table_id="AERmon",
        ),
    ),
    obs4mips(
        "esmvaltool-ozone-obs4mips",
        dict(
            source_id="C3S-GTO-ECV-9-0",
            variable_id="toz",
        ),
    ),
    # ILAMB data
    cmip6(
        "ilamb-data",
        dict(
            variable_id=[
                "areacella",
                "sftlf",
                "gpp",
                "pr",
                "tas",
                "mrro",
                "mrsos",
                "cSoil",
                "lai",
                "areacella",
                "burntFractionAll",
                "snc",
                "nbp",
                "et",
            ],
            experiment_id=["historical"],
        ),
    ),
    # ILAMB lai reference
    obs4mips(
        "ilamb-lai-obs4mips",
        dict(
            source_id="NOAA-NCEI-LAI-AVHRR-5-0",
            variable_id="lai",
        ),
    ),
    # IOMB data
    cmip6(  # Already provided by the ESMValTool ENSO request.
        "iomb-data",
        dict(
            variable_id=["areacello", "tos"],
            experiment_id=["historical"],
        ),
    ),
    cmip6(
        "iomb-data-2",
        dict(
            variable_id=["sftof", "sos", "msftmz"],
            experiment_id=["historical"],
        ),
    ),
    # Large 4D ocean datasets
    # A small set of models are fetched here for now
    cmip6(
        "iomb-data-large",
        dict(
            variable_id=["volcello", "thetao"],
            experiment_id=["historical"],
            source_id=[
                "AWI-ESM-1-1-LR",
                "CAMS-CSM1-0",
                "TaiESM1",
                "CanESM5",
                "CanESM5-1",
                "CanESM5-CanOE",
                "FGOALS-g3",
                "FGOALS-f3-L",
                "CAS-ESM2-0",
                "BCC-ESM1",
                "BCC-CSM2-MR",
            ],
        ),
    ),
    # PMP modes of variability data
    cmip6(
        "pmp-modes-of-variability",
        dict(
            variable_id=["areacella", "ts", "psl"],
            experiment_id=["historical", "hist-GHG"],
            table_id="Amon",
        ),
    ),
    # PMP ENSO reference data.
    # One request per source_id: the ESGF search treats source_id and variable_id as an
    # intersection, so a combined request would ask each source for variables it does not have.
    obs4mips(
        "pmp-enso-gpcp-obs4mips",
        dict(
            source_id="GPCP-Monthly-3-2",
            variable_id="pr",
        ),
    ),
    obs4mips(
        "pmp-enso-tropflux-obs4mips",
        dict(
            source_id="TropFlux-1-0",
            variable_id=["hfls", "hfss", "tauu", "ts"],
        ),
    ),
    obs4mips(
        "pmp-enso-ceres-obs4mips",
        dict(
            source_id="CERES-EBAF-4-2",
            variable_id=["rlds", "rlus", "rsds", "rsus"],
        ),
    ),
    # Shared by PMP ENSO, the PMP ts modes of variability, and the example global-sst-bias.
    obs4mips(
        "pmp-hadisst-obs4mips",
        dict(
            source_id="HadISST-1-1",
            variable_id="ts",
        ),
    ),
    obs4mips(
        "pmp-modes-20cr-obs4mips",
        dict(
            source_id="20CR-V2",
            variable_id="psl",
        ),
    ),
]


class Outcome(enum.Enum):
    """
    The result of attempting a single request
    """

    OK = "ok"
    """The search matched datasets and every file was retrieved."""

    EMPTY = "empty"
    """The search matched no datasets. Not treated as a failure."""

    FAILED = "failed"
    """The request could not be completed. Causes a non-zero exit code."""


@define
class RequestResult:
    """
    The outcome of a single request, used to build the final summary
    """

    request_id: str
    """The id of the request that was attempted."""

    outcome: Outcome
    """Whether the request succeeded, matched nothing, or failed."""

    dataset_count: int = 0
    """Number of datasets retrieved. Always zero unless the outcome is `OK`."""

    error: str | None = None
    """A description of the failure. Only set when the outcome is `FAILED`."""


def fetch_with_retry(
    request: Request,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    retry_delay: float = DEFAULT_RETRY_DELAY,
) -> pd.DataFrame:
    """
    Fetch a request, retrying transient ESGF access failures with an exponential backoff.

    Only the errors in `TRANSIENT_ERRORS` are retried.
    Everything else, including `DatasetResolutionError`, is raised immediately.

    Parameters
    ----------
    request
        The request to fetch.
    max_attempts
        Total number of attempts, including the first.
    retry_delay
        Seconds to wait before the second attempt. Doubles on each subsequent retry.

    Returns
    -------
    :
        The fetched datasets, one row per dataset.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            return request.fetch_datasets()
        except TRANSIENT_ERRORS as exc:
            if attempt == max_attempts:
                raise
            delay = retry_delay * 2 ** (attempt - 1)
            logger.warning(
                f"Transient error fetching '{request.slug}' "
                f"(attempt {attempt}/{max_attempts}): {type(exc).__name__}. "
                f"Retrying in {delay:.0f}s"
            )
            time.sleep(delay)

    raise AssertionError("unreachable")  # pragma: no cover


def run_request(
    request: Request,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    retry_delay: float = DEFAULT_RETRY_DELAY,
) -> RequestResult:
    """
    Fetch a request and classify the result.

    A search that matches nothing is reported but is not a failure,
    because a request may legitimately have no data for the configured facets.
    Any other error is a failure:
    the datasets exist but could not be retrieved.

    Returns
    -------
    :
        The outcome of the request.
    """
    logger.info(f"Processing request: {request.slug}")
    try:
        datasets = fetch_with_retry(request, max_attempts=max_attempts, retry_delay=retry_delay)
    except DatasetResolutionError:
        logger.warning(f"No datasets matched request '{request.slug}'")
        return RequestResult(request.slug, Outcome.EMPTY)
    except Exception as exc:
        logger.exception(f"Failed to fetch request '{request.slug}'")
        return RequestResult(request.slug, Outcome.FAILED, error=f"{type(exc).__name__}: {exc}")

    logger.info(f"Fetched {len(datasets)} datasets for request '{request.slug}'")
    return RequestResult(request.slug, Outcome.OK, dataset_count=len(datasets))


def report(results: list[RequestResult]) -> None:
    """
    Log a summary of every request that was attempted
    """
    failed = [r for r in results if r.outcome is Outcome.FAILED]
    empty = [r for r in results if r.outcome is Outcome.EMPTY]
    ok = [r for r in results if r.outcome is Outcome.OK]

    logger.info(f"Summary: {len(ok)} succeeded, {len(empty)} matched no datasets, {len(failed)} failed")
    for result in empty:
        logger.warning(f"  no datasets: {result.request_id}")
    for result in failed:
        logger.error(f"  failed: {result.request_id} -- {result.error}")


_KIND_TYPES: dict[str, type[Request]] = {
    "cmip6": CMIP6Request,
    "obs4mips": Obs4MIPsRequest,
}


def main(
    request_id: str = typer.Option(
        None, help="ID of a specific request to run. If not provided, all requests will be run."
    ),
    kind: str = typer.Option(
        "all",
        help=(
            "Category of data to fetch: 'all', 'cmip6', or 'obs4mips'. "
            "Use --kind obs4mips to fetch only the observational reference data."
        ),
    ),
    remove_ensembles: bool = typer.Option(
        True,
        help=(
            "Remove ensemble members, keeping only one per model. "
            "Use --no-remove-ensembles to fetch all ensembles."
        ),
    ),
    max_attempts: int = typer.Option(
        DEFAULT_MAX_ATTEMPTS,
        help="Number of attempts per request before a transient ESGF error is treated as a failure.",
    ),
    retry_delay: float = typer.Option(
        DEFAULT_RETRY_DELAY,
        help="Seconds to wait before retrying a transient error. Doubles on each subsequent retry.",
    ),
):
    """
    Fetch CMIP6 and Obs4MIPs datasets from ESGF.

    This script can run all predefined requests, a single request by ID, or a category of
    requests via --kind (e.g. --kind obs4mips for just the observational reference data).
    By default, only one ensemble member per model is fetched, but this can be changed
    with the --no-remove-ensembles flag.

    Exits with a non-zero status if any request failed, so that an unattended bulk fetch
    does not silently report success. A request that matches no datasets is not a failure.
    """
    if request_id:
        # Find and run the specific request
        matching_requests = [req for req in requests if req.slug == request_id]
        if not matching_requests:
            logger.error(f"No request found with ID '{request_id}'")
            logger.info("Available request IDs:")
            for req in requests:
                logger.info(f"  - {req.slug}")
            raise typer.Exit(1)

        logger.info(f"Running single request: {request_id}")
        selected = matching_requests
    elif kind == "all":
        selected = requests
    elif kind in _KIND_TYPES:
        selected = [req for req in requests if isinstance(req, _KIND_TYPES[kind])]
    else:
        logger.error(f"Unknown kind '{kind}'. Valid options: all, {', '.join(_KIND_TYPES)}")
        raise typer.Exit(1)

    if not request_id:
        logger.info(f"Running {len(selected)} '{kind}' requests...")
    if not remove_ensembles:
        logger.info("Fetching all ensemble members")
        selected = [copy.copy(request) for request in selected]
        for request in selected:
            request.remove_ensembles = False

    results = [
        run_request(
            request,
            max_attempts=max_attempts,
            retry_delay=retry_delay,
        )
        for request in selected
    ]
    report(results)

    if any(result.outcome is Outcome.FAILED for result in results):
        raise typer.Exit(1)


# joblib.Parallel(n_jobs=2)(joblib.delayed(run_request)(request) for request in requests)

if __name__ == "__main__":
    typer.run(main)
