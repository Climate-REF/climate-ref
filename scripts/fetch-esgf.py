"""
CLI tool for fetching the required CMIP6 and Obs4MIPs datasets from ESGF.

This script can either run all predefined requests or a specific request by ID.
By default, only one ensemble member per model is fetched to reduce the total data volume.
This can be changed with the --no-remove-ensembles flag.

This fetches about 3TB of datasets into the default location for intake esgf.
This can be adjusted via `~/.config/intake-esgf/config.yaml`.
"""

import enum
import time
from pathlib import Path

import intake_esgf
import typer
from attrs import define
from intake_esgf.exceptions import (
    DatasetLoadError,
    GlobusTransferError,
    NoSearchResults,
    StalledDownload,
)
from loguru import logger

PathDict = dict[str, list[Path]]

TRANSIENT_ERRORS = (DatasetLoadError, StalledDownload, GlobusTransferError)
"""
Errors raised when the search succeeded but the data could not be retrieved.

ESGF data nodes intermittently return 5xx responses or stall mid-transfer,
so these are worth retrying before giving up.
A `NoSearchResults` is deliberately excluded:
it means the request matched nothing, which retrying cannot change.
"""

DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_RETRY_DELAY = 5.0


@define
class CMIP6Request:
    """
    A set of CMIP6 data that will be fetched from ESGF
    """

    id: str
    facets: dict[str, str | tuple[str, ...] | list[str]]

    def fetch(self, remove_ensembles: bool = True) -> PathDict:
        """
        Fetch CMIP6 data from the ESGF catalog.

        Parameters
        ----------
        remove_ensembles : bool, default True
            Whether to remove ensemble members, keeping only one per model.
            If False, all ensemble members will be included.

        Returns
        -------
        :
            Mapping of dataset key to the local paths of the downloaded files.

        Raises
        ------
        NoSearchResults
            The search matched no datasets.
        IntakeESGFException
            The datasets were found but could not be retrieved.
        """
        catalog = intake_esgf.ESGFCatalog()
        search_parameters = {
            "project": "CMIP6",
            "frequency": ["mon", "fx"],
            **self.facets,
        }
        logger.debug(f"Fetching CMIP6 data: {search_parameters}")
        cmip6_data = catalog.search(**search_parameters)
        if remove_ensembles:
            cmip6_data = cmip6_data.remove_ensembles()
        return cmip6_data.to_path_dict()


@define
class Obs4MIPsRequest:
    """
    A set of Obs4MIPs data that will be fetched from ESGF
    """

    id: str
    facets: dict[str, str | tuple[str, ...] | list[str]]

    def fetch(self, remove_ensembles: bool = True) -> PathDict:
        """
        Fetch Obs4MIPs data from the ESGF catalog.

        Parameters
        ----------
        remove_ensembles : bool, default True
            Ignored as Obs4MIPs data does not have ensembles.

        Returns
        -------
        :
            Mapping of dataset key to the local paths of the downloaded files.

        Raises
        ------
        NoSearchResults
            The search matched no datasets.
        IntakeESGFException
            The datasets were found but could not be retrieved.
        """
        catalog = intake_esgf.ESGFCatalog()
        search_parameters = {
            "project": "obs4MIPs",
            "frequency": ["mon", "fx"],
            **self.facets,
        }
        logger.info(f"Fetching Obs4MIPs data: {search_parameters}")
        obs_data = catalog.search(**search_parameters)
        return obs_data.to_path_dict(minimal_keys=False)


Request = CMIP6Request | Obs4MIPsRequest

# TODO use the data requirements from the diagnostics directly
requests: list[Request] = [
    CMIP6Request(
        id="esmvaltool-climate-at-global-warmings-levels",
        facets=dict(
            variable_id=["pr", "tas"],
            experiment_id=["ssp126", "ssp245", "ssp370", "ssp585", "historical"],
            table_id="Amon",
        ),
    ),
    # ESMValTool Cloud radiative effects
    CMIP6Request(
        id="esmvaltool-cloud-radiative-effects",
        facets=dict(
            variable_id=["rlut", "rlutcs", "rsut", "rsutcs"],
            experiment_id="historical",
            table_id="Amon",
        ),
    ),
    # ESMValTool cloud scatterplots
    CMIP6Request(
        id="esmvaltool-cloud-scatterplots-cmip6",
        facets=dict(
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
    Obs4MIPsRequest(
        id="esmvaltool-cloud-scatterplots-obs4mips",
        facets=dict(
            source_id="ERA-5",
            variable_id="ta",
        ),
    ),
    # ESMValTool ECS data
    CMIP6Request(
        id="esmvaltool-ecs",
        facets=dict(
            variable_id=["rlut", "rsdt", "rsut", "tas"],
            experiment_id=["abrupt-4xCO2", "piControl"],
            table_id="Amon",
        ),
    ),
    # ESMValTool ENSO data
    CMIP6Request(
        id="esmvaltool-enso",
        facets=dict(
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
    CMIP6Request(
        id="esmvaltool-fire",
        facets=dict(
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
    CMIP6Request(
        id="esmvaltool-historical-cmip6",
        facets=dict(
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
    Obs4MIPsRequest(
        id="esmvaltool-historical-obs4mips",
        facets=dict(
            source_id="ERA-5",
            variable_id=[
                "psl",
                "tas",
                "ua",
            ],
        ),
    ),
    # ESMValTool TCR data
    CMIP6Request(
        id="esmvaltool-tcr",
        facets=dict(
            variable_id=["tas"],
            experiment_id=["1pctCO2", "piControl"],
            table_id="Amon",
        ),
    ),
    # ESMValTool TCRE data
    CMIP6Request(
        id="esmvaltool-tcre",
        facets=dict(
            variable_id=["fco2antt", "tas"],
            experiment_id=["esm-1pctCO2", "esm-piControl"],
            table_id="Amon",
        ),
    ),
    # ESMValTool ZEC data
    CMIP6Request(
        id="esmvaltool-zec",
        facets=dict(
            variable_id=["areacella", "tas"],
            experiment_id=["1pctCO2", "esm-1pct-brch-1000PgC"],
            table_id="Amon",
        ),
    ),
    # ESMValTool Sea Ice Area Seasonal Cycle data
    CMIP6Request(
        id="esmvaltool-sea-ice-area-seasonal-cycle",
        facets=dict(
            variable_id=["areacello", "siconc"],
            experiment_id=["historical"],
        ),
    ),
    # ESMValTool Ozone
    CMIP6Request(
        id="esmvaltool-ozone",
        facets=dict(
            variable_id=["toz", "o3"],
            experiment_id=["historical"],
            table_id="AERmon",
        ),
    ),
    Obs4MIPsRequest(
        id="esmvaltool-ozone-obs4mips",
        facets=dict(
            source_id="C3S-GTO-ECV-9-0",
            variable_id="toz",
        ),
    ),
    # ILAMB data
    CMIP6Request(
        id="ilamb-data",
        facets=dict(
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
    Obs4MIPsRequest(
        id="ilamb-lai-obs4mips",
        facets=dict(
            source_id="NOAA-NCEI-LAI-AVHRR-5-0",
            variable_id="lai",
        ),
    ),
    # IOMB data
    CMIP6Request(  # Already provided by the ESMValTool ENSO request.
        id="iomb-data",
        facets=dict(
            variable_id=["areacello", "tos"],
            experiment_id=["historical"],
        ),
    ),
    CMIP6Request(
        id="iomb-data-2",
        facets=dict(
            variable_id=["sftof", "sos", "msftmz"],
            experiment_id=["historical"],
        ),
    ),
    # Large 4D ocean datasets
    # A small set of models are fetched here for now
    CMIP6Request(
        id="iomb-data-large",
        facets=dict(
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
    CMIP6Request(
        id="pmp-modes-of-variability",
        facets=dict(
            variable_id=["areacella", "ts", "psl"],
            experiment_id=["historical", "hist-GHG"],
            table_id="Amon",
        ),
    ),
    # PMP ENSO reference data.
    # One request per source_id: the ESGF search treats source_id and variable_id as an
    # intersection, so a combined request would ask each source for variables it does not have.
    Obs4MIPsRequest(
        id="pmp-enso-gpcp-obs4mips",
        facets=dict(
            source_id="GPCP-Monthly-3-2",
            variable_id="pr",
        ),
    ),
    Obs4MIPsRequest(
        id="pmp-enso-tropflux-obs4mips",
        facets=dict(
            source_id="TropFlux-1-0",
            variable_id=["hfls", "hfss", "tauu", "ts"],
        ),
    ),
    Obs4MIPsRequest(
        id="pmp-enso-ceres-obs4mips",
        facets=dict(
            source_id="CERES-EBAF-4-2",
            variable_id=["rlds", "rlus", "rsds", "rsus"],
        ),
    ),
    # Shared by PMP ENSO, the PMP ts modes of variability, and the example global-sst-bias.
    Obs4MIPsRequest(
        id="pmp-hadisst-obs4mips",
        facets=dict(
            source_id="HadISST-1-1",
            variable_id="ts",
        ),
    ),
    # The PMP psl modes of variability request source_id "20CR", which is the obs4REF spelling.
    Obs4MIPsRequest(
        id="pmp-modes-20cr-obs4mips",
        facets=dict(
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
    remove_ensembles: bool = True,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    retry_delay: float = DEFAULT_RETRY_DELAY,
) -> PathDict:
    """
    Fetch a request, retrying transient ESGF access failures with an exponential backoff.

    Only the errors in `TRANSIENT_ERRORS` are retried.
    Everything else, including `NoSearchResults`, is raised immediately.

    Parameters
    ----------
    request
        The request to fetch.
    remove_ensembles
        Passed through to the request.
    max_attempts
        Total number of attempts, including the first.
    retry_delay
        Seconds to wait before the second attempt. Doubles on each subsequent retry.

    Returns
    -------
    :
        Mapping of dataset key to the local paths of the downloaded files.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            return request.fetch(remove_ensembles=remove_ensembles)
        except TRANSIENT_ERRORS as exc:
            if attempt == max_attempts:
                raise
            delay = retry_delay * 2 ** (attempt - 1)
            logger.warning(
                f"Transient error fetching '{request.id}' "
                f"(attempt {attempt}/{max_attempts}): {type(exc).__name__}. "
                f"Retrying in {delay:.0f}s"
            )
            time.sleep(delay)

    raise AssertionError("unreachable")  # pragma: no cover


def run_request(
    request: Request,
    remove_ensembles: bool = True,
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
    logger.info(f"Processing request: {request.id}")
    try:
        paths = fetch_with_retry(
            request,
            remove_ensembles=remove_ensembles,
            max_attempts=max_attempts,
            retry_delay=retry_delay,
        )
    except NoSearchResults:
        logger.warning(f"No datasets matched request '{request.id}'")
        return RequestResult(request.id, Outcome.EMPTY)
    except Exception as exc:
        logger.exception(f"Failed to fetch request '{request.id}'")
        return RequestResult(request.id, Outcome.FAILED, error=f"{type(exc).__name__}: {exc}")

    logger.info(f"Fetched {len(paths)} datasets for request '{request.id}'")
    return RequestResult(request.id, Outcome.OK, dataset_count=len(paths))


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
        matching_requests = [req for req in requests if req.id == request_id]
        if not matching_requests:
            logger.error(f"No request found with ID '{request_id}'")
            logger.info("Available request IDs:")
            for req in requests:
                logger.info(f"  - {req.id}")
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

    results = [
        run_request(
            request,
            remove_ensembles=remove_ensembles,
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
