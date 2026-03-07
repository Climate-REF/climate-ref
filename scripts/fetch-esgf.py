"""
CLI tool for fetching the required CMIP6 and Obs4MIPs datasets from ESGF.

This script can either run all predefined requests or a specific request by ID.
By default, up to 10 ensemble members per source_id are fetched to reduce the total data volume.
This can be changed with the --max-ensembles flag (0 = no limit).

This fetches about 3TB of datasets into the default location for intake esgf.
This can be adjusted via `~/.config/intake-esgf/config.yaml`.
"""

import intake_esgf
import typer
from attrs import define
from loguru import logger


@define
class CMIP6Request:
    """
    A set of CMIP6 data that will be fetched from ESGF
    """

    id: str
    facets: dict[str, str | tuple[str, ...] | list[str]]

    def fetch(self, max_members: int = 1):
        """
        Fetch CMIP6 data from the ESGF catalog and return it as a DataFrame.

        Parameters
        ----------
        max_members : int, default 10
            Maximum number of ensemble members to fetch per source_id.
            Set to 0 for no limit (fetch all ensemble members).

        Returns
        -------
        pd.DataFrame
            DataFrame containing the metadata for the CMIP6 datasets.
        """
        catalog = intake_esgf.ESGFCatalog()
        search_parameters = {
            "project": "CMIP6",
            "frequency": ["mon", "fx"],
            **self.facets,
        }
        logger.debug(f"Fetching CMIP6 data: {search_parameters}")
        try:
            cmip6_data = catalog.search(**search_parameters)
            if max_members > 0 and cmip6_data.df is not None:
                df = cmip6_data.df
                mask = df.groupby("source_id")["member_id"].transform(
                    lambda s: s.isin(s.unique()[:max_members])
                )
                cmip6_data.df = df[mask]
            return cmip6_data.to_path_dict()
        except Exception:
            logger.info(f"Error fetching CMIP6 data: {search_parameters}")
        return {}


@define
class Obs4MIPsRequest:
    """
    A set of Obs4MIPs data that will be fetched from ESGF
    """

    id: str
    facets: dict[str, str | tuple[str, ...] | list[str]]

    def fetch(self, max_members: int = 1):
        """
        Fetch Obs4MIPs data from the ESGF catalog and return it as a DataFrame.

        Parameters
        ----------
        max_members : int, default 1
            Ignored as Obs4MIPs data does not have ensembles.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the metadata for the Obs4MIPs datasets.
        """
        catalog = intake_esgf.ESGFCatalog()
        search_parameters = {
            "project": "obs4MIPs",
            "frequency": ["mon", "fx"],
            **self.facets,
        }
        logger.info(f"Fetching Obs4MIPs data: {search_parameters}")
        try:
            obs_data = catalog.search(**search_parameters)
            return obs_data.to_path_dict(minimal_keys=False)
        except Exception:
            logger.exception(f"Error fetching Obs4MIPs data: {search_parameters}")
        return {}


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
]


def run_request(request: Request, max_members: int = 1):
    """
    Fetch and log the results of a request
    """
    print(f"Processing request: {request.id}")
    df = request.fetch(max_members=max_members)
    print(f"{len(df)} datasets")
    print("\n")


def main(
    request_id: str = typer.Option(
        None, help="ID of a specific request to run. If not provided, all requests will be run."
    ),
    max_members: int = typer.Option(
        1,
        help=(
            "Maximum number of ensemble members to fetch per source_id. "
            "Set to 0 to fetch all ensemble members."
        ),
    ),
):
    """
    Fetch CMIP6 datasets from ESGF.

    This script can either run all predefined requests or a specific request by ID.
    By default, up to 1 ensemble members per source_id are fetched, but this can be
    changed with --max-ensembles (use 0 for no limit).
    """
    if request_id:
        # Find and run the specific request
        matching_requests = [req for req in requests if req.id == request_id]
        if not matching_requests:
            logger.info(f"Error: No request found with ID '{request_id}'")
            logger.info("Available request IDs:")
            for req in requests:
                logger.info(f"  - {req.id}")
            raise typer.Exit(1)

        logger.info(f"Running single request: {request_id}")
        limit = max_members if max_members > 0 else "unlimited"
        logger.info(f"Max ensemble members per source_id: {limit}")
        run_request(matching_requests[0], max_members=max_members)
    else:
        logger.info("Running all requests...")
        limit = max_members if max_members > 0 else "unlimited"
        logger.info(f"Max ensemble members per source_id: {limit}")
        for request in requests:
            run_request(request, max_members=max_members)


# joblib.Parallel(n_jobs=2)(joblib.delayed(run_request)(request) for request in requests)

if __name__ == "__main__":
    typer.run(main)
