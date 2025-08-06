"""
Basic script for fetching a single ensemble member for each of the diagnostics of interest

This fetches about 660GB of datasets into the default location for intake esgf.
"""

import intake_esgf
from attrs import define


@define
class CMIP6Request:
    """
    A set of CMIP6 data that will be fetched from ESGF
    """

    id: str
    facets: dict[str, str | tuple[str, ...] | list[str]]

    def fetch(self):
        """
        Fetch CMIP6 data from the ESGF catalog and return it as a DataFrame.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the metadata for the CMIP6 datasets.
        """
        catalog = intake_esgf.ESGFCatalog()
        try:
            cmip6_data = catalog.search(
                project="CMIP6",
                frequency=["mon", "fx"],
                **self.facets,
            )
            return cmip6_data.remove_ensembles().to_path_dict()
        except Exception as e:
            print(e)
        return []


requests = [
    CMIP6Request(
        id="esmvaltool-climate-at-global-warmings-levels",
        facets=dict(
            variable_id=["pr", "tas"],
            experiment_id=["ssp126", "ssp245", "ssp370", "ssp585", "historical"],
        ),
    ),
    # ESMValTool Cloud radiative effects
    CMIP6Request(
        id="esmvaltool-cloud-radiative-effects",
        facets=dict(
            variable_id=["rlut", "rlutcs", "rsut", "rsutcs"],
            experiment_id="historical",
        ),
    ),
    # ESMValTool ECS data
    CMIP6Request(
        id="esmvaltool-ecs",
        facets=dict(
            variable_id=["rlut", "rsdt", "rsut", "tas"],
            experiment_id=["abrupt-4xCO2", "piControl"],
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
        ),
    ),
    # ESMValTool TCR data
    CMIP6Request(
        id="esmvaltool-tcr",
        facets=dict(
            variable_id=["tas"],
            experiment_id=["1pctCO2", "piControl"],
        ),
    ),
    # ESMValTool TCRE data
    CMIP6Request(
        id="esmvaltool-tcre",
        facets=dict(
            variable_id=["fco2antt", "tas"],
            experiment_id=["esm-1pctCO2", "esm-piControl"],
        ),
    ),
    # ESMValTool ZEC data
    CMIP6Request(
        id="esmvaltool-zec",
        facets=dict(
            variable_id=["areacella", "tas"],
            experiment_id=["1pctCO2", "esm-1pct-brch-1000PgC"],
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
    # PMP modes of variability data
    CMIP6Request(
        id="pmp-modes-of-variability",
        facets=dict(
            variable_id=["areacella", "ts", "psl"],
            experiment_id=["historical", "hist-GHG"],
        ),
    ),
]


def run_request(request: CMIP6Request):
    """
    Fetch and log the results of a request
    """
    print(f"Processing request: {request.id}")
    df = request.fetch()
    print(f"{len(df)} datasets")
    print("\n")


for request in requests:
    run_request(request)
# joblib.Parallel(n_jobs=2)(joblib.delayed(run_request)(request) for request in requests)
