from pathlib import Path
from typing import Any

import dask.config
import ilamb3
import ilamb3.regions as ilr
import pandas as pd
import pooch
import xarray as xr
from ilamb3 import run

from climate_ref_core.constraints import AddSupplementaryDataset, RequireFacets
from climate_ref_core.dataset_registry import dataset_registry_manager
from climate_ref_core.datasets import FacetFilter, SourceDatasetType
from climate_ref_core.diagnostics import (
    DataRequirement,
    Diagnostic,
    ExecutionDefinition,
    ExecutionResult,
)
from climate_ref_core.esgf import CMIP6Request, RegistryRequest
from climate_ref_core.metric_values.typing import SeriesMetricValue
from climate_ref_core.pycmec.metric import CMECMetric
from climate_ref_core.pycmec.output import CMECOutput, OutputCV
from climate_ref_core.testing import TestCase, TestDataSpecification
from climate_ref_ilamb.datasets import (
    registry_to_collection,
)


def _parse_test_cases_from_config(
    test_cases_config: list[dict[str, Any]] | None,
) -> TestDataSpecification | None:
    """
    Parse test cases configuration from YAML into a TestDataSpecification.

    Parameters
    ----------
    test_cases_config
        List of test case configurations from YAML, or None if not specified

    Returns
    -------
        TestDataSpecification with parsed test cases, or None if no test cases specified
    """
    if not test_cases_config:
        return None

    parsed_cases: list[TestCase] = []
    for tc_config in test_cases_config:
        # Parse CMIP6 requests
        requests: list[CMIP6Request | RegistryRequest] = []
        for req_config in tc_config.get("cmip6_requests", []):
            time_span = req_config.get("time_span")
            if time_span is not None:
                time_span = tuple(time_span)

            # Convert list values to tuples for consistency
            facets = {}
            for k, v in req_config.items():
                if k in ("slug", "time_span"):
                    continue
                facets[k] = tuple(v) if isinstance(v, list) else v

            requests.append(
                CMIP6Request(
                    slug=req_config["slug"],
                    facets=facets,
                    time_span=time_span,
                )
            )

        # Parse registry requests
        for req_config in tc_config.get("registry_requests", []):
            time_span = req_config.get("time_span")
            if time_span is not None:
                time_span = tuple(time_span)

            # Convert list values to tuples for consistency
            facets = {}
            for k, v in req_config.items():
                if k in ("slug", "registry_name", "source_type", "time_span"):
                    continue
                facets[k] = tuple(v) if isinstance(v, list) else v

            requests.append(
                RegistryRequest(
                    slug=req_config["slug"],
                    registry_name=req_config["registry_name"],
                    facets=facets,
                    source_type=req_config.get("source_type", "obs4REF"),
                    time_span=time_span,
                )
            )

        parsed_cases.append(
            TestCase(
                name=tc_config["name"],
                description=tc_config.get("description", ""),
                requests=tuple(requests) if requests else None,
                datasets_file=tc_config.get("datasets_file"),
            )
        )

    return TestDataSpecification(test_cases=tuple(parsed_cases))


def format_cmec_output_bundle(
    dataset: pd.DataFrame,
    dimensions: list[str],
    metadata_columns: list[str],
    value_column: str = "value",
) -> dict[str, Any]:
    """
    Create a CMEC output bundle for the dataset.

    Parameters
    ----------
    dataset
        Processed dataset
    dimensions
        The dimensions of the dataset (e.g., ["source_id", "member_id", "region"])
    metadata_columns
        The columns to be used as metadata (e.g., ["Description", "LongName"])
    value_column
        The column containing the values

    Returns
    -------
        A CMEC output bundle ready to be written to disk
    """
    # Validate that all required columns exist
    required_columns = set(dimensions) | {value_column} | set(metadata_columns)
    missing_columns = required_columns - set(dataset.columns)
    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    # Build the dimensions section
    dimensions_dict: dict[str, dict[str, dict[str, str]]] = {}

    # For each dimension, create a dictionary of unique values and their metadata
    for dim in dimensions:
        unique_values = dataset[dim].unique()
        dim_dict: dict[str, dict[str, str]] = {}

        for val in unique_values:
            # Get the row for this dimension value

            dim_dict[str(val)] = {}

            if dim == dimensions[-1]:
                # If this is the last dimension, add the value column to the metadata

                dim_dict[str(val)] = dataset[dataset[dim] == val].iloc[0][metadata_columns].to_dict()

        dimensions_dict[dim] = dim_dict

    # Build the results section - create nested structure based on dimensions
    def nest_results(df: pd.DataFrame, dims: list[str]) -> dict[str, Any] | float:
        if not dims:
            return float(df[value_column].iloc[0].item())

        current_dim = dims[0]
        remaining_dims = dims[1:]

        return {
            str(group_name): nest_results(group_df, remaining_dims)
            for group_name, group_df in df.groupby(current_dim)
        }

    results = nest_results(dataset, list(dimensions))

    return {"DIMENSIONS": {"json_structure": list(dimensions), **dimensions_dict}, "RESULTS": results}


def _build_cmec_bundle(df: pd.DataFrame) -> dict[str, Any]:
    """
    Build a CMEC bundle from information in the dataframe.

    """
    # TODO: Handle the reference data
    # reference_df = df[df["source"] == "Reference"]
    model_df = df[df["source"] != "Reference"]

    # Strip out units from the name (available in the attributes)
    extracted_source = model_df.name.str.extract(r"(.*)\s\[.*\]")
    model_df.loc[:, "name"] = extracted_source[0]

    model_df = model_df.rename(
        columns={
            "analysis": "metric",
            "name": "statistic",
        }
    )

    # Convert the value column to numeric, coercing errors to NaN
    model_df.loc[:, "value"] = pd.to_numeric(model_df["value"], errors="coerce")
    model_df = model_df.astype({"value": "float64"})

    dimensions = ["experiment_id", "source_id", "member_id", "grid_label", "region", "metric", "statistic"]
    attributes = ["type", "units"]

    bundle = format_cmec_output_bundle(
        model_df,
        dimensions=dimensions,
        metadata_columns=attributes,
        value_column="value",
    )

    ilamb_regions = ilr.Regions()
    for region, region_info in bundle["DIMENSIONS"]["region"].items():
        if region == "None":
            region_info["LongName"] = "None"
            region_info["Description"] = "Reference data extents"
            region_info["Generator"] = "N/A"
        else:
            region_info["LongName"] = ilamb_regions.get_name(region)
            region_info["Description"] = ilamb_regions.get_name(region)
            region_info["Generator"] = ilamb_regions.get_source(region)

    return bundle


def _set_ilamb3_options(registry: pooch.Pooch, registry_file: str) -> None:
    """
    Set options for ILAMB based on which registry file is being used.
    """
    ilamb3.conf.reset()  # type: ignore
    ilamb_regions = ilr.Regions()
    if registry_file == "ilamb":
        ilamb_regions.add_netcdf(registry.fetch("ilamb/regions/GlobalLand.nc"))
        ilamb_regions.add_netcdf(registry.fetch("ilamb/regions/Koppen_coarse.nc"))
        ilamb3.conf.set(regions=["global", "tropical"])
    # REF's data requirement correctly will add measure data from another
    # ensemble, but internally I also groupby. Since REF is only giving 1
    # source_id/member_id/grid_label at a time, relax the groupby option here so
    # these measures are part of the dataframe in ilamb3.
    ilamb3.conf.set(comparison_groupby=["source_id", "grid_label"])


def _load_csv_and_merge(output_directory: Path) -> pd.DataFrame:
    """
    Load individual csv scalar data and merge into a dataframe.
    """
    df = pd.concat(
        [pd.read_csv(f, keep_default_na=False, na_values=["NaN"]) for f in output_directory.glob("*.csv")]
    ).drop_duplicates(subset=["source", "region", "analysis", "name"])
    return df


class ILAMBStandard(Diagnostic):
    """
    Apply the standard ILAMB analysis with respect to a given reference dataset.
    """

    def __init__(
        self,
        registry_file: str,
        metric_name: str,
        sources: dict[str, str],
        **ilamb_kwargs: Any,
    ):
        # Setup the diagnostic
        if len(sources) != 1:
            raise ValueError("Only single source ILAMB diagnostics have been implemented.")
        self.variable_id = next(iter(sources.keys()))
        if "sources" not in ilamb_kwargs:  # pragma: no cover
            ilamb_kwargs["sources"] = sources
        if "relationships" not in ilamb_kwargs:
            ilamb_kwargs["relationships"] = {}

        # Extract and parse test_cases before passing to ilamb_kwargs
        test_cases_config = ilamb_kwargs.pop("test_cases", None)
        self.test_data_spec = _parse_test_cases_from_config(test_cases_config)

        self.ilamb_kwargs = ilamb_kwargs

        # REF stuff
        self.name = metric_name
        self.slug = self.name.lower().replace(" ", "-")
        self.data_requirements = (
            DataRequirement(
                source_type=SourceDatasetType.CMIP6,
                filters=(
                    FacetFilter(
                        facets={
                            "variable_id": (
                                self.variable_id,
                                *ilamb_kwargs.get("alternate_vars", []),
                                *ilamb_kwargs.get("related_vars", []),
                                *ilamb_kwargs.get("relationships", {}).keys(),
                            ),
                            "frequency": "mon",
                            "experiment_id": ("historical", "land-hist"),
                            "table_id": (
                                "AERmonZ",
                                "Amon",
                                "CFmon",
                                "Emon",
                                "EmonZ",
                                "LImon",
                                "Lmon",
                                "Omon",
                                "SImon",
                            ),
                        }
                    ),
                ),
                constraints=(
                    RequireFacets(
                        "variable_id",
                        (
                            self.variable_id,
                            *ilamb_kwargs.get("alternate_vars", []),
                            *ilamb_kwargs.get("related_vars", []),
                        ),
                        operator="any",
                    ),
                    *(
                        [
                            RequireFacets(
                                "variable_id",
                                required_facets=tuple(ilamb_kwargs.get("relationships", {}).keys()),
                            )
                        ]
                        if "relationships" in ilamb_kwargs
                        else []
                    ),
                    *(
                        (
                            AddSupplementaryDataset.from_defaults("areacella", SourceDatasetType.CMIP6),
                            AddSupplementaryDataset.from_defaults("sftlf", SourceDatasetType.CMIP6),
                        )
                        if registry_file == "ilamb"
                        else (
                            AddSupplementaryDataset.from_defaults("volcello", SourceDatasetType.CMIP6),
                            AddSupplementaryDataset.from_defaults("areacello", SourceDatasetType.CMIP6),
                            AddSupplementaryDataset.from_defaults("sftof", SourceDatasetType.CMIP6),
                        )
                    ),
                ),
                group_by=("experiment_id", "source_id", "member_id", "grid_label"),
            ),
        )

        self.facets = (
            "experiment_id",
            "source_id",
            "member_id",
            "grid_label",
            "region",
            "metric",
            "statistic",
        )

        # Setup ILAMB data and options
        self.registry_file = registry_file
        self.registry = dataset_registry_manager[self.registry_file]
        self.ilamb_data = registry_to_collection(
            dataset_registry_manager[self.registry_file],
        )

    def execute(self, definition: ExecutionDefinition) -> None:
        """
        Run the ILAMB standard analysis.
        """
        _set_ilamb3_options(self.registry, self.registry_file)
        ref_datasets = self.ilamb_data.datasets.set_index(self.ilamb_data.slug_column)

        # Run ILAMB in a single-threaded mode to avoid issues with multithreading (#394)
        with dask.config.set(scheduler="synchronous"):
            run.run_single_block(
                self.slug,
                ref_datasets,
                definition.datasets[SourceDatasetType.CMIP6].datasets,
                definition.output_directory,
                **self.ilamb_kwargs,
            )

    def build_execution_result(self, definition: ExecutionDefinition) -> ExecutionResult:
        """
        Build the diagnostic result after running ILAMB.

        Parameters
        ----------
        definition
            The definition of the diagnostic execution

        Returns
        -------
            An execution result object
        """
        _set_ilamb3_options(self.registry, self.registry_file)
        # In ILAMB, scalars are saved in CSV files in the output directory. To
        # be compatible with the REF system we will need to add the metadata
        # that is associated with the execution group, called the selector.
        df = _load_csv_and_merge(definition.output_directory)
        selectors = definition.datasets[SourceDatasetType.CMIP6].selector_dict()

        # TODO: Fix reference data once we are using the obs4MIPs dataset
        dataset_source = self.name.split("-")[1] if "-" in self.name else "None"
        common_dimensions = {**selectors, "reference_source_id": dataset_source}
        for key, value in common_dimensions.items():
            df[key] = value
        metric_bundle = CMECMetric.model_validate(_build_cmec_bundle(df))

        # Add each png file plot to the output
        output_bundle = CMECOutput.create_template()
        for plotfile in definition.output_directory.glob("*.png"):
            relative_path = str(definition.as_relative_path(plotfile))
            caption, figure_dimensions = _caption_from_filename(plotfile, common_dimensions)

            output_bundle[OutputCV.PLOTS.value][relative_path] = {
                OutputCV.FILENAME.value: relative_path,
                OutputCV.LONG_NAME.value: caption,
                OutputCV.DESCRIPTION.value: "",
                OutputCV.DIMENSIONS.value: figure_dimensions,
            }

        # Add the html page to the output
        index_html = definition.to_output_path("index.html")
        if index_html.exists():
            relative_path = str(definition.as_relative_path(index_html))
            output_bundle[OutputCV.HTML.value][relative_path] = {
                OutputCV.FILENAME.value: relative_path,
                OutputCV.LONG_NAME.value: "Results page",
                OutputCV.DESCRIPTION.value: "Page displaying scalars and plots from the ILAMB execution.",
                OutputCV.DIMENSIONS.value: common_dimensions,
            }
            output_bundle[OutputCV.INDEX.value] = relative_path

        # Add series to the output based on the time traces we find in the
        # output files
        series = []
        for ncfile in definition.output_directory.glob("*.nc"):
            ds = xr.open_dataset(ncfile, use_cftime=True)
            for name, da in ds.items():
                # Only create series for 1d DataArray's with these dimensions
                if not (da.ndim == 1 and set(da.dims).intersection(["time", "month"])):
                    continue
                # Convert dimension values
                attrs = {
                    "units": da.attrs.get("units", ""),
                    "long_name": da.attrs.get("long_name", str(name)),
                    "standard_name": da.attrs.get("standard_name", ""),
                }
                str_name = str(name)
                index_name = str(da.dims[0])
                index = ds[index_name].values.tolist()
                if hasattr(index[0], "isoformat"):
                    index = [v.isoformat() for v in index]
                if hasattr(index[0], "calendar"):
                    attrs["calendar"] = index[0].calendar

                # Parse out some dimensions
                if ncfile.stem == "Reference":
                    dimensions = {
                        "source_id": "Reference",
                        "metric": str_name,
                    }
                else:
                    dimensions = {"metric": str_name, **common_dimensions}

                # Split the metric into metric and region if possible
                if "_" in str_name:
                    dimensions["metric"] = str_name.split("_")[0]
                    dimensions["region"] = str_name.split("_")[1]
                else:
                    dimensions["region"] = "None"

                series.append(
                    SeriesMetricValue(
                        dimensions=dimensions,
                        values=da.values.tolist(),
                        index=index,
                        index_name=index_name,
                        attributes=attrs,
                    )
                )

        return ExecutionResult.build_from_output_bundle(
            definition, cmec_output_bundle=output_bundle, cmec_metric_bundle=metric_bundle, series=series
        )


def _caption_from_filename(filename: Path, common_dimensions: dict[str, str]) -> tuple[str, dict[str, str]]:
    source, region, plot = filename.stem.split("_")
    plot_texts = {
        "bias": "bias",
        "biasscore": "bias score",
        "cycle": "annual cycle",
        "cyclescore": "annual cycle score",
        "mean": "period mean",
        "rmse": "RMSE",
        "rmsescore": "RMSE score",
        "shift": "shift in maximum month",
        "tmax": "maxmimum month",
        "trace": "regional mean",
        "taylor": "Taylor diagram",
        "distribution": "distribution",
        "response": "response",
    }
    # Name of statistics dimension in CMEC output
    plot_statistics = {
        "bias": "Bias",
        "biasscore": "Bias score",
        "cycle": "Annual cycle",
        "cyclescore": "Annual cycle score",
        "mean": "Period Mean",
        "rmse": "RMSE",
        "rmsescore": "RMSE score",
        "shift": "Shift in maximum month",
        "tmax": "Maximum month",
        "trace": "Regional mean",
        "taylor": "Taylor diagram",
        "distribution": "Distribution",
        "response": "Response",
    }
    figure_dimensions = {
        "region": region,
    }
    plot_option = None
    # Some plots have options appended with a dash (distribution-pr, response-tas)
    if "-" in plot:
        plot, plot_option = plot.split("-", 1)

    if plot not in plot_texts:
        return "", figure_dimensions

    # Build the caption
    caption = f"The {plot_texts.get(plot)}"
    if plot_option is not None:
        caption += f" of {plot_option}"
    if source != "None":
        caption += f" for {'the reference data' if source == 'Reference' else source}"
    if region.lower() != "none":
        caption += f" over the {ilr.Regions().get_name(region)} region."

    # Use the statistic dimension to determine what is being plotted
    if plot_statistics.get(plot) is not None:
        figure_dimensions["statistic"] = plot_statistics[plot]
        if plot_option is not None:
            figure_dimensions["statistic"] += f"|{plot_option}"

    # If the source is the reference we don't need some dimensions as they are not applicable
    if source == "Reference":
        figure_dimensions["source_id"] = "Reference"
    else:
        figure_dimensions = {**common_dimensions, **figure_dimensions}

    return caption, figure_dimensions
