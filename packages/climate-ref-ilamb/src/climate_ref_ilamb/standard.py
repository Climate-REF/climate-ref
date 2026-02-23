from pathlib import Path
from typing import Any

import dask.config
import ilamb3
import ilamb3.regions as ilr
import pandas as pd
import pooch
import xarray as xr
from ilamb3 import run
from loguru import logger

from climate_ref_core.cmip6_to_cmip7 import get_dreq_entry
from climate_ref_core.constraints import AddSupplementaryDataset, RequireFacets
from climate_ref_core.dataset_registry import dataset_registry_manager
from climate_ref_core.datasets import ExecutionDatasetCollection, FacetFilter, SourceDatasetType
from climate_ref_core.diagnostics import (
    DataRequirement,
    Diagnostic,
    ExecutionDefinition,
    ExecutionResult,
)
from climate_ref_core.esgf import CMIP6Request, CMIP7Request
from climate_ref_core.metric_values.typing import SeriesMetricValue
from climate_ref_core.pycmec.metric import CMECMetric
from climate_ref_core.pycmec.output import CMECOutput, OutputCV
from climate_ref_core.testing import TestCase, TestDataSpecification
from climate_ref_ilamb.datasets import (
    registry_to_collection,
)

# CMIP6 tables to search, ordered by likelihood for land vs ocean registries
_LAND_TABLES = ("Lmon", "Emon", "LImon", "Amon", "CFmon", "AERmonZ", "EmonZ")
_OCEAN_TABLES = ("Omon", "SImon", "Ofx")


def _get_branded_variable(
    variable_ids: tuple[str, ...],
    registry_file: str,
) -> tuple[str, ...]:
    """
    Look up CMIP7 branded variable names for a set of CMIP6 variable IDs.

    Tries the most relevant CMIP6 tables based on registry type (land vs ocean).
    Variables without a Data Request mapping are silently skipped.

    Parameters
    ----------
    variable_ids
        CMIP6 variable IDs to look up
    registry_file
        ILAMB registry name ("ilamb", "ilamb-test", or "iomb")

    Returns
    -------
    :
        Branded variable names found in the Data Request
    """
    tables = _LAND_TABLES if registry_file in ("ilamb", "ilamb-test") else _OCEAN_TABLES

    branded: list[str] = []
    for var_id in variable_ids:
        found = False
        for table in tables:
            try:
                entry = get_dreq_entry(table, var_id)
                branded.append(entry.branded_variable)
                found = True
            except KeyError:
                continue
        if not found:
            logger.debug(f"No CMIP7 branded variable name found for {var_id}")

    return tuple(branded)


def _get_cmip_source_type(
    datasets: ExecutionDatasetCollection,
) -> SourceDatasetType:
    """Get the CMIP source type (CMIP6 or CMIP7) from available datasets."""
    if SourceDatasetType.CMIP7 in datasets:
        return SourceDatasetType.CMIP7
    return SourceDatasetType.CMIP6


def _get_selectors(datasets: ExecutionDatasetCollection) -> dict[str, str]:
    """
    Get selector dict from the CMIP source, normalizing CMIP7 facet names.

    Renames ``variant_label`` to ``member_id`` so the CMEC output bundle
    uses a consistent dimension name regardless of input MIP era.
    """
    cmip_source = _get_cmip_source_type(datasets)
    selectors = datasets[cmip_source].selector_dict()
    if "variant_label" in selectors and "member_id" not in selectors:
        selectors["member_id"] = selectors.pop("variant_label")
    return selectors


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

        # Allow per-diagnostic override of the test source_id
        # (not all variables are available from CanESM5)
        test_source_id = ilamb_kwargs.pop("test_source_id", "CanESM5")

        self.ilamb_kwargs = ilamb_kwargs

        # REF stuff
        self.name = metric_name
        self.slug = self.name.lower().replace(" ", "-")

        # Collect all variable IDs used by this diagnostic
        all_variable_ids = (
            self.variable_id,
            *ilamb_kwargs.get("alternate_vars", []),
            *ilamb_kwargs.get("related_vars", []),
            *ilamb_kwargs.get("relationships", {}).keys(),
        )

        # Variables that the primary diagnostic requires (not relationship vars)
        primary_variable_ids = (
            self.variable_id,
            *ilamb_kwargs.get("alternate_vars", []),
            *ilamb_kwargs.get("related_vars", []),
        )

        # Look up CMIP7 branded variable names
        branded_variables = _get_branded_variable(all_variable_ids, registry_file)

        # Determine realm/region for CMIP7 filter based on registry
        is_land = registry_file in ("ilamb", "ilamb-test")

        # CMIP6 data requirement
        cmip6_requirement = DataRequirement(
            source_type=SourceDatasetType.CMIP6,
            filters=(
                FacetFilter(
                    facets={
                        "variable_id": all_variable_ids,
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
                    primary_variable_ids,
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
                    if is_land
                    else (
                        AddSupplementaryDataset.from_defaults("volcello", SourceDatasetType.CMIP6),
                        AddSupplementaryDataset.from_defaults("areacello", SourceDatasetType.CMIP6),
                        AddSupplementaryDataset.from_defaults("sftof", SourceDatasetType.CMIP6),
                    )
                ),
            ),
            group_by=("experiment_id", "source_id", "member_id", "grid_label"),
        )

        # CMIP7 data requirement
        cmip7_requirement = DataRequirement(
            source_type=SourceDatasetType.CMIP7,
            filters=(
                FacetFilter(
                    facets={
                        "branded_variable": branded_variables,
                        "frequency": "mon",
                        "experiment_id": ("historical", "land-hist"),
                        "region": "glb",
                    }
                ),
            ),
            constraints=(
                RequireFacets(
                    "variable_id",
                    primary_variable_ids,
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
                        AddSupplementaryDataset.from_defaults("areacella", SourceDatasetType.CMIP7),
                        AddSupplementaryDataset.from_defaults("sftlf", SourceDatasetType.CMIP7),
                    )
                    if is_land
                    else (
                        AddSupplementaryDataset.from_defaults("volcello", SourceDatasetType.CMIP7),
                        AddSupplementaryDataset.from_defaults("areacello", SourceDatasetType.CMIP7),
                        AddSupplementaryDataset.from_defaults("sftof", SourceDatasetType.CMIP7),
                    )
                ),
            ),
            group_by=("experiment_id", "source_id", "variant_label", "grid_label"),
        )

        self.data_requirements = (
            (cmip6_requirement,),
            (cmip7_requirement,),
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

        # Build test data spec for CMIP6 and CMIP7 test cases
        supplementary_vars: list[str]
        if is_land:
            supplementary_vars = ["areacella", "sftlf"]
        else:
            supplementary_vars = ["volcello", "areacello", "sftof"]
        test_variable_ids = sorted(set(all_variable_ids) | set(supplementary_vars))

        # Build branded variable names for the test spec (include supplementary)
        test_branded_names = sorted(
            set(
                _get_branded_variable(
                    tuple(test_variable_ids),
                    registry_file,
                )
            )
        )

        self.test_data_spec = TestDataSpecification(
            test_cases=(
                TestCase(
                    name="cmip6",
                    description="Test with CMIP6 data.",
                    requests=(
                        CMIP6Request(
                            slug="cmip6",
                            facets={
                                "experiment_id": "historical",
                                "source_id": test_source_id,
                                "variable_id": test_variable_ids,
                                "frequency": ["fx", "mon"],
                            },
                            remove_ensembles=True,
                        ),
                    ),
                ),
                TestCase(
                    name="cmip7",
                    description="Test with CMIP7 data.",
                    requests=(
                        CMIP7Request(
                            slug="cmip7",
                            facets={
                                "experiment_id": "historical",
                                "source_id": test_source_id,
                                "variable_id": test_variable_ids,
                                "branded_variable": test_branded_names,
                                "variant_label": "r1i1p1f1",
                                "frequency": ["fx", "mon"],
                                "region": "glb",
                            },
                            remove_ensembles=True,
                        ),
                        # Separate request for volcello without variant_label
                        # constraint. Some models (e.g. CanESM5) publish volcello
                        # only for certain physics versions (p2), so the main
                        # request pinned to r1i1p1f1 won't find it.
                        *(
                            (
                                CMIP7Request(
                                    slug="cmip7-volcello",
                                    facets={
                                        "experiment_id": "historical",
                                        "source_id": test_source_id,
                                        "variable_id": "volcello",
                                        "frequency": "fx",
                                        "region": "glb",
                                    },
                                    remove_ensembles=True,
                                ),
                            )
                            if not is_land
                            else ()
                        ),
                    ),
                ),
            )
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

        cmip_source = _get_cmip_source_type(definition.datasets)
        model_datasets = definition.datasets[cmip_source].datasets

        # CMIP7 uses variant_label instead of member_id; ilamb3 expects member_id
        if cmip_source == SourceDatasetType.CMIP7 and "member_id" not in model_datasets.columns:
            model_datasets = model_datasets.copy()
            model_datasets["member_id"] = model_datasets["variant_label"]

        # When both a 3D primary variable (e.g. thetao) and its 2D alternate
        # (e.g. tos) are present, drop the primary so ilamb3 uses the alternate.
        # This avoids two problems with merging datasets of different time ranges:
        # 1) time_bnds gets NaN-filled for the shorter variable's missing steps,
        #    corrupting the time frequency calculation in ilamb3
        # 2) The 3D variable may not cover the reference period (e.g. thetao
        #    covers 1850-1860 but WOA2023 reference covers 2005-2014)
        # The alternate surface variable is equivalent after select_depth and
        # typically has full temporal coverage.
        alternate_vars = self.ilamb_kwargs.get("alternate_vars", [])
        if alternate_vars:
            available_alternates = [v for v in alternate_vars if v in model_datasets["variable_id"].values]
            if available_alternates and self.variable_id in model_datasets["variable_id"].values:
                model_datasets = model_datasets[model_datasets["variable_id"] != self.variable_id]

        # Run ILAMB in a single-threaded mode to avoid issues with multithreading (#394)
        with dask.config.set(scheduler="synchronous"):
            run.run_single_block(
                self.slug,
                ref_datasets,
                model_datasets,
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
        selectors = _get_selectors(definition.datasets)

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
                    metric_name, region_name = str_name.split("_", maxsplit=1)
                    dimensions["metric"] = metric_name
                    dimensions["region"] = region_name
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
