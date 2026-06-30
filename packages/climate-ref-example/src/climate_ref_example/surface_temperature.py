from pathlib import Path
from typing import Any, cast

import numpy as np
import xarray as xr
from loguru import logger

from climate_ref_core.constraints import RequireContiguousTimerange
from climate_ref_core.datasets import DatasetCollection, FacetFilter, SourceDatasetType
from climate_ref_core.diagnostics import (
    DataRequirement,
    Diagnostic,
    ExecutionDefinition,
    ExecutionResult,
)
from climate_ref_core.esgf import CMIP6Request, CMIP7Request, RegistryRequest
from climate_ref_core.pycmec.metric import CMECMetric
from climate_ref_core.pycmec.output import CMECOutput
from climate_ref_core.testing import TestCase, TestDataSpecification

# The surface-temperature variable shared by the model and the reference dataset.
# Both the CMIP `ts` field and the HadISST observational record use the CF name ``ts``.
_VARIABLE = "ts"

# Reference observations: the HadISST-1-1 sea-surface/skin-temperature record,
# distributed via the REF obs4REF registry.
_REFERENCE_SOURCE_ID = "HadISST-1-1"


def latest_version_files(collection: DatasetCollection) -> list[Path]:
    """
    Return the file paths for the most recent version in a dataset collection.

    A collection grouped only by ``source_id``/``variable_id`` can contain more than one
    dataset version (for example, the obs4REF archive ships two HadISST-1-1 ``ts``
    versions). Combining versions with overlapping time ranges would make
    :func:`global_mean_surface_temperature` ambiguous, so we deterministically keep only
    the latest version before opening the files.

    Parameters
    ----------
    collection
        The datasets selected for one source type of an execution.

    Returns
    -------
    :
        Paths belonging to the latest version present in the collection.
    """
    df = collection.datasets
    if "version" in df.columns:
        versions = sorted(str(version) for version in df["version"].dropna().unique())
        if len(versions) > 1:
            latest = versions[-1]
            logger.warning(f"Multiple versions present ({', '.join(versions)}); using the latest, {latest}.")
            df = df[df["version"].astype(str) == latest]
    return [Path(path) for path in df["path"].to_list()]


def global_mean_surface_temperature(input_files: list[Path]) -> xr.DataArray:
    """
    Calculate an annual, area-weighted global-mean surface-temperature series.

    The same calculation is applied to both the model and the reference dataset so the
    two series are directly comparable. Cells are weighted by the cosine of their
    latitude, which avoids the need for a model-specific ``areacella`` file and works
    for any regular latitude/longitude grid.

    Parameters
    ----------
    input_files
        List of input files holding the surface-temperature (``ts``) field.

        The dataset may be split across multiple files.

    Returns
    -------
    :
        Annual global-mean surface temperature, indexed by integer calendar ``year``.
    """
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    xr_ds = xr.open_mfdataset(
        input_files,
        combine="by_coords",
        chunks=None,
        decode_times=time_coder,
        data_vars="all",
        compat="no_conflicts",
    )

    da = xr_ds[_VARIABLE]

    # Cosine-of-latitude weighting approximates the grid-cell area for a regular grid.
    weights = cast(xr.DataArray, np.cos(np.deg2rad(xr_ds["lat"])))
    spatial_mean = da.weighted(weights.fillna(0)).mean(dim=["lat", "lon"], keep_attrs=True)

    # Grouping by integer year keeps the model and reference series alignable even when
    # their calendars differ (e.g. proleptic_gregorian model vs gregorian observations).
    annual_mean = spatial_mean.groupby("time.year").mean()
    annual_mean.name = _VARIABLE
    return annual_mean


def compare_model_and_reference(
    model_series: xr.DataArray,
    reference_series: xr.DataArray,
) -> xr.Dataset:
    """
    Build a comparison dataset from a model and a reference series.

    The two series are restricted to their overlapping years before the bias is computed,
    so the result is well defined even when the model run and the observational record
    cover different periods.

    Parameters
    ----------
    model_series
        Annual global-mean surface temperature for the model.
    reference_series
        Annual global-mean surface temperature for the reference dataset.

    Returns
    -------
    :
        Dataset exposing the aligned ``model``, ``reference`` and ``bias`` series, with the
        ``rmse`` and ``mean_bias`` scalar statistics stored as attributes.
    """
    common_years = np.intersect1d(model_series["year"].values, reference_series["year"].values)
    if common_years.size == 0:
        raise ValueError("The model and reference datasets do not share any overlapping years.")

    model = model_series.sel(year=common_years)
    reference = reference_series.sel(year=common_years)
    bias = (model - reference).rename("bias")

    rmse = float(np.sqrt((bias**2).mean()))
    mean_bias = float(bias.mean())

    comparison = xr.Dataset(
        {
            "model": model.rename("model"),
            "reference": reference.rename("reference"),
            "bias": bias,
        }
    )
    comparison.attrs["rmse"] = rmse
    comparison.attrs["mean_bias"] = mean_bias
    comparison.attrs["reference_source_id"] = _REFERENCE_SOURCE_ID
    return comparison


def format_cmec_output_bundle() -> dict[str, Any]:
    """
    Create a CMEC output bundle that registers the comparison series.

    Returns
    -------
        A CMEC output bundle ready to be written to disk.
    """
    cmec_output = CMECOutput.create_template()

    cmec_output["data"]["surface_temperature_comparison"] = {
        "filename": "global_mean_surface_temperature.nc",
        "long_name": "Global Mean Surface Temperature: model, reference and bias",
        "description": (
            "Annual area-weighted global-mean surface temperature for the model and the "
            "HadISST-1-1 reference dataset, together with the model-minus-reference bias."
        ),
    }

    CMECOutput.model_validate(cmec_output)
    return cmec_output


def format_cmec_metric_bundle(comparison: xr.Dataset) -> dict[str, Any]:
    """
    Create a CMEC diagnostic bundle from the model-vs-reference comparison.

    Parameters
    ----------
    comparison
        Comparison dataset produced by :func:`compare_model_and_reference`.

    Returns
    -------
        A CMEC diagnostic bundle ready to be written to disk.
    """
    cmec_metric = {
        "DIMENSIONS": {
            "json_structure": [
                "region",
                "metric",
                "statistic",
            ],
            "region": {"global": {}},
            "metric": {_VARIABLE: {}},
            "statistic": {"rmse": {}, "mean-bias": {}},
        },
        "RESULTS": {
            "global": {
                _VARIABLE: {
                    "rmse": comparison.attrs["rmse"],
                    "mean-bias": comparison.attrs["mean_bias"],
                }
            },
        },
    }

    CMECMetric.model_validate(cmec_metric)
    return cmec_metric


class GlobalMeanSurfaceTemperatureBias(Diagnostic):
    """
    Compare modelled surface temperature against observations.

    This diagnostic computes the annual, area-weighted global-mean surface temperature
    (``ts``) for a model and for the HadISST-1-1 observational record, then reports the
    root-mean-square error and mean bias between them. It is a deliberately small
    model-versus-observation example that runs in seconds on the sample data, and shows
    how a diagnostic combines a model dataset with a reference dataset.

    Supports both CMIP6 and CMIP7 model datasets via alternative data requirements.
    """

    name = "Global Mean Surface Temperature Bias"
    slug = "global-mean-surface-temperature-bias"

    # Each option is an AND-group: a model requirement combined with the observational
    # reference requirement. The diagnostic runs if either the CMIP6 or the CMIP7 model
    # data is available alongside the reference dataset.
    _reference_requirement = DataRequirement(
        source_type=SourceDatasetType.obs4MIPs,
        filters=(FacetFilter(facets={"source_id": (_REFERENCE_SOURCE_ID,), "variable_id": (_VARIABLE,)}),),
        group_by=("source_id", "variable_id"),
    )
    data_requirements = (
        # Option 1: CMIP6 model data + reference observations
        (
            DataRequirement(
                source_type=SourceDatasetType.CMIP6,
                filters=(
                    FacetFilter(
                        facets={
                            "variable_id": (_VARIABLE,),
                            "experiment_id": ("historical",),
                        }
                    ),
                ),
                group_by=("source_id", "variable_id", "experiment_id", "variant_label"),
                constraints=(RequireContiguousTimerange(group_by=("instance_id",)),),
            ),
            _reference_requirement,
        ),
        # Option 2: CMIP7 model data + reference observations
        (
            DataRequirement(
                source_type=SourceDatasetType.CMIP7,
                filters=(
                    FacetFilter(
                        facets={
                            "variable_id": (_VARIABLE,),
                            "experiment_id": ("historical",),
                        }
                    ),
                ),
                group_by=("source_id", "variable_id", "experiment_id", "variant_label"),
                constraints=(RequireContiguousTimerange(group_by=("instance_id",)),),
            ),
            _reference_requirement,
        ),
    )
    facets = (
        "source_id",
        "experiment_id",
        "variant_label",
        "reference_source_id",
        "region",
        "metric",
        "statistic",
    )

    test_data_spec = TestDataSpecification(
        test_cases=(
            TestCase(
                name="default",
                description="Historical ts from ACCESS-ESM1-5 compared against HadISST-1-1 observations",
                requests=(
                    RegistryRequest(
                        slug="surface-temperature-obs",
                        registry_name="obs4ref",
                        source_type="obs4MIPs",
                        facets={"source_id": _REFERENCE_SOURCE_ID, "variable_id": _VARIABLE},
                    ),
                    CMIP6Request(
                        slug="surface-temperature-tas",
                        facets={
                            "source_id": "ACCESS-ESM1-5",
                            "experiment_id": "historical",
                            "variable_id": _VARIABLE,
                            "member_id": "r1i1p1f1",
                            "table_id": "Amon",
                        },
                        time_span=("2000-01", "2014-12"),
                    ),
                ),
            ),
            TestCase(
                name="cmip7",
                description="CMIP7 historical ts from ACCESS-ESM1-5 compared against HadISST-1-1",
                requests=(
                    RegistryRequest(
                        slug="surface-temperature-obs-cmip7",
                        registry_name="obs4ref",
                        source_type="obs4MIPs",
                        facets={"source_id": _REFERENCE_SOURCE_ID, "variable_id": _VARIABLE},
                    ),
                    CMIP7Request(
                        slug="surface-temperature-tas-cmip7",
                        facets={
                            "source_id": "ACCESS-ESM1-5",
                            "experiment_id": "historical",
                            "variable_id": _VARIABLE,
                            "variant_label": "r1i1p1f1",
                            "table_id": "Amon",
                        },
                        time_span=("2000-01", "2014-12"),
                    ),
                ),
            ),
        ),
    )

    def _get_model_source_type(self, definition: ExecutionDefinition) -> SourceDatasetType:
        """Determine which model source type is present in the datasets."""
        if SourceDatasetType.CMIP7 in definition.datasets:
            return SourceDatasetType.CMIP7
        return SourceDatasetType.CMIP6

    def execute(self, definition: ExecutionDefinition) -> None:
        """
        Run the diagnostic.

        Parameters
        ----------
        definition
            A description of the information needed for this execution of the diagnostic.
        """
        model_source_type = self._get_model_source_type(definition)
        model_datasets = definition.datasets[model_source_type]
        reference_datasets = definition.datasets[SourceDatasetType.obs4MIPs]

        # Guard against multiple dataset versions being grouped into a single execution.
        model_series = global_mean_surface_temperature(latest_version_files(model_datasets))
        reference_series = global_mean_surface_temperature(latest_version_files(reference_datasets))

        comparison = compare_model_and_reference(model_series, reference_series)
        comparison.to_netcdf(definition.output_directory / "global_mean_surface_temperature.nc")

    def build_execution_result(self, definition: ExecutionDefinition) -> ExecutionResult:
        """
        Create a result object from the output of the diagnostic.
        """
        comparison = xr.open_dataset(definition.output_directory / "global_mean_surface_temperature.nc")

        model_source_type = self._get_model_source_type(definition)
        model_selectors = definition.datasets[model_source_type].selector_dict()
        reference_selectors = definition.datasets[SourceDatasetType.obs4MIPs].selector_dict()

        cmec_metric_bundle = CMECMetric(**format_cmec_metric_bundle(comparison)).prepend_dimensions(
            {
                "source_id": model_selectors["source_id"],
                "experiment_id": model_selectors["experiment_id"],
                "variant_label": model_selectors["variant_label"],
                "reference_source_id": reference_selectors["source_id"],
            }
        )

        return ExecutionResult.build_from_output_bundle(
            definition,
            cmec_output_bundle=format_cmec_output_bundle(),
            cmec_metric_bundle=cmec_metric_bundle,
        )
