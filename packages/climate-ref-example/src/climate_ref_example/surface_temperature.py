from pathlib import Path
from typing import Any, cast

import numpy as np
import xarray as xr
from loguru import logger

from climate_ref_core.constraints import AddSupplementaryDataset, RequireContiguousTimerange
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

# Model variable: sea surface temperature on the ocean grid (CMIP `tos`, reported in degC).
_MODEL_VARIABLE = "tos"
# Ocean cell-area field used to area-weight the model mean.
_AREA_VARIABLE = "areacello"
# Reference observations: the HadISST-1-1 sea surface temperature record (CF name ``ts``,
# reported in K), distributed via the REF obs4REF registry. It is ocean-only.
_REFERENCE_VARIABLE = "ts"
_REFERENCE_SOURCE_ID = "HadISST-1-1"


def latest_version_files(collection: DatasetCollection) -> list[Path]:
    """
    Return the file paths for the most recent version in a dataset collection.

    A collection grouped only by ``source_id``/``variable_id`` can contain more than one
    dataset version (for example, the obs4REF archive ships two HadISST-1-1 versions).
    Combining versions with overlapping time ranges would make the global mean ambiguous,
    so we deterministically keep only the latest version before opening the files. When the
    collection mixes variables (e.g. ``tos`` plus its ``areacello`` supplementary) the
    latest version is selected per variable so a supplementary is never dropped.

    Parameters
    ----------
    collection
        The datasets selected for one source type of an execution.

    Returns
    -------
    :
        Paths belonging to the latest version (per variable) present in the collection.
    """
    df = collection.datasets
    if "version" not in df.columns:
        return [Path(path) for path in df["path"].to_list()]

    group_column = "variable_id" if "variable_id" in df.columns else None
    groups = df.groupby(group_column) if group_column else [(None, df)]

    paths: list[Path] = []
    for _, group in groups:
        versions = sorted(str(version) for version in group["version"].dropna().unique())
        selected = group
        if len(versions) > 1:
            latest = versions[-1]
            logger.warning(f"Multiple versions present ({', '.join(versions)}); using the latest, {latest}.")
            selected = group[group["version"].astype(str) == latest]
        paths.extend(Path(path) for path in selected["path"].to_list())
    return paths


def _to_celsius(da: xr.DataArray) -> xr.DataArray:
    """Return ``da`` in degrees Celsius, converting from Kelvin when needed."""
    units = str(da.attrs.get("units", "")).strip().lower()
    if units in ("k", "kelvin", "degk", "deg_k"):
        return da - 273.15
    return da


def _global_mean_series(da: xr.DataArray, weights: xr.DataArray) -> xr.DataArray:
    """
    Reduce a gridded field to an annual, area-weighted global-mean series.

    The weighted mean is taken over every non-time dimension, so it works for both a
    regular latitude/longitude grid (weighted by cosine of latitude) and a curvilinear
    ocean grid (weighted by ``areacello``). Land cells are excluded because the surface
    temperature there is missing.

    Parameters
    ----------
    da
        Gridded field with a ``time`` dimension.
    weights
        Area weights broadcastable against the spatial dimensions of ``da``.

    Returns
    -------
    :
        Annual global mean, indexed by integer calendar ``year``.
    """
    spatial_dims = [dim for dim in da.dims if dim != "time"]
    spatial_mean = da.weighted(weights.fillna(0)).mean(dim=spatial_dims, keep_attrs=True)
    # Grouping by integer year keeps the model and reference series alignable even when
    # their calendars differ. Materialise with ``.load()`` so it survives closing the file.
    annual_mean = spatial_mean.groupby("time.year").mean().load()
    return annual_mean


def model_global_mean_sst(input_files: list[Path]) -> xr.DataArray:
    """
    Annual area-weighted global-mean SST for the model, weighted by ``areacello``.

    Parameters
    ----------
    input_files
        The model ``tos`` files together with the ``areacello`` supplementary.
    """
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    with xr.open_mfdataset(
        input_files,
        combine="by_coords",
        chunks=None,
        decode_times=time_coder,
        data_vars="all",
        compat="no_conflicts",
    ) as xr_ds:
        sst = _to_celsius(xr_ds[_MODEL_VARIABLE])
        series = _global_mean_series(sst, xr_ds[_AREA_VARIABLE])
        series.name = _MODEL_VARIABLE
        return series


def reference_global_mean_sst(input_files: list[Path]) -> xr.DataArray:
    """
    Annual global-mean SST for the reference, weighted by cosine of latitude.

    The reference (HadISST-1-1) is on a regular latitude/longitude grid and is already
    ocean-only, so cosine-of-latitude weighting is sufficient.
    """
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    with xr.open_mfdataset(
        input_files,
        combine="by_coords",
        chunks=None,
        decode_times=time_coder,
        data_vars="all",
        compat="no_conflicts",
    ) as xr_ds:
        sst = _to_celsius(xr_ds[_REFERENCE_VARIABLE])
        weights = cast(xr.DataArray, np.cos(np.deg2rad(xr_ds["lat"])))
        series = _global_mean_series(sst, weights)
        series.name = _REFERENCE_VARIABLE
        return series


def compare_model_and_reference(
    model_series: xr.DataArray,
    reference_series: xr.DataArray,
) -> xr.Dataset:
    """
    Build a comparison dataset from a model and a reference series.

    The two series are restricted to their overlapping years before the bias is computed,
    so the result is well defined even when the model run and the observational record
    cover different periods.

    Returns
    -------
    :
        Dataset exposing the aligned ``model``, ``reference`` and ``bias`` series (degC),
        with the ``rmse`` and ``mean_bias`` scalar statistics stored as attributes.
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


def plot_comparison(comparison: xr.Dataset, output_directory: Path) -> None:
    """
    Write the two summary figures for the comparison.

    Produces ``surface_temperature_timeseries.png`` (model and reference series together)
    and ``surface_temperature_bias.png`` (the model-minus-reference bias).
    """
    # Imported lazily so the provider module stays cheap to import for diagnostic discovery,
    # and so the headless Agg backend is selected before pyplot is loaded.
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    years = comparison["year"].values

    # Figure 1 - model and reference series together ("A" and "B").
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(years, comparison["model"].values, label="Model", color="C1")
    ax.plot(years, comparison["reference"].values, label=f"Reference ({_REFERENCE_SOURCE_ID})", color="C0")
    ax.set_xlabel("Year")
    ax.set_ylabel("Global-mean SST (degC)")
    ax.set_title("Global-mean sea surface temperature")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_directory / "surface_temperature_timeseries.png", dpi=120)
    plt.close(fig)

    # Figure 2 - the model-minus-reference bias ("A - B").
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(years, comparison["bias"].values, color="C3")
    ax.axhline(0.0, color="k", linewidth=0.6)
    ax.set_xlabel("Year")
    ax.set_ylabel("Model - reference (degC)")
    ax.set_title(f"Global-mean SST bias (model - {_REFERENCE_SOURCE_ID})")
    fig.tight_layout()
    fig.savefig(output_directory / "surface_temperature_bias.png", dpi=120)
    plt.close(fig)


def format_cmec_output_bundle() -> dict[str, Any]:
    """
    Create a CMEC output bundle registering the comparison series and figures.
    """
    cmec_output = CMECOutput.create_template()

    cmec_output["data"]["surface_temperature_comparison"] = {
        "filename": "global_mean_surface_temperature.nc",
        "long_name": "Global Mean Sea Surface Temperature: model, reference and bias",
        "description": (
            "Annual area-weighted global-mean sea surface temperature for the model "
            "(tos, weighted by areacello) and the HadISST-1-1 reference, with the "
            "model-minus-reference bias."
        ),
    }
    cmec_output["plots"]["timeseries"] = {
        "filename": "surface_temperature_timeseries.png",
        "long_name": "Global-mean SST: model and reference",
        "description": "Annual global-mean SST for the model and the HadISST-1-1 reference.",
    }
    cmec_output["plots"]["bias"] = {
        "filename": "surface_temperature_bias.png",
        "long_name": "Global-mean SST bias (model - reference)",
        "description": "Annual model-minus-reference global-mean SST bias.",
    }

    CMECOutput.model_validate(cmec_output)
    return cmec_output


def format_cmec_metric_bundle(comparison: xr.Dataset) -> dict[str, Any]:
    """
    Create a CMEC diagnostic bundle from the model-vs-reference comparison.
    """
    cmec_metric = {
        "DIMENSIONS": {
            "json_structure": [
                "region",
                "metric",
                "statistic",
            ],
            "region": {"global": {}},
            "metric": {_MODEL_VARIABLE: {}},
            "statistic": {"rmse": {}, "mean-bias": {}},
        },
        "RESULTS": {
            "global": {
                _MODEL_VARIABLE: {
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
    Compare modelled sea surface temperature against observations.

    This diagnostic computes the annual, area-weighted global-mean sea surface
    temperature for a model (``tos``, weighted by ``areacello``) and for the HadISST-1-1
    observational record, then reports the root-mean-square error and mean bias between
    them. Because both sides are ocean SST, this is a like-for-like comparison. It is a
    deliberately small model-versus-observation example that runs in seconds on the sample
    data, and shows how a diagnostic combines a model dataset with a reference dataset and
    emits both metrics and figures.

    Supports both CMIP6 and CMIP7 model datasets via alternative data requirements.
    """

    name = "Global Mean Surface Temperature Bias"
    slug = "global-mean-surface-temperature-bias"

    # Each option is an AND-group: a model requirement combined with the observational
    # reference requirement. The diagnostic runs if either the CMIP6 or the CMIP7 model
    # data is available alongside the reference dataset.
    _reference_requirement = DataRequirement(
        source_type=SourceDatasetType.obs4MIPs,
        filters=(
            FacetFilter(facets={"source_id": (_REFERENCE_SOURCE_ID,), "variable_id": (_REFERENCE_VARIABLE,)}),
        ),
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
                            "variable_id": (_MODEL_VARIABLE,),
                            "experiment_id": ("historical",),
                        }
                    ),
                ),
                group_by=("source_id", "variable_id", "experiment_id", "variant_label"),
                constraints=(
                    RequireContiguousTimerange(group_by=("instance_id",)),
                    AddSupplementaryDataset.from_defaults(_AREA_VARIABLE, SourceDatasetType.CMIP6),
                ),
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
                            "variable_id": (_MODEL_VARIABLE,),
                            "experiment_id": ("historical",),
                        }
                    ),
                ),
                group_by=("source_id", "variable_id", "experiment_id", "variant_label"),
                constraints=(
                    RequireContiguousTimerange(group_by=("instance_id",)),
                    AddSupplementaryDataset.from_defaults(_AREA_VARIABLE, SourceDatasetType.CMIP7),
                ),
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
                description="Historical tos from ACCESS-ESM1-5 compared against HadISST-1-1 observations",
                requests=(
                    RegistryRequest(
                        slug="surface-temperature-obs",
                        registry_name="obs4ref",
                        source_type="obs4MIPs",
                        facets={"source_id": _REFERENCE_SOURCE_ID, "variable_id": _REFERENCE_VARIABLE},
                    ),
                    CMIP6Request(
                        slug="surface-temperature-tos",
                        facets={
                            "source_id": "ACCESS-ESM1-5",
                            "experiment_id": "historical",
                            "variable_id": _MODEL_VARIABLE,
                            "member_id": "r1i1p1f1",
                            "table_id": "Omon",
                        },
                        time_span=("2000-01", "2014-12"),
                    ),
                    CMIP6Request(
                        slug="surface-temperature-areacello",
                        facets={
                            "source_id": "ACCESS-ESM1-5",
                            "experiment_id": "historical",
                            "variable_id": _AREA_VARIABLE,
                            "member_id": "r1i1p1f1",
                            "table_id": "Ofx",
                        },
                    ),
                ),
            ),
            TestCase(
                name="cmip7",
                description="CMIP7 historical tos from ACCESS-ESM1-5 compared against HadISST-1-1",
                requests=(
                    RegistryRequest(
                        slug="surface-temperature-obs-cmip7",
                        registry_name="obs4ref",
                        source_type="obs4MIPs",
                        facets={"source_id": _REFERENCE_SOURCE_ID, "variable_id": _REFERENCE_VARIABLE},
                    ),
                    CMIP7Request(
                        slug="surface-temperature-tos-cmip7",
                        facets={
                            "source_id": "ACCESS-ESM1-5",
                            "experiment_id": "historical",
                            "variable_id": _MODEL_VARIABLE,
                            "variant_label": "r1i1p1f1",
                            "table_id": "Omon",
                        },
                        time_span=("2000-01", "2014-12"),
                    ),
                    CMIP7Request(
                        slug="surface-temperature-areacello-cmip7",
                        facets={
                            "source_id": "ACCESS-ESM1-5",
                            "experiment_id": "historical",
                            "variable_id": _AREA_VARIABLE,
                            "variant_label": "r1i1p1f1",
                            "table_id": "Ofx",
                        },
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

        # latest_version_files guards against multiple dataset versions being grouped into a
        # single execution (selecting the latest version per variable, so areacello is kept).
        model_series = model_global_mean_sst(latest_version_files(model_datasets))
        reference_series = reference_global_mean_sst(latest_version_files(reference_datasets))

        comparison = compare_model_and_reference(model_series, reference_series)
        comparison.to_netcdf(definition.output_directory / "global_mean_surface_temperature.nc")
        plot_comparison(comparison, definition.output_directory)

    def build_execution_result(self, definition: ExecutionDefinition) -> ExecutionResult:
        """
        Create a result object from the output of the diagnostic.
        """
        with xr.open_dataset(
            definition.output_directory / "global_mean_surface_temperature.nc"
        ) as comparison:
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
