import hashlib
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import dask.config
import ilamb3
import ilamb3.regions as ilr
import ilamb3.transform
import numpy as np
import pandas as pd
import pint
import pooch
import xarray as xr
from ilamb3 import run
from ilamb3.dataset import coarsen_dataset, get_dim_name
from ilamb3.transform.base import ILAMBTransform
from loguru import logger

from climate_ref_core.cmip6_to_cmip7 import get_dreq_entry
from climate_ref_core.constraints import AddSupplementaryDataset, RequireFacets
from climate_ref_core.dataset_registry import dataset_registry_manager, resolve_cache_dir
from climate_ref_core.datasets import ExecutionDatasetCollection, FacetFilter, SourceDatasetType
from climate_ref_core.diagnostics import (
    DataRequirement,
    Diagnostic,
    ExecutionDefinition,
    ExecutionResult,
)
from climate_ref_core.esgf import CMIP6Request, CMIP7Request, RegistryRequest
from climate_ref_core.esgf.obs4mips import Obs4MIPsRequest
from climate_ref_core.metric_values.typing import MetricValueKind, SeriesMetricValue
from climate_ref_core.pycmec.metric import CMECMetric
from climate_ref_core.pycmec.output import CMECOutput, OutputCV
from climate_ref_core.testing import TestCase, TestDataSpecification
from climate_ref_ilamb.datasets import (
    registry_to_collection,
)

# CMIP6 tables to search, ordered by likelihood for land vs ocean registries
_LAND_TABLES = ("Lmon", "Emon", "LImon", "Amon", "CFmon", "AERmonZ", "EmonZ")
_OCEAN_TABLES = ("Omon", "SImon", "Ofx")


class _RelationshipTimeTransform(ILAMBTransform):
    """
    Keep relationship-variable datasets aligned with the primary variable.

    ILAMB loads relationship variables into the same dataset as the primary variable before running.
    If the relationship variable spans a different time range,
    xarray's outer merge can make the primary analyses use the relationship variable's time bounds.
    """

    def __init__(self, variable_id: str):
        self.variable_id = variable_id

    def required_variables(self) -> list[str]:
        return [self.variable_id]

    def __call__(self, ds: xr.Dataset) -> xr.Dataset:
        if "time" not in ds.coords:
            return ds

        if ds["time"].dtype == object:
            try:
                ds = ds.convert_calendar("proleptic_gregorian", use_cftime=False)
            except ValueError:
                pass

        if self.variable_id not in ds or "time" not in ds[self.variable_id].dims:
            return ds

        valid = ds[self.variable_id].notnull()
        other_dims = [dim for dim in valid.dims if dim != "time"]
        if other_dims:
            valid = valid.any(dim=other_dims)
        valid = valid.compute()

        if not valid.any():
            return ds

        return ds.sel(time=ds["time"].where(valid, drop=True))


ilamb3.transform.ALL_TRANSFORMS.setdefault("climate_ref_relationship_time", _RelationshipTimeTransform)


class _CoarsenSpatial(ILAMBTransform):
    """
    Conservatively coarsen a very fine field before comparison.

    Some obs4MIPs references (e.g. ``NOAA-NCEI-LAI-AVHRR-5-0`` at ~0.05 degrees) are far finer
    than any CMIP model, which makes ilamb3's regrid/scoring step intractable.
    This coarsens the field to a common target resolution, which ESM outputs are compared to.
    Fields already at or coarser than the target (the models) are left untouched.

    The coarsened reference is the same for every model comparison in a solve, so the result
    is cached to disk keyed on the source data and resolution. The first execution computes it,
    later ones read the cached file. Set ``REF_ILAMB_COARSEN_NO_CACHE`` to disable the cache.
    """

    # Bump when the coarsening algorithm changes so stale cache entries are ignored.
    _CACHE_VERSION = 1
    _CACHE_NAME = "ilamb-coarsened"

    def __init__(self, variable_id: str, resolution: float = 0.5):
        self.variable_id = variable_id
        self.resolution = resolution

    def required_variables(self) -> list[str]:
        return [self.variable_id]

    def __call__(self, ds: xr.Dataset) -> xr.Dataset:
        if self.variable_id not in ds:
            return ds
        lat = ds[get_dim_name(ds, "lat")]
        lon = ds[get_dim_name(ds, "lon")]
        current = float(
            np.sqrt(lat.diff(lat.name).mean().values ** 2 + lon.diff(lon.name).mean().values ** 2)
        )
        if current >= self.resolution:
            # Already at or coarser than the target (e.g. model data); leave it lazy.
            return ds

        cache_path = self._cache_path(ds)
        if cache_path is not None and cache_path.exists():
            logger.debug(f"Reusing cached coarsened reference {cache_path}")
            return xr.open_dataset(cache_path).load()

        coarse = self._coarsen(ds)
        if cache_path is not None:
            self._write_cache(coarse, cache_path)
        return coarse

    # Time steps coarsened per pass. One pass of the ~0.05 degree AVHRR LAI reference is
    # about 12 * 3600 * 7200 * 8 bytes ~= 2.5 GB, well within a CI runner.
    _TIME_CHUNK = 12

    def _coarsen(self, ds: xr.Dataset) -> xr.Dataset:
        """
        Coarsen the fine reference without materialising the whole cube.

        The reference is opened dask-backed and chunked over time (``open_mfdataset`` in ilamb3).
        Coarsening the whole field at once needs the entire cube in memory
        (the AVHRR LAI reference is ~0.05 degrees, roughly 100 GB as float64, and the
        conservative weighting makes several copies of it).
        ``coarsen_dataset`` also cannot run on a lazy field, because its nan-masking indexes with
        a boolean array that must be concrete. So the field is coarsened one time block at a time,
        materialising only that block, and the small coarse blocks are concatenated.
        """
        if "time" not in ds.dims:
            # A single 2D slice is small enough to coarsen in one pass.
            return coarsen_dataset(ds.compute(), self.variable_id, res=self.resolution)

        ntime = ds.sizes["time"]
        blocks = [
            coarsen_dataset(
                ds.isel(time=slice(start, start + self._TIME_CHUNK)).compute(),
                self.variable_id,
                res=self.resolution,
            )
            for start in range(0, ntime, self._TIME_CHUNK)
        ]
        if len(blocks) == 1:
            return blocks[0]
        # ``data_vars="minimal"`` concatenates only the time-varying field, so the time-invariant
        # ``cell_measures`` is kept once rather than broadcast across every block.
        return xr.concat(blocks, dim="time", data_vars="minimal", coords="minimal", compat="override")

    def _cache_path(self, ds: xr.Dataset) -> Path | None:
        if os.environ.get("REF_ILAMB_COARSEN_NO_CACHE"):
            return None
        fingerprint = self._source_fingerprint(ds)
        if fingerprint is None:
            return None
        digest = hashlib.sha256(fingerprint.encode()).hexdigest()[:32]
        return resolve_cache_dir(self._CACHE_NAME) / f"{self.variable_id}_{digest}.nc"

    def _source_fingerprint(self, ds: xr.Dataset) -> str | None:
        """
        Build a stable identity for the source data behind a cached coarsening.

        Prefers the on-disk source files (path, size, mtime) recorded in the dataset encoding.
        Falls back to the grid and identifying attributes when no source path is available.
        Returns ``None`` only when nothing stable can be derived, which skips the cache.
        """
        parts = [f"v{self._CACHE_VERSION}", self.variable_id, f"{self.resolution:g}"]

        sources = set()
        for candidate in (ds, ds.get(self.variable_id)):
            source = getattr(candidate, "encoding", {}).get("source")
            if isinstance(source, str):
                sources.add(source)
        if sources:
            for source in sorted(sources):
                try:
                    stat = os.stat(source)
                    parts.append(f"{source}:{stat.st_size}:{stat.st_mtime_ns}")
                except OSError:
                    parts.append(source)
            return "|".join(parts)

        # No source path (merged or in-memory dataset): fall back to grid + identity attrs.
        try:
            for dim in ("time", "lat", "lon"):
                name = get_dim_name(ds, dim) if dim in ("lat", "lon") else dim
                if name in ds.coords:
                    values = np.asarray(ds[name].values)
                    parts.append(f"{name}:{values.shape}:{values.dtype}")
                    parts.append(hashlib.sha256(np.ascontiguousarray(values).tobytes()).hexdigest())
        except Exception:  # pragma: no cover - defensive; skip caching if the grid is unreadable
            return None
        for attr in ("tracking_id", "source_id", "activity_id", "version", "variable_id"):
            if attr in ds.attrs:
                parts.append(f"{attr}={ds.attrs[attr]}")
        return "|".join(parts)

    def _write_cache(self, coarse: xr.Dataset, cache_path: Path) -> None:
        """Write the coarsened reference atomically so concurrent executions never read a partial file."""
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = cache_path.with_name(f"{cache_path.stem}.{os.getpid()}.tmp{cache_path.suffix}")
            coarse.to_netcdf(tmp_path)
            os.replace(tmp_path, cache_path)
            logger.debug(f"Cached coarsened reference {cache_path}")
        except Exception as exc:  # pragma: no cover - a cache write failure must not fail the run
            logger.warning(f"Could not cache coarsened reference {cache_path}: {exc}")


ilamb3.transform.ALL_TRANSFORMS.setdefault("climate_ref_coarsen_spatial", _CoarsenSpatial)


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
                # Last dimension carries the value column's metadata.
                metadata = dataset[dataset[dim] == val].iloc[0][metadata_columns].to_dict()
                dim_dict[str(val)] = {column: str(value) for column, value in metadata.items()}

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


def _clean_units(units: str) -> str:
    """
    Normalise a units string to a clean CF/UDUNITS form.

    Model traces hold a pint repr (``"kg / meter ** 2 / second"``) while references keep
    CF units (``"kg m-2 s-1"``); parsing through the ilamb3 pint registry and re-emitting
    symbols gives both the same form. Unparseable and dimensionless strings pass through.
    """
    if not units:
        return units
    ureg = pint.get_application_registry()  # type: ignore[no-untyped-call]
    try:
        unit = ureg.Unit(units)
    except Exception:
        # Not something pint understands (e.g. "months since 1980"); leave it as-is.
        return units
    parts = []
    # Positive (and zero) exponents first, then negatives; alphabetical within each group.
    for name, exponent in sorted(unit._units.items(), key=lambda item: (item[1] < 0, item[0])):
        symbol = ureg.get_symbol(name)
        power = int(exponent) if exponent == int(exponent) else exponent
        parts.append(symbol if power == 1 else f"{symbol}{power}")
    return " ".join(parts) or units


def _build_series(
    output_directory: Path,
    dataset_source: str,
    common_dimensions: Mapping[str, str],
) -> list[SeriesMetricValue]:
    """
    Build series metric values from the 1-d time traces in the output directory.

    A ``Reference.nc`` trace is a reference series carrying only the reference identity;
    every other trace is a model series that also keeps the reference it was scored
    against so the two group together. Units are normalised to a clean CF form.
    """
    series: list[SeriesMetricValue] = []
    for ncfile in sorted(output_directory.glob("*.nc")):
        time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
        ds = xr.open_dataset(ncfile, decode_times=time_coder)
        for name, da in sorted(ds.items()):
            # Only create series for 1d DataArray's with these dimensions
            if not (da.ndim == 1 and set(da.dims).intersection(["time", "month"])):
                continue
            str_name = str(name)
            index_name = str(da.dims[0])
            raw_index = ds[index_name].values.tolist()
            # Capture the calendar from the cftime index before converting it to
            # ISO strings (afterwards the values are plain strings with no calendar).
            calendar = raw_index[0].calendar if hasattr(raw_index[0], "calendar") else None
            if hasattr(raw_index[0], "isoformat"):
                index = [v.isoformat() for v in raw_index]
            else:
                index = raw_index

            # Presentation metadata; units normalised so model and reference agree.
            value_units = _clean_units(da.attrs.get("units", ""))
            value_long_name = da.attrs.get("long_name", str_name)
            index_units = ds[index_name].attrs.get("units") or None
            attrs: dict[str, str | float | int | None] = {
                "units": value_units,
                "long_name": value_long_name,
                "standard_name": da.attrs.get("standard_name", ""),
            }
            if calendar is not None:
                attrs["calendar"] = calendar

            # Parse out some dimensions
            kind: MetricValueKind
            if ncfile.stem == "Reference":
                # Reference series carry the reference identity only; drop the
                # "Reference" source_id sentinel (model identity is for comparisons).
                kind = "reference"
                dimensions = {
                    "reference_source_id": dataset_source,
                    "metric": str_name,
                }
            else:
                kind = "model"
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
                    kind=kind,
                    dimensions=dimensions,
                    values=da.values.tolist(),
                    index=index,
                    index_name=index_name,
                    value_units=value_units,
                    value_long_name=value_long_name,
                    index_units=index_units,
                    calendar=calendar,
                    attributes=attrs,
                )
            )
    return series


def _build_cmec_bundle(df: pd.DataFrame) -> dict[str, Any]:
    """
    Build a CMEC bundle from information in the dataframe.

    """
    # TODO: Handle the reference data
    # reference_df = df[df["source"] == "Reference"]
    model_df = df[df["source"] != "Reference"].copy()

    # Strip out units from the name (available in the attributes)
    extracted_source = model_df.name.str.extract(r"(.*)\s\[.*\]")[0]
    model_df = model_df.assign(name=extracted_source.where(extracted_source.notna(), model_df["name"]))

    model_df = model_df.rename(
        columns={
            "analysis": "metric",
            "name": "statistic",
        }
    )

    # Convert the value column to numeric, coercing errors to NaN
    model_df = model_df.assign(value=pd.to_numeric(model_df["value"], errors="coerce").astype("float64"))

    # Scalars are model-vs-reference comparisons: kind "model", keeping both identities.
    model_df["kind"] = "model"

    # kind and reference_source_id are constant, so they go outermost and wrap uniformly;
    # statistic stays terminal since metrics expose different (ragged) statistic sets.
    dimensions = [
        "kind",
        "reference_source_id",
        "experiment_id",
        "source_id",
        "member_id",
        "grid_label",
        "region",
        "metric",
        "statistic",
    ]
    attributes = ["type", "units"]
    for dimension in dimensions:
        model_df[dimension] = model_df[dimension].where(model_df[dimension].notna(), "None")

    bundle = format_cmec_output_bundle(
        model_df,
        dimensions=dimensions,
        metadata_columns=attributes,
        value_column="value",
    )

    ilamb_regions = ilr.Regions()
    for region, region_info in bundle["DIMENSIONS"]["region"].items():
        if region in {"None", "nan", "<NA>"}:
            region_info["LongName"] = "None"
            region_info["Description"] = "Reference data extents"
            region_info["Generator"] = "N/A"
        else:
            region_info["LongName"] = ilamb_regions.get_name(region)
            region_info["Description"] = ilamb_regions.get_name(region)
            region_info["Generator"] = ilamb_regions.get_source(region)

    return bundle


def _build_cmip_data_requirement(  # noqa: PLR0913
    source_type: SourceDatasetType,
    filters: dict[str, Any],
    group_by: tuple[str, ...],
    primary_variable_ids: tuple[str, ...],
    relationship_variable_ids: tuple[str, ...],
    is_land: bool,
) -> DataRequirement:
    """
    Build a CMIP data requirement with shared constraint logic.

    The constraints (RequireFacets for primary/relationship variables and
    AddSupplementaryDataset for land/ocean ancillary data) are identical
    for CMIP6 and CMIP7 except for the ``source_type`` parameter.

    Parameters
    ----------
    source_type
        CMIP6 or CMIP7
    filters
        Facet filters specific to this source type
    group_by
        Columns to group executions by
    primary_variable_ids
        Variables the diagnostic requires (primary + alternates + related)
    relationship_variable_ids
        Variables used in relationship analyses
    is_land
        Whether this is a land diagnostic (determines supplementary variables)

    Returns
    -------
    :
        A DataRequirement for the given CMIP source type
    """
    constraints: list[RequireFacets | AddSupplementaryDataset] = [
        RequireFacets(
            "variable_id",
            primary_variable_ids,
            operator="any",
        ),
    ]

    if relationship_variable_ids:
        constraints.append(
            RequireFacets(
                "variable_id",
                required_facets=relationship_variable_ids,
            )
        )

    if is_land:
        constraints.extend(
            [
                AddSupplementaryDataset.from_defaults("areacella", source_type),
                AddSupplementaryDataset.from_defaults("sftlf", source_type),
            ]
        )
    else:
        constraints.extend(
            [
                AddSupplementaryDataset.from_defaults("volcello", source_type),
                AddSupplementaryDataset.from_defaults("areacello", source_type),
                AddSupplementaryDataset.from_defaults("sftof", source_type),
            ]
        )

    return DataRequirement(
        source_type=source_type,
        filters=(FacetFilter(facets=filters),),
        constraints=tuple(constraints),
        group_by=group_by,
    )


def _build_test_data_spec(  # noqa: PLR0913
    all_variable_ids: tuple[str, ...],
    registry_file: str,
    test_source_id: str,
    is_land: bool,
    obs_filters: Mapping[str, tuple[str, ...]],
    obs_source: str | None = "obs4mips",
) -> TestDataSpecification:
    """
    Build a TestDataSpecification for an ILAMB diagnostic.

    Parameters
    ----------
    all_variable_ids
        All variable IDs used by the diagnostic (primary + alternate + related + relationships)
    registry_file
        ILAMB registry name ("ilamb", "ilamb-test", or "iomb")
    test_source_id
        CMIP source_id to use in test cases (e.g. "CanESM5")
    is_land
        Whether this is a land diagnostic (determines supplementary variables)
    obs_filters
        Filters extracted from dict-based sources.
        If non-empty the test-case will search for non-ilamb datasets.
    obs_source
        Which source to use for obs4MIPs data ("obs4ref" or "obs4mips").
        Determines if we fetch from ESGF or use the pre-fetched obs4REF registry.
        Ignored if no obs_filters provided.

    Returns
    -------
    :
        Test data specification with cmip6 and cmip7 test cases
    """
    if is_land:
        supplementary_vars = ["areacella", "sftlf"]
    else:
        supplementary_vars = ["volcello", "areacello", "sftof"]
    test_variable_ids = tuple(sorted(set(all_variable_ids) | set(supplementary_vars)))

    test_branded_names = sorted(
        set(
            _get_branded_variable(
                tuple(test_variable_ids),
                registry_file,
            )
        )
    )

    obs4mips_requests: tuple[RegistryRequest | Obs4MIPsRequest, ...] = ()
    if obs_filters and obs_source is not None:
        slug = obs_filters.get("source_id", ["obs4mips"])[0]
        if obs_source == "obs4ref":
            obs4mips_requests = (
                RegistryRequest(
                    slug=slug,
                    registry_name="obs4ref",
                    facets=obs_filters,  # type: ignore
                    source_type="obs4MIPs",
                ),
            )
        elif obs_source == "obs4mips":
            obs4mips_requests = (
                Obs4MIPsRequest(
                    slug=slug,
                    facets=obs_filters,  # type: ignore
                ),
            )
        else:
            raise ValueError(f"Invalid obs_source: {obs_source}")

    return TestDataSpecification(
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
                            "frequency": ("fx", "mon"),
                        },
                        remove_ensembles=True,
                    ),
                    *obs4mips_requests,
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
                            "frequency": ["fx", "mon"],
                            "region": "glb",
                        },
                        remove_ensembles=True,
                    ),
                    *obs4mips_requests,
                ),
            ),
        )
    )


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
    # You can control how models are known in the results, drop some facets for
    # legibility.
    ilamb3.conf.set(model_name_facets=["source_id"])


def _load_csv_and_merge(output_directory: Path) -> pd.DataFrame:
    """
    Load individual csv scalar data and merge into a dataframe.
    """
    df = pd.concat(
        [pd.read_csv(f, keep_default_na=False, na_values=["NaN"]) for f in output_directory.glob("*.csv")]
    ).drop_duplicates(subset=["source", "region", "analysis", "name"])
    return df


def _register_data_outputs(output_bundle: dict[str, Any], definition: ExecutionDefinition) -> None:
    """
    Register ILAMB's scalar CSV and netCDF outputs in the data section of a CMEC output bundle.

    These files are re-read by ``build_execution_result`` (via ``_load_csv_and_merge`` and the
    netCDF series loop) to reconstruct the metrics and series, so they must be persisted with the
    execution outputs and captured in the regression baseline rather than left in the scratch
    directory.
    """
    for datafile in sorted(
        [*definition.output_directory.glob("*.csv"), *definition.output_directory.glob("*.nc")]
    ):
        relative_path = str(definition.as_relative_path(datafile))
        output_bundle[OutputCV.DATA.value][relative_path] = {
            OutputCV.FILENAME.value: relative_path,
            OutputCV.LONG_NAME.value: datafile.name,
            OutputCV.DESCRIPTION.value: "Scalar and time-series data produced by ILAMB.",
        }


class ILAMBStandard(Diagnostic):
    """
    Apply the standard ILAMB analysis with respect to a given reference dataset.
    """

    version = 2
    """
    Default version for ILAMB diagnostics.

    Individual diagnostics can override this with a ``version`` key in their configuration entry.
    """

    def __init__(
        self,
        registry_file: str,
        metric_name: str,
        sources: dict[str, str | dict[str, str]],
        **ilamb_kwargs: Any,
    ):
        # Setup the diagnostic
        self.variable_id = ilamb_kwargs.get("analysis_variable", next(iter(sources.keys())))
        if "sources" not in ilamb_kwargs:  # pragma: no cover
            ilamb_kwargs["sources"] = sources
        if "relationships" not in ilamb_kwargs:
            ilamb_kwargs["relationships"] = {}
        if ilamb_kwargs["relationships"]:
            transforms = list(ilamb_kwargs.get("transforms", []))
            transform_names = [
                next(iter(transform.keys())) if isinstance(transform, dict) else transform
                for transform in transforms
            ]
            if "climate_ref_relationship_time" not in transform_names:
                transforms.append({"climate_ref_relationship_time": {"variable_id": self.variable_id}})
            ilamb_kwargs["transforms"] = transforms

        # Allow per-diagnostic override of the test source_id
        # (not all variables are available from CanESM5)
        test_source_id = ilamb_kwargs.pop("test_source_id", "CanESM5")

        diagnostic_version = ilamb_kwargs.pop("version", None)
        if diagnostic_version is not None:
            self.version = diagnostic_version

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

        relationship_variable_ids = tuple(ilamb_kwargs.get("relationships", {}).keys())

        # Create the data requirement for the dataset under test
        cmip6_requirement = _build_cmip_data_requirement(
            source_type=SourceDatasetType.CMIP6,
            filters={
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
            },
            group_by=("experiment_id", "source_id", "member_id", "grid_label"),
            primary_variable_ids=primary_variable_ids,
            relationship_variable_ids=relationship_variable_ids,
            is_land=is_land,
        )

        cmip7_requirement = _build_cmip_data_requirement(
            source_type=SourceDatasetType.CMIP7,
            filters={
                "branded_variable": branded_variables,
                "frequency": "mon",
                "experiment_id": ("historical", "land-hist"),
                "region": "glb",
            },
            group_by=("experiment_id", "source_id", "variant_label", "grid_label"),
            primary_variable_ids=primary_variable_ids,
            relationship_variable_ids=relationship_variable_ids,
            is_land=is_land,
        )

        # obs4MIPs data requirement, normally ilamb3 expects the `sources` to
        # resolve to keys in one of its data registries. If instead we find a
        # dictionary, then assume that these keys are meant to be keywords in a
        # REF data requirement.
        # obs_source key is used to determine whether to fetch from ESGF
        # or use the pre-fetched obs4REF registry.
        filters: dict[str, tuple[str, ...]] = {}
        obs_source = None
        for _, source in sources.items():
            if isinstance(source, dict):
                obs_source = source.pop("obs_source", None)
                for key, val in source.items():
                    if key not in filters:
                        filters[key] = ()
                    filters[key] += (val,)
        obs4mips_requirement = (
            DataRequirement(
                source_type=SourceDatasetType.obs4MIPs,
                filters=(FacetFilter(facets=filters),),
                group_by=tuple(filters.keys()),
            )
            if filters
            else None
        )

        data_requirements: Sequence[Sequence[DataRequirement]]
        if obs4mips_requirement is None:
            data_requirements = (
                (cmip6_requirement,),
                (cmip7_requirement,),
            )
        else:
            data_requirements = (
                (cmip6_requirement, obs4mips_requirement),
                (cmip7_requirement, obs4mips_requirement),
            )
        self.data_requirements = data_requirements

        self.facets = (
            "experiment_id",
            "source_id",
            "member_id",
            "grid_label",
            "region",
            "metric",
            "statistic",
        )

        self.test_data_spec = _build_test_data_spec(
            all_variable_ids=all_variable_ids,
            registry_file=registry_file,
            test_source_id=test_source_id,
            is_land=is_land,
            obs_filters=filters,
            obs_source=obs_source,
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
        # Temporary hack of the ilamb3 inputs while we still need to refer to
        # data not yet available in obs4{MIPs,REF}. This logic allows for
        # DataRequirement filters to be added as a 'source' in the ilamb
        # configure file. If a dictionary instead of a string was found, we
        # populate an obs4MIPs requirement.
        if SourceDatasetType.obs4MIPs in definition.datasets:
            # ilamb3 will expect the reference dataset dataframe to have a `key`
            # column that uniquely describes each dataset. Create one using the
            # `instance_id` and an integer and then modify the ilamb3 configure
            # so that it finds the proper data.
            ref_datasets = definition.datasets[SourceDatasetType.obs4MIPs].datasets
            ref_datasets = ref_datasets.reset_index()
            ref_datasets["key"] = ref_datasets["instance_id"] + ref_datasets.index.astype(str)
            for instance_id, df in ref_datasets.groupby("instance_id"):
                variable_id = df["variable_id"].unique()[0]
                self.ilamb_kwargs["sources"][variable_id] = f"{instance_id}*"
            # Relationship analyses (and any remaining legacy string-path sources)
            # still refer to keys in the ILAMB/obs4REF registries.
            # Keep those keys in the reference dataframe alongside the ingested obs4MIPs datasets so
            # they remain resolvable when a diagnostic mixes obs4REF and registry data.
            ref_datasets = pd.concat(
                [
                    ref_datasets,
                    self.ilamb_data.datasets,
                    registry_to_collection(dataset_registry_manager["obs4ref"]).datasets,
                ],
                ignore_index=True,
            )
        else:
            # If the data is not ingested yet but in a registry, we concat the
            # obs4REF registries to the ilamb one so that any key may be used
            # from either. Eventually (?) this can be removed.
            ref_datasets = pd.concat(
                [
                    self.ilamb_data.datasets.set_index(self.ilamb_data.slug_column),
                    registry_to_collection(dataset_registry_manager["obs4ref"]).datasets.set_index("key"),
                ]
            )
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

        # In ilamb3, this is run with all the models that we know about to
        # create different colors. For REF this will at least make the model
        # line color not be black
        run.set_model_colors(model_datasets)

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

        # Add each png file plot to the output.
        output_bundle = CMECOutput.create_template()
        for plotfile in sorted(definition.output_directory.glob("*.png")):
            relative_path = str(definition.as_relative_path(plotfile))
            caption, figure_dimensions = _caption_from_filename(plotfile, common_dimensions)

            output_bundle[OutputCV.PLOTS.value][relative_path] = {
                OutputCV.FILENAME.value: relative_path,
                OutputCV.LONG_NAME.value: caption,
                OutputCV.DESCRIPTION.value: "",
                OutputCV.DIMENSIONS.value: figure_dimensions,
            }

        # Register the scalar CSV files and the netCDF time-trace files in the data section so
        # they are persisted with the execution outputs and captured in the regression baseline;
        # build_execution_result re-reads them to reconstruct the metrics and series.
        _register_data_outputs(output_bundle, definition)

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
        series = _build_series(definition.output_directory, dataset_source, common_dimensions)

        return ExecutionResult.build_from_output_bundle(
            definition, cmec_output_bundle=output_bundle, cmec_metric_bundle=metric_bundle, series=series
        )


def _caption_from_filename(filename: Path, common_dimensions: dict[str, str]) -> tuple[str, dict[str, str]]:
    source, region, plot = filename.stem.split("_")
    if region.lower() in {"none", "nan", "<na>"}:
        region = "None"
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

    # Reference plots carry the reference identity, not a model source_id.
    if source == "Reference":
        figure_dimensions["kind"] = "reference"
        figure_dimensions["reference_source_id"] = common_dimensions.get("reference_source_id", "None")
    else:
        figure_dimensions = {**common_dimensions, **figure_dimensions}

    return caption, figure_dimensions
