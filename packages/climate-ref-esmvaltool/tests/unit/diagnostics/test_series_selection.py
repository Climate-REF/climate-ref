"""The model and reference series are picked out of ESMValTool's multi-dataset files by label.

ESMValTool's ``monitor/multi_datasets.py`` sorts the datasets case-insensitively by label before
``io.save_1d_data`` stacks them along ``dim0``, so the observation comes first for any model whose
name sorts after it (e.g. ``"CERES-EBAF-4-2-1" < "CESM2"``). The existing test cases use CanESM5,
which sorts before both CERES-EBAF-4-2-1 and OSI-450, so they cannot catch a positional selection.
"""

import numpy as np
import pandas
import pytest
import xarray as xr
from climate_ref_esmvaltool.diagnostics.base import index_dataset_labels
from climate_ref_esmvaltool.diagnostics.cloud_radiative_effects import CloudRadiativeEffects
from climate_ref_esmvaltool.diagnostics.sea_ice_area_basic import SeaIceAreaBasic

from climate_ref_core.datasets import DatasetCollection, ExecutionDatasetCollection, SourceDatasetType

MODEL_VALUE = 1.0
REFERENCE_VALUE = 2.0

# Models on both sides of the references, and lowercase-sensitive cases ("Can" vs "CER"):
# a case-sensitive sort would put "CanESM5" after "CERES", ESMValTool's does not.
MODELS = ["ACCESS-CM2", "CanESM5", "CESM2", "UKESM1-3-LL"]


def write_multi_dataset_file(path, var_name, index_name, labels_and_values):
    """Write a file laid out like ESMValTool's ``io.save_1d_data`` output."""
    ordered = sorted(labels_and_values.items(), key=lambda item: item[0].lower())
    width = max(len(label) for label, _ in ordered)
    labels = np.array([label.ljust(width).encode() for label, _ in ordered], dtype=f"|S{width}")
    index = np.arange(3, dtype=float)
    values = np.array([np.full(index.shape, value) for _, value in ordered])
    ds = xr.Dataset(
        {var_name: (("dim0", "dim1"), values)},
        coords={"dataset": ("dim0", labels), index_name: ("dim1", index)},
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(path)


def extract(diagnostic, definition_factory, model, relative_file):
    """Return the file path to write and a callable running the series extraction on it."""
    selector = (("experiment_id", "historical"), ("source_id", model))
    collection = ExecutionDatasetCollection(
        {SourceDatasetType.CMIP6: DatasetCollection(pandas.DataFrame(), "instance_id", selector=selector)}
    )
    definition = definition_factory(diagnostic=diagnostic, execution_dataset_collection=collection)
    filename = definition.to_output_path("executions") / "recipe" / relative_file
    return filename, lambda: diagnostic._extract_series_from_file(
        definition,
        filename,
        definition.as_relative_path(filename),
        caption="",
        input_selectors=dict(selector),
    )


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("var_name", ["lwcre", "swcre"])
def test_cloud_radiative_effects_reference_is_ceres(definition_factory, model, var_name):
    filename, run = extract(
        CloudRadiativeEffects(),
        definition_factory,
        model,
        f"work/plot_profiles/plot/variable_vs_lat_{var_name}_ambiguous_dataset_Amon.nc",
    )
    write_multi_dataset_file(
        filename, var_name, "lat", {model: MODEL_VALUE, "CERES-EBAF-4-2-1": REFERENCE_VALUE}
    )

    series = run()

    by_kind = {s.kind: s for s in series}
    assert len(series) == 2
    assert set(by_kind) == {"model", "reference"}
    assert by_kind["reference"].values == [REFERENCE_VALUE] * 3
    assert by_kind["reference"].dimensions["reference_source_id"] == "CERES-EBAF-4-2-1"
    assert by_kind["model"].values == [MODEL_VALUE] * 3


@pytest.mark.parametrize("model", [*MODELS, "NorESM2-LM", "UKESM1-0-LL"])
@pytest.mark.parametrize(
    "relative_file,var_name,index_name",
    [
        (
            "work/siarea_min/allplots/timeseries_sea_ice_area_{region}_sep_ambiguous_dataset.nc",
            "siconc",
            "time",
        ),
        (
            "work/siarea_seas/allplots/annual_cycle_sea_ice_area_{region}_ambiguous_dataset.nc",
            "siconc",
            "month_number",
        ),
    ],
    ids=["september-minimum", "seasonal-cycle"],
)
@pytest.mark.parametrize("region", ["nh", "sh"])
def test_sea_ice_area_basic_reference_is_osi_450(
    definition_factory, model, relative_file, var_name, index_name, region
):
    filename, run = extract(
        SeaIceAreaBasic(),
        definition_factory,
        model,
        relative_file.format(region=region),
    )
    reference = f"OSI-450-{region}"
    write_multi_dataset_file(filename, var_name, index_name, {model: MODEL_VALUE, reference: REFERENCE_VALUE})

    series = run()

    by_kind = {s.kind: s for s in series}
    assert len(series) == 2
    assert set(by_kind) == {"model", "reference"}
    assert by_kind["reference"].values == [REFERENCE_VALUE] * 3
    assert by_kind["reference"].dimensions["reference_source_id"] == reference
    assert by_kind["model"].values == [MODEL_VALUE] * 3


def test_missing_label_fails_loudly(definition_factory):
    """A label that is not in the file must not fall back to some other dataset."""
    filename, run = extract(
        CloudRadiativeEffects(),
        definition_factory,
        "CESM2",
        "work/plot_profiles/plot/variable_vs_lat_lwcre_ambiguous_dataset_Amon.nc",
    )
    write_multi_dataset_file(
        filename, "lwcre", "lat", {"SomeOtherModel": MODEL_VALUE, "CERES-EBAF-4-2-1": REFERENCE_VALUE}
    )

    with pytest.raises(KeyError, match="CESM2"):
        run()


def test_index_dataset_labels_decodes_padded_byte_strings():
    ds = xr.Dataset(
        {"x": ("dim0", [1.0, 2.0])},
        coords={"dataset": ("dim0", np.array([b"CERES-EBAF-4-2-1", b"CESM2           "]))},
    )

    indexed = index_dataset_labels(ds)

    assert indexed["dataset"].values.tolist() == ["CERES-EBAF-4-2-1", "CESM2"]
    assert float(indexed.sel(dataset="CESM2")["x"]) == 2.0
    # Positional selection on the bare dimension keeps working for single-dataset files.
    assert float(indexed.sel(dim0=0)["x"]) == 1.0


def test_index_dataset_labels_without_labels_is_unchanged():
    ds = xr.Dataset({"x": ("dim0", [1.0, 2.0])})

    assert index_dataset_labels(ds) is ds
