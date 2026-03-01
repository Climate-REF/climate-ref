import ilamb3
import pandas as pd
from climate_ref_ilamb.standard import ILAMBStandard, _set_ilamb3_options
from climate_ref_pmp import provider as ilamb_provider

from climate_ref.solver import solve_executions
from climate_ref_core.dataset_registry import dataset_registry_manager
from climate_ref_core.datasets import DatasetCollection, SourceDatasetType


def test_standard_site(cmip6_data_catalog, definition_factory):
    diagnostic = ILAMBStandard(
        registry_file="ilamb-test", metric_name="test-site-tas", sources={"tas": "ilamb/test/Site/tas.nc"}
    )
    _, ds = next(
        iter(
            cmip6_data_catalog[
                (cmip6_data_catalog["experiment_id"] == "historical")
                & (cmip6_data_catalog["variable_id"] == "tas")
            ].groupby("instance_id")
        )
    )
    definition = definition_factory(
        diagnostic=diagnostic,
        cmip6=DatasetCollection(
            ds,
            "instance_id",
            selector=(
                ("experiment_id", "historical"),
                ("variable_id", "tas"),
                ("source_id", "CanESM5"),
                ("member_id", "r1i1p1f1"),
                ("grid_label", "gn"),
            ),
        ),
    )
    definition.output_directory.mkdir(parents=True, exist_ok=True)

    result = diagnostic.run(definition)

    assert str(result.output_bundle_filename) == "output.json"

    output_bundle_path = definition.output_directory / result.output_bundle_filename

    assert result.successful
    assert output_bundle_path.exists()
    assert output_bundle_path.is_file()

    assert str(result.metric_bundle_filename) == "diagnostic.json"

    metric_bundle_path = definition.output_directory / result.metric_bundle_filename

    assert result.successful
    assert metric_bundle_path.exists()
    assert metric_bundle_path.is_file()


def test_standard_grid(cmip6_data_catalog, definition_factory):
    diagnostic = ILAMBStandard(
        registry_file="ilamb-test",
        metric_name="test-grid-gpp",
        sources={"gpp": "ilamb/test/Grid/gpp.nc"},
        relationships={"pr": "ilamb/test/Grid/pr.nc"},
    )
    _, ds = next(
        iter(
            cmip6_data_catalog[
                (cmip6_data_catalog["experiment_id"] == "historical")
                & ((cmip6_data_catalog["variable_id"] == "gpp") | (cmip6_data_catalog["variable_id"] == "pr"))
            ].groupby(["source_id", "member_id", "grid_label"])
        )
    )

    definition = definition_factory(
        diagnostic=diagnostic,
        cmip6=DatasetCollection(
            ds,
            "instance_id",
            selector=(
                ("experiment_id", "historical"),
                ("variable_id", "tas"),
                ("source_id", "CanESM5"),
                ("member_id", "r1i1p1f1"),
                ("grid_label", "gn"),
            ),
        ),
    )
    definition.output_directory.mkdir(parents=True, exist_ok=True)

    result = diagnostic.run(definition)

    assert str(result.output_bundle_filename) == "output.json"

    output_bundle_path = definition.output_directory / result.output_bundle_filename

    assert result.successful
    assert output_bundle_path.exists()
    assert output_bundle_path.is_file()

    assert str(result.metric_bundle_filename) == "diagnostic.json"

    metric_bundle_path = definition.output_directory / result.metric_bundle_filename

    assert result.successful
    assert metric_bundle_path.exists()
    assert metric_bundle_path.is_file()


def test_options():
    _set_ilamb3_options(dataset_registry_manager["ilamb"], "ilamb")
    assert set(["global", "tropical"]).issubset(ilamb3.conf["regions"])


def test_expected_executions():
    diagnostic = ILAMBStandard(
        registry_file="ilamb",
        metric_name="cSoil-HWSD2",
        sources={"cSoil": "ilamb/cSoil/HWSD2/cSoil_fx_HWSD2_19600101-20220101.nc"},
    )

    # No Obs4MIPs datasets are used yet
    data_catalog = {
        SourceDatasetType.CMIP6: pd.DataFrame(
            [
                ["cSoil", "ACCESS-ESM1-5", "historical", "r1i1p1f1", "mon", "gn", "Amon", "v20191115"],
                ["cSoil", "ACCESS-ESM1-5", "ssp119", "r1i1p1f1", "mon", "gn", "Amon", "v20191115"],
                ["cSoil", "ACCESS-ESM1-5", "historical", "r2i1p1f1", "mon", "gn", "Amon", "v20191115"],
                ["ts", "ACCESS-ESM1-5", "historical", "r1i1p1f1", "mon", "gn", "Amon", "v20191115"],
                ["areacella", "ACCESS-ESM1-5", "fx", "r1i1p1f1", "mon", "gn", "Amon", "v20191115"],
            ],
            columns=(
                "variable_id",
                "source_id",
                "experiment_id",
                "member_id",
                "frequency",
                "grid_label",
                "table_id",
                "version",
            ),
        ),
    }
    executions = list(solve_executions(data_catalog, diagnostic, provider=ilamb_provider))
    assert len(executions) == 2
    assert executions[0].datasets[SourceDatasetType.CMIP6].selector == (
        ("experiment_id", "historical"),
        ("grid_label", "gn"),
        ("member_id", "r1i1p1f1"),
        ("source_id", "ACCESS-ESM1-5"),
    )
    assert executions[0].datasets[SourceDatasetType.CMIP6].datasets["variable_id"].tolist() == [
        "cSoil",
        "areacella",
    ]
    assert executions[0].datasets[SourceDatasetType.CMIP6].datasets["member_id"].tolist() == [
        "r1i1p1f1",
        "r1i1p1f1",
    ]
