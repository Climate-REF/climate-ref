import json
import re
from pathlib import Path

import pandas as pd
import pytest
from attr import evolve

from cmip_ref_core.datasets import FacetFilter, SourceDatasetType
from cmip_ref_core.metrics import (
    CommandLineMetric,
    DataRequirement,
    MetricExecutionDefinition,
    MetricExecutionResult,
    ensure_relative_path,
)
from cmip_ref_core.providers import CommandLineMetricsProvider, MetricsProvider
from cmip_ref_core.pycmec.metric import CMECMetric
from cmip_ref_core.pycmec.output import CMECOutput


@pytest.fixture
def cmec_right_output_dict():
    return {
        "provenance": {
            "environment": {
                "OS": "LINUX",
                "Package": "ILAMB",
                "Machine": "Frontier",
                "Variable": "Biomass",
            },
            "modeldata": ["./modeldata", "./othermodels"],
            "obsdata": {
                "GlobalCarbon": {
                    "version": "5.1",
                    "title": "Global forest live biomass carbon",
                },
                "NBCD2000": {
                    "version": "unknown",
                    "name": "National Biomass and Carbon data set for the Year 2000",
                },
            },
            "log": "cmec_output.log",
        },
        "index": "index.html",
        "data": {
            "gpp_bias": {
                "filename": "gpp_bias.nc",
                "long_name": "mean gpp bias",
                "description": "bias",
            },
        },
        "html": None,
        "metrics": None,
        "plots": None,
    }


@pytest.fixture
def cmec_right_metric_dict():
    return {
        "DIMENSIONS": {
            "json_structure": ["model", "metric", "statistic"],
            "model": {
                "E3SM": {"name": "E3SM"},
                "CESM2": {"name": "CESM2"},
                "IPSL-CM5A-LR": {"name": "IPSL-CM5A-LR"},
            },
            "metric": {
                "Ecosystem and Carbon Cycle": {"name": "Ecosystem and Carbon Cycle"},
                "Hydrology Cycle": {"name": "Hydrology Cycle"},
            },
            "statistic": {
                "overall score": {"name": "overall sccore", "units": "-"},
                "bias": {"name": "mean bias", "units": "inherit"},
                "rmse": {"name": "root mean squre error", "units": "inherit"},
            },
        },
        "RESULTS": {
            "E3SM": {
                "Ecosystem and Carbon Cycle": {"overall score": 0.11, "bias": 0.56, "rmse": -999.0},
                "Hydrology Cycle": {"overall score": 0.26, "bias": 0.70, "rmse": -999.0},
            },
            "CESM2": {
                "Ecosystem and Carbon Cycle": {"overall score": 0.05, "bias": 0.72, "rmse": -999.0},
                "Hydrology Cycle": {"overall score": 0.61, "bias": 0.18, "rmse": -999.0},
            },
            "IPSL-CM5A-LR": {
                "Ecosystem and Carbon Cycle": {
                    "overall score": 0.08,
                    "bias": 0.92,
                    "rmse": 0.34,
                },
                "Hydrology Cycle": {"overall score": 0.67, "bias": -999.0, "rmse": 0.68},
            },
        },
        "DISCLAIMER": {},
        "NOTES": {},
        "PROVENANCE": {},
    }


@pytest.fixture(params=["dict", "CMECMetric"])
def cmec_right_metric_data(request, cmec_right_metric_dict):
    if request.param == "dict":
        return cmec_right_metric_dict
    elif request.param == "CMECMetric":
        return CMECMetric(**cmec_right_metric_dict)


@pytest.fixture(params=["dict", "CMECOutput"])
def cmec_right_output_data(request, cmec_right_output_dict):
    if request.param == "dict":
        return cmec_right_output_dict
    elif request.param == "CMECOutput":
        return CMECOutput(**cmec_right_output_dict)


class TestMetric:
    def test_provider(self, provider):
        metric = provider.metrics()[0]
        assert isinstance(metric.provider, MetricsProvider)

    def test_no_provider(self, mock_metric):
        with pytest.raises(ValueError, match="register .* with a MetricsProvider before using"):
            mock_metric.provider


class TestCommandLineMetric:
    def test_run(self, mocker):
        mocker.patch.object(
            CommandLineMetricsProvider,
            "run",
            create_autospec=True,
        )

        provider = CommandLineMetricsProvider("provider_name", "v0.23")

        metric_result = mocker.sentinel.result
        cmd = mocker.sentinel.cmd
        run_definition = mocker.sentinel.definition

        class TestMetric(CommandLineMetric):
            name = "test-metric"
            slug = "test-metric"
            data_requirements = mocker.Mock()

            def build_cmd(self, definition):
                assert definition == run_definition
                return cmd

            def build_metric_result(self, definition):
                assert definition == run_definition
                return metric_result

        metric = TestMetric()
        provider.register(metric)

        result = metric.run(run_definition)

        provider.run.assert_called_with(cmd)
        assert result == metric_result


class TestMetricResult:
    def test_build_from_output_bundle(
        self,
        cmec_right_output_data,
        cmec_right_output_dict,
        cmec_right_metric_dict,
        tmp_path,
    ):
        definition = MetricExecutionDefinition(
            root_directory=tmp_path,
            output_directory=tmp_path,
            dataset_key="mocked-metric-slug",
            metric_dataset=None,
        )

        result = MetricExecutionResult.build_from_output_bundle(
            definition,
            cmec_output_bundle=cmec_right_output_data,
            cmec_metric_bundle=cmec_right_metric_dict,
        )

        assert result.successful

        # Convert relative path to absolute path
        output_filename = result.to_output_path(result.output_bundle_filename)

        assert output_filename.exists()
        assert output_filename.is_file()
        with open(output_filename) as f:
            cmec_output = json.load(f)
        assert cmec_output == cmec_right_output_dict

        assert output_filename.is_relative_to(tmp_path)

    def test_build_from_metric_bundle(
        self,
        definition_factory,
        cmec_right_metric_data,
        cmec_right_metric_dict,
        cmec_right_output_dict,
        tmp_path,
    ):
        definition = definition_factory()
        # Setting the output directory generally happens as a side effect of the executor
        definition = evolve(definition, output_directory=tmp_path)

        result = MetricExecutionResult.build_from_output_bundle(
            definition,
            cmec_output_bundle=cmec_right_output_dict,
            cmec_metric_bundle=cmec_right_metric_data,
        )

        assert result.successful

        # Convert relative path to absolute path
        output_filename = result.to_output_path(result.metric_bundle_filename)

        assert output_filename.exists()
        assert output_filename.is_file()
        with open(output_filename) as f:
            cmec_metric = json.load(f)

        assert cmec_metric == cmec_right_metric_dict

        assert output_filename.is_relative_to(tmp_path)

    def test_build_from_failure(self, tmp_path):
        definition = MetricExecutionDefinition(
            root_directory=tmp_path,
            output_directory=tmp_path,
            dataset_key="mocked-metric-slug",
            metric_dataset=None,
        )
        result = MetricExecutionResult.build_from_failure(definition)

        assert not result.successful
        assert result.output_bundle_filename is None
        assert result.metric_bundle_filename is None
        assert result.definition == definition


@pytest.fixture
def apply_data_catalog():
    return pd.DataFrame(
        {
            "variable": ["tas", "pr", "rsut", "tas", "tas"],
            "source_id": ["CESM2", "CESM2", "CESM2", "ACCESS", "CAS"],
        }
    )


@pytest.mark.parametrize(
    "facet_filter, expected_data, expected_index",
    [
        (
            {"variable": "tas"},
            {
                "variable": ["tas", "tas", "tas"],
                "source_id": [
                    "CESM2",
                    "ACCESS",
                    "CAS",
                ],
            },
            [0, 3, 4],
        ),
        (
            {"variable": "tas", "source_id": ["CESM2", "ACCESS"]},
            {
                "variable": ["tas", "tas"],
                "source_id": [
                    "CESM2",
                    "ACCESS",
                ],
            },
            [0, 3],
        ),
    ],
)
def test_apply_filters_single(apply_data_catalog, facet_filter, expected_data, expected_index):
    requirement = DataRequirement(
        source_type=SourceDatasetType.CMIP6,
        filters=(FacetFilter(facet_filter),),
        group_by=None,
    )

    filtered = requirement.apply_filters(apply_data_catalog)

    pd.testing.assert_frame_equal(
        filtered,
        pd.DataFrame(
            expected_data,
            index=expected_index,
        ),
    )


def test_apply_filters_multi(apply_data_catalog):
    requirement = DataRequirement(
        source_type=SourceDatasetType.CMIP6,
        filters=(
            FacetFilter({"variable": "tas"}),
            FacetFilter({"source_id": "ACCESS"}, keep=False),
        ),
        group_by=None,
    )

    filtered = requirement.apply_filters(apply_data_catalog)

    pd.testing.assert_frame_equal(
        filtered,
        pd.DataFrame(
            {
                "variable": ["tas", "tas"],
                "source_id": ["CESM2", "CAS"],
            },
            index=[0, 4],
        ),
    )


def test_apply_filters_dont_keep(apply_data_catalog):
    requirement = DataRequirement(
        source_type=SourceDatasetType.CMIP6,
        filters=(FacetFilter({"variable": "tas"}, keep=False),),
        group_by=None,
    )

    filtered = requirement.apply_filters(apply_data_catalog)

    pd.testing.assert_frame_equal(
        filtered,
        pd.DataFrame(
            {
                "variable": ["pr", "rsut"],
                "source_id": [
                    "CESM2",
                    "CESM2",
                ],
            },
            index=[1, 2],
        ),
    )


def test_apply_filters_dont_keep_multifacet(apply_data_catalog):
    """Test that all facet values must match to exclude a file from the catalog."""
    requirement = DataRequirement(
        source_type=SourceDatasetType.CMIP6,
        filters=(
            FacetFilter(
                {
                    "variable": "tas",
                    "source_id": "CAS",
                },
                keep=False,
            ),
        ),
        group_by=None,
    )

    filtered = requirement.apply_filters(apply_data_catalog)
    pd.testing.assert_frame_equal(
        filtered,
        pd.DataFrame(
            {
                "variable": ["tas", "pr", "rsut", "tas"],
                "source_id": [
                    "CESM2",
                    "CESM2",
                    "CESM2",
                    "ACCESS",
                ],
            },
            index=[0, 1, 2, 3],
        ),
    )


def test_apply_filters_missing(apply_data_catalog):
    requirement = DataRequirement(
        source_type=SourceDatasetType.CMIP6,
        filters=(FacetFilter({"missing": "tas"}, keep=False),),
        group_by=None,
    )

    with pytest.raises(
        KeyError,
        match=re.escape("Facet 'missing' not in data catalog columns: ['variable', 'source_id']"),
    ):
        requirement.apply_filters(apply_data_catalog)


@pytest.mark.parametrize(
    "input_path, expected",
    (
        (Path("/example/test"), Path("test")),
        ("/example/test", Path("test")),
        ("/example/test/other", Path("test/other")),
        ("test/other", Path("test/other")),
        (Path("test/other"), Path("test/other")),
    ),
)
def test_ensure_relative_path(input_path, expected):
    assert ensure_relative_path(input_path, root_directory=Path("/example")) == expected


@pytest.mark.parametrize(
    "input_path, expected",
    (
        (Path("example/test"), Path("test")),
        ("example/test", Path("test")),
        ("example/test/other", Path("test/other")),
        ("test/other", Path("test/other")),
        (Path("test/other"), Path("test/other")),
    ),
)
def test_ensure_relative_path_non_absolute(input_path, expected):
    assert ensure_relative_path(input_path, root_directory=Path("example")) == expected


def test_ensure_relative_path_failed():
    with pytest.raises(ValueError):
        ensure_relative_path("/other", root_directory=Path("/example"))
