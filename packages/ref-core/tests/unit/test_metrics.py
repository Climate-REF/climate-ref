import re
from pathlib import Path

import pandas as pd
import pytest

from cmip_ref_core.datasets import FacetFilter, SourceDatasetType
from cmip_ref_core.metrics import (
    DataRequirement,
    MetricExecutionDefinition,
    MetricResult,
    ProposedMetricExecutionDefinition,
)


class TestMetricResult:
    def test_build_from_output_bundle(self, tmp_path):
        definition = MetricExecutionDefinition(
            output_directory=tmp_path, key="mocked-metric-slug", metric_dataset=None
        )

        result = MetricResult.build_from_output_bundle(definition, {"data": "value"})

        assert result.successful

        # Convert relative path to absolute path
        output_filename = result.to_output_path(result.bundle_filename)

        assert output_filename.exists()
        assert output_filename.is_file()
        with open(output_filename) as f:
            assert f.read() == '{"data": "value"}'

        assert output_filename.is_relative_to(tmp_path)

    def test_build_from_failure(self, tmp_path):
        definition = MetricExecutionDefinition(
            output_directory=tmp_path, key="mocked-metric-slug", metric_dataset=None
        )
        result = MetricResult.build_from_failure(definition)

        assert not result.successful
        assert result.bundle_filename is None
        assert result.definition == definition


class TestProposedMetricExecutionDefinition:
    def test_to_metric_execution_definition(self, metric_dataset, mocker):
        metric_dataset.to_abs_paths = mocker.Mock()
        proposed_definition = ProposedMetricExecutionDefinition(
            key="test-key",
            metric_dataset=metric_dataset,
            output_fragment=Path("relative/output"),
        )
        data_directory = Path("/absolute/data")
        scratch_directory = Path("/absolute/scratch")

        metric_execution_definition = proposed_definition.to_metric_execution_definition(
            data_directory=data_directory, scratch_directory=scratch_directory
        )

        assert metric_execution_definition.key == "test-key"

        assert metric_execution_definition.metric_dataset == metric_dataset.to_abs_paths.return_value
        assert metric_execution_definition.output_directory == scratch_directory / "relative/output"


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
