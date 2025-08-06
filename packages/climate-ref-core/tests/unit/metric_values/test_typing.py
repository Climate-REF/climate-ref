import re
from pathlib import Path

import pytest

from climate_ref_core.metric_values.typing import SeriesMetricValue


class TestSeriesMetricValue:
    def test_matching_lengths(self):
        value = SeriesMetricValue(
            dimensions={"model": "test"},
            values=[1.0, 2.0, 3.0],
            index=[0, 1, 2],
            index_name="time",
            attributes={"attr": "value"},
        )
        assert value.values == [1.0, 2.0, 3.0]
        assert value.index == [0, 1, 2]
        assert value.index_name == "time"
        assert value.dimensions == {"model": "test"}
        assert value.attributes == {"attr": "value"}

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError, match=re.escape("Index length (2) must match values length (3)")):
            SeriesMetricValue(
                dimensions={"model": "test"},
                values=[1.0, 2.0, 3.0],
                index=[0, 1],
                index_name="time",
                attributes=None,
            )

    @pytest.mark.parametrize(
        "index",
        [
            [0, "1", 2.0],
            ["apr", "may", "jun"],
        ],
    )
    def test_index_types(self, index):
        value = SeriesMetricValue(
            dimensions={"model": "test"},
            values=[1.0, 2.0, 3.0],
            index=index,
            index_name="time",
        )
        assert value.values == [1.0, 2.0, 3.0]
        assert value.index == index

    def test_str_values(self):
        value = SeriesMetricValue(
            dimensions={"model": "test"},
            values=["1"],
            index=[1.0],
            index_name="time",
        )
        assert value.values == [1.0]

        with pytest.raises(
            ValueError, match="Input should be a valid number, unable to parse string as a number"
        ):
            SeriesMetricValue(
                dimensions={"model": "test"},
                values=["a"],
                index=[1.0],
                index_name="time",
            )

    def test_dump_and_load_json(self, tmp_path: Path):
        series = [
            SeriesMetricValue(
                dimensions={"model": "test1"},
                values=[1.0, 2.0, 3.0],
                index=[0, 1, 2],
                index_name="time",
                attributes={"attr": "value1"},
            ),
            SeriesMetricValue(
                dimensions={"model": "test2"},
                values=[4.0, 5.0],
                index=["a", "b"],
                index_name="other",
                attributes=None,
            ),
        ]
        path = tmp_path / "test.json"

        SeriesMetricValue.dump_to_json(path, series)
        loaded_series = SeriesMetricValue.load_from_json(path)

        assert loaded_series == series

    def test_load_from_json_not_a_list(self, tmp_path: Path):
        path = tmp_path / "test.json"
        path.write_text('{"not": "a list"}')

        with pytest.raises(ValueError, match="Expected a list of series values, got <class 'dict'>"):
            SeriesMetricValue.load_from_json(path)
