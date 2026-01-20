import pytest

from climate_ref.cli._utils import format_size, parse_facet_filters


class TestFormatSize:
    """Tests for format_size function."""

    @pytest.mark.parametrize(
        "size_bytes,expected",
        [
            (0, "0.0 B"),
            (1, "1.0 B"),
            (512, "512.0 B"),
            (1023, "1023.0 B"),
            (1024, "1.0 KB"),
            (1536, "1.5 KB"),
            (1048576, "1.0 MB"),
            (1572864, "1.5 MB"),
            (1073741824, "1.0 GB"),
            (1610612736, "1.5 GB"),
            (1099511627776, "1.0 TB"),
            (1649267441664, "1.5 TB"),
        ],
    )
    def test_format_size_various_values(self, size_bytes, expected):
        """Test format_size with various byte values."""
        assert format_size(size_bytes) == expected

    def test_format_size_with_float_input(self):
        """Test format_size handles float input."""
        assert format_size(1024.5) == "1.0 KB"
        assert format_size(1536.0) == "1.5 KB"

    def test_format_size_boundary_values(self):
        """Test format_size at unit boundaries."""
        # Just under 1KB
        assert format_size(1023) == "1023.0 B"
        # Exactly 1KB
        assert format_size(1024) == "1.0 KB"
        # Just under 1MB
        assert format_size(1048575) == "1024.0 KB"
        # Exactly 1MB
        assert format_size(1048576) == "1.0 MB"


def test_parse_facet_filters_valid_input():
    filters = ["source_id=GFDL-ESM4", "variable_id=tas"]
    expected = {"source_id": "GFDL-ESM4", "variable_id": "tas"}
    assert parse_facet_filters(filters) == expected


def test_parse_facet_filters_empty_list():
    assert parse_facet_filters([]) == {}


def test_parse_facet_filters_none_input():
    assert parse_facet_filters(None) == {}


def test_parse_facet_filters_with_whitespace():
    filters = ["  key1 = value1  ", "key2=value2 "]
    expected = {"key1": "value1", "key2": "value2"}
    assert parse_facet_filters(filters) == expected


def test_parse_facet_filters_duplicate_key(caplog):
    filters = ["key=value1", "key=value2"]
    expected = {"key": "value2"}
    with caplog.at_level("WARNING"):
        result = parse_facet_filters(filters)
    assert result == expected
    assert "Filter key 'key' specified multiple times. Using last value: 'value2'" in caplog.text


def test_parse_facet_filters_invalid_format_no_equals():
    with pytest.raises(ValueError, match="Invalid filter format: 'no_equals_sign'"):
        parse_facet_filters(["no_equals_sign"])


def test_parse_facet_filters_empty_key():
    with pytest.raises(ValueError, match="Empty key in filter: '=value'"):
        parse_facet_filters(["=value"])


def test_parse_facet_filters_empty_value():
    with pytest.raises(ValueError, match="Empty value in filter: 'key='"):
        parse_facet_filters(["key="])


def test_parse_facet_filters_value_with_equals():
    filters = ["query=some_key=some_value"]
    expected = {"query": "some_key=some_value"}
    assert parse_facet_filters(filters) == expected


def test_parse_facet_filters_mixed_valid_and_invalid(caplog):
    filters = ["key1=value1", "invalid", "key2=value2"]
    with pytest.raises(ValueError, match="Invalid filter format: 'invalid'"):
        parse_facet_filters(filters)
    assert not caplog.text
