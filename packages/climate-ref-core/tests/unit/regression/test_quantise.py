"""Unit tests for :mod:`climate_ref_core.regression._quantise`."""

import math

import pytest

from climate_ref_core.regression._quantise import round_floats


@pytest.mark.parametrize(
    "value, expected",
    [
        (1.23456789, 1.234568),
        (0.000123456789, 0.0001234568),
        (123456789.0, 123456800.0),
        (-1.23456789, -1.234568),
        (3.14159265358979, 3.141593),
    ],
)
def test_round_floats_seven_sig_figs(value, expected):
    """A bare float is rounded to seven significant figures."""
    assert round_floats(value) == expected


def test_round_floats_is_idempotent():
    """Rounding an already-rounded value returns it unchanged."""
    once = round_floats(1.23456789)
    twice = round_floats(once)
    assert once == twice


@pytest.mark.parametrize("value", [0, 1, -42, 1_000_000_000_000])
def test_round_floats_leaves_ints_untouched(value):
    """Ints pass through unchanged and stay ints."""
    result = round_floats(value)
    assert result == value
    assert type(result) is int


@pytest.mark.parametrize("value", [True, False])
def test_round_floats_leaves_bools_untouched(value):
    """Bools are not treated as floats and stay bools (bool subclasses int)."""
    result = round_floats(value)
    assert result is value
    assert type(result) is bool


@pytest.mark.parametrize("value", ["1.23456789", "hello", "", None])
def test_round_floats_leaves_strings_and_none_untouched(value):
    """Strings and None pass through unchanged."""
    assert round_floats(value) == value if value is not None else round_floats(value) is None


def test_round_floats_recurses_into_nested_dict():
    """Floats nested in dicts (incl. nested dicts) are rounded; other types preserved."""
    obj = {
        "a": 1.23456789,
        "b": {"c": 0.000123456789, "d": "text", "e": 7},
        "flag": True,
        "n": None,
    }
    result = round_floats(obj)
    assert result == {
        "a": 1.234568,
        "b": {"c": 0.0001234568, "d": "text", "e": 7},
        "flag": True,
        "n": None,
    }
    assert type(result["flag"]) is bool
    assert type(result["b"]["e"]) is int


def test_round_floats_recurses_into_lists():
    """Floats in lists (incl. nested lists) are rounded."""
    obj = [1.23456789, [0.000123456789, "x"], 5]
    assert round_floats(obj) == [1.234568, [0.0001234568, "x"], 5]


def test_round_floats_handles_list_of_dicts():
    """series.json is a list of per-series dicts; each is rounded recursively."""
    obj = [
        {"values": [1.23456789, 2.345678912], "index": ["2000-01-16T12:00:00"]},
        {"values": [9.87654321], "index": ["2001-01-16T12:00:00"]},
    ]
    assert round_floats(obj) == [
        {"values": [1.234568, 2.345679], "index": ["2000-01-16T12:00:00"]},
        {"values": [9.876543], "index": ["2001-01-16T12:00:00"]},
    ]


def test_round_floats_rounds_tuples_to_lists():
    """Tuples are walked recursively (returned as lists, matching JSON semantics)."""
    assert round_floats((1.23456789, "x", 3)) == [1.234568, "x", 3]


def test_round_floats_custom_sig_figs():
    """The number of significant figures is configurable."""
    assert round_floats(1.23456789, sig_figs=3) == 1.23


@pytest.mark.parametrize(
    "value",
    [
        1.843240715970751,
        2.813496471229112,
        0.2389045018665123,
        123456789.987654,
        1.0e-12,
        -9.99999999e10,
    ],
)
def test_round_floats_stays_within_compare_tolerance(value):
    """
    Rounding error stays an order of magnitude under the regression compare tolerance
    (``rtol=1e-6``/``atol=1e-8``), so it can never flip a boundary gate verdict.
    """
    rounded = round_floats(value)
    assert math.isclose(value, rounded, rel_tol=1e-6, abs_tol=1e-8)
