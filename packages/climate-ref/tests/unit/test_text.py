"""
Tests for the helpers behind the text the REF shows a person.
"""

import pytest

from climate_ref.text import pluralise


@pytest.mark.parametrize(
    "count, expected",
    [(0, "0 diagnostics"), (1, "1 diagnostic"), (2, "2 diagnostics")],
)
def test_pluralise(count, expected):
    assert pluralise(count, "diagnostic") == expected


def test_pluralise_with_an_irregular_plural():
    assert pluralise(2, "registry", "registries") == "2 registries"
