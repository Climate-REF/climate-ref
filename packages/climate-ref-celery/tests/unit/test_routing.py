import pytest
from climate_ref_celery.routing import (
    ROUTES_ENV_VAR,
    RoutingTable,
    RoutingTableError,
    load_routing_table,
)

EXAMPLE = """
default = "medium"

[esmvaltool]
default = "medium"
rules = [
  { match = "portrait-*", size = "large" },
  { match = "climate-at-global-warming-level", size = "large" },
  { match = "sea-ice-basic", size = "small" },
]

[ilamb]
default = "small"
"""


@pytest.fixture
def routes_file(tmp_path):
    def write(content):
        path = tmp_path / "routes.toml"
        path.write_text(content)
        return path

    return write


class TestQueueFor:
    @pytest.fixture
    def table(self, routes_file):
        return RoutingTable.from_file(routes_file(EXAMPLE))

    @pytest.mark.parametrize(
        "provider, diagnostic, expected",
        [
            ("esmvaltool", "portrait-plot", "esmvaltool-large"),
            ("esmvaltool", "climate-at-global-warming-level", "esmvaltool-large"),
            ("esmvaltool", "sea-ice-basic", "esmvaltool-small"),
            ("esmvaltool", "anything-else", "esmvaltool-medium"),
            ("ilamb", "gpp-fluxnet2015", "ilamb-small"),
            ("pmp", "annual-cycle", "pmp-medium"),
        ],
    )
    def test_resolution(self, table, provider, diagnostic, expected):
        assert table.queue_for(provider, diagnostic) == expected

    def test_first_match_wins(self, routes_file):
        table = RoutingTable.from_file(
            routes_file(
                """
                [pmp]
                rules = [
                  { match = "annual-*", size = "large" },
                  { match = "annual-cycle", size = "small" },
                ]
                """
            )
        )
        assert table.queue_for("pmp", "annual-cycle") == "pmp-large"

    def test_no_match_no_defaults(self, routes_file):
        table = RoutingTable.from_file(
            routes_file(
                """
                [pmp]
                rules = [{ match = "annual-*", size = "large" }]
                """
            )
        )
        assert table.queue_for("pmp", "variability-modes") == "pmp"
        assert table.queue_for("ilamb", "gpp") == "ilamb"

    def test_empty_table(self):
        assert RoutingTable().queue_for("pmp", "annual-cycle") == "pmp"


class TestValidation:
    def test_missing_file(self, tmp_path):
        path = tmp_path / "missing.toml"
        with pytest.raises(RoutingTableError, match=str(path)):
            RoutingTable.from_file(path)

    def test_invalid_toml(self, routes_file):
        with pytest.raises(RoutingTableError, match="invalid TOML"):
            RoutingTable.from_file(routes_file("not = ["))

    def test_non_string_default(self, routes_file):
        with pytest.raises(RoutingTableError, match="top-level 'default'"):
            RoutingTable.from_file(routes_file("default = 3"))

    def test_non_string_size(self, routes_file):
        with pytest.raises(RoutingTableError, match=r"\[pmp\] rule 0 'size'"):
            RoutingTable.from_file(routes_file('[pmp]\nrules = [{ match = "a", size = 2 }]'))

    def test_missing_rule_keys(self, routes_file):
        with pytest.raises(RoutingTableError, match=r"\[pmp\] rule 0"):
            RoutingTable.from_file(routes_file('[pmp]\nrules = [{ match = "a" }]'))

    def test_unknown_provider_key(self, routes_file):
        with pytest.raises(RoutingTableError, match=r"\[pmp\] has unknown keys \['rule'\]"):
            RoutingTable.from_file(routes_file("[pmp]\nrule = []"))

    def test_provider_entry_not_a_table(self, routes_file):
        with pytest.raises(RoutingTableError, match="'pmp' must be a provider table"):
            RoutingTable.from_file(routes_file('pmp = "small"'))

    def test_unregistered_provider_warns(self, routes_file, caplog):
        RoutingTable.from_file(routes_file(EXAMPLE), known_providers=["esmvaltool"])
        assert "provider 'ilamb' is not currently registered" in caplog.text

    def test_registered_providers_no_warning(self, routes_file, caplog):
        RoutingTable.from_file(routes_file(EXAMPLE), known_providers=["esmvaltool", "ilamb"])
        assert "not currently registered" not in caplog.text

    def test_duplicate_pattern_warns(self, routes_file, caplog):
        table = RoutingTable.from_file(
            routes_file(
                """
                [pmp]
                rules = [
                  { match = "annual-*", size = "large" },
                  { match = "annual-*", size = "small" },
                ]
                """
            )
        )
        assert "duplicate pattern 'annual-*'" in caplog.text
        assert table.queue_for("pmp", "annual-cycle") == "pmp-large"


class TestLoadRoutingTable:
    def test_unset(self, monkeypatch):
        monkeypatch.delenv(ROUTES_ENV_VAR, raising=False)
        assert load_routing_table() == RoutingTable()

    def test_set(self, monkeypatch, routes_file):
        monkeypatch.setenv(ROUTES_ENV_VAR, str(routes_file(EXAMPLE)))
        table = load_routing_table()
        assert table.queue_for("ilamb", "gpp") == "ilamb-small"
