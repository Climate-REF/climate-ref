"""
Tests for the doctor check registry, including the checks a plugin contributes.
"""

import importlib.metadata

import pytest

from climate_ref.doctor import DoctorContext, Finding, Severity, iter_checks
from climate_ref.doctor import registry as registry_module
from climate_ref.doctor.registry import RegisteredCheck, run_checks

BUILT_IN_SLUGS = {
    "duplicate-coverage",
    "missing-reference-data",
    "unreachable-source-type",
    "overlapping-registries",
}


@pytest.fixture
def isolated_registry(monkeypatch):
    """
    Give a test its own registry, so what it registers does not leak into the others.

    Returns the module so a test can call `register_check` or `load_plugin_checks` against it.
    """
    monkeypatch.setattr(registry_module, "_REGISTRY", dict(registry_module._REGISTRY))
    monkeypatch.setattr(registry_module, "_LOAD_ERRORS", {})
    monkeypatch.setattr(registry_module, "_LOADED_PLUGINS", set())
    return registry_module


def _entry_points(monkeypatch, entry_points):
    """Make `importlib.metadata.entry_points` return exactly ``entry_points`` for our group."""

    def _fake(group):
        assert group == registry_module.CHECK_ENTRY_POINT_GROUP
        return entry_points

    monkeypatch.setattr(importlib.metadata, "entry_points", _fake)


class _EntryPoint:
    """A stand-in for a plugin's entry point, whose load runs ``register``."""

    def __init__(self, name, register):
        self.name = name
        self.value = f"{name}:checks"
        self._register = register

    def load(self):
        return self._register()


def _context():
    return DoctorContext.from_catalogs({}, [])


class TestRegistration:
    def test_the_built_in_checks_are_registered(self):
        assert BUILT_IN_SLUGS <= {registered.slug for registered in iter_checks()}

    def test_every_built_in_check_is_built_in(self):
        sources = {c.slug: c.source for c in iter_checks() if c.slug in BUILT_IN_SLUGS}

        assert set(sources.values()) == {registry_module.BUILT_IN}

    def test_every_check_describes_itself(self):
        assert all(registered.description for registered in iter_checks())

    def test_a_duplicate_slug_is_refused(self, isolated_registry):
        # Two checks sharing a slug would be indistinguishable in the output.
        with pytest.raises(ValueError, match="duplicate-coverage"):
            isolated_registry.register_check(
                RegisteredCheck(slug="duplicate-coverage", description="clash", func=lambda ctx: [])
            )

    def test_the_decorator_returns_the_function_unchanged(self, isolated_registry):
        def find_nothing(context):
            return []

        decorated = isolated_registry.check("decorated", "A check")(find_nothing)

        assert decorated is find_nothing
        assert decorated in [c.func for c in isolated_registry.iter_checks()]


class TestPluginChecks:
    def test_a_plugin_check_is_collected_and_run(self, isolated_registry, monkeypatch):
        def register():
            isolated_registry.check("from-plugin", "Contributed by a plugin")(
                lambda context: [Finding(severity=Severity.WARNING, summary="a plugin finding")]
            )

        _entry_points(monkeypatch, [_EntryPoint("my_provider", register)])

        findings = run_checks(_context())

        plugin_findings = [f for f in findings if f.check == "from-plugin"]
        assert len(plugin_findings) == 1
        assert plugin_findings[0].summary == "a plugin finding"

    def test_a_plugin_check_is_attributed_to_its_entry_point(self, isolated_registry, monkeypatch):
        def register():
            isolated_registry.check("from-plugin", "Contributed by a plugin")(lambda context: [])

        _entry_points(monkeypatch, [_EntryPoint("my_provider", register)])

        sources = {c.slug: c.source for c in isolated_registry.iter_checks()}

        assert sources["from-plugin"] == "my_provider"
        assert sources["duplicate-coverage"] == registry_module.BUILT_IN

    def test_plugins_are_loaded_once(self, isolated_registry, monkeypatch):
        calls = []

        def register():
            calls.append(1)
            isolated_registry.check(f"plugin-{len(calls)}", "A check")(lambda context: [])

        _entry_points(monkeypatch, [_EntryPoint("my_provider", register)])

        isolated_registry.iter_checks()
        isolated_registry.iter_checks()

        # A second load would raise on the duplicate slug, so this also guards against that.
        assert len(calls) == 1

    def test_a_broken_plugin_is_reported_rather_than_raised(self, isolated_registry, monkeypatch):
        def register():
            raise ImportError("no module named 'nope'")

        _entry_points(monkeypatch, [_EntryPoint("broken_provider", register)])

        findings = run_checks(_context())

        load_findings = [f for f in findings if f.check == "plugin-load"]
        assert len(load_findings) == 1
        assert load_findings[0].severity == Severity.ERROR
        assert "broken_provider" in load_findings[0].summary
        assert "no module named 'nope'" in load_findings[0].detail

    def test_a_broken_plugin_does_not_stop_the_other_checks(self, isolated_registry, monkeypatch):
        def broken():
            raise ImportError("boom")

        def working():
            isolated_registry.check("from-plugin", "A check")(
                lambda context: [Finding(severity=Severity.INFO, summary="still ran")]
            )

        _entry_points(
            monkeypatch,
            [_EntryPoint("broken_provider", broken), _EntryPoint("working_provider", working)],
        )

        checks = {c.slug for c in isolated_registry.iter_checks()}

        assert "from-plugin" in checks
        assert BUILT_IN_SLUGS <= checks
