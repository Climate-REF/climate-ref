import importlib.metadata

from climate_ref_esmvaltool import __version__, provider


def test_provider():
    assert provider.name == "ESMValTool"
    assert provider.slug == "esmvaltool"
    assert provider.version == __version__

    diagnostic_modules = importlib.resources.files("climate_ref_esmvaltool").glob("diagnostics/*.py")
    diagnostics_per_module = {
        "__init__.py": 0,
        "base.py": 0,
        "cloud_scatterplots.py": 5,
        "enso.py": 2,
        "regional_historical_changes.py": 3,
    }
    n_diagnostics = sum(diagnostics_per_module.get(f.name, 1) for f in diagnostic_modules)
    assert len(provider) == n_diagnostics
