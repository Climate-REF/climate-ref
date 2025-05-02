import importlib.metadata

from cmip_ref_metrics_esmvaltool import __version__, provider


def test_provider():
    assert provider.name == "ESMValTool"
    assert provider.slug == "esmvaltool"
    assert provider.version == __version__

    metric_modules = importlib.resources.files("cmip_ref_metrics_esmvaltool").glob("metrics/*.py")
    diagnostics_per_module = {
        "__init__.py": 0,
        "base.py": 0,
        "cloud_scatterplots.py": 5,
    }
    n_diagnostics = sum(diagnostics_per_module.get(f.name, 1) for f in metric_modules)
    assert len(provider) == n_diagnostics
