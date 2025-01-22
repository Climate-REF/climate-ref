import importlib.metadata

from cmip_ref_metrics_esmvaltool import __version__, provider


def test_provider():
    assert provider.name == "ESMValTool"
    assert provider.slug == "esmvaltool"
    assert provider.version == __version__

    metric_modules = importlib.resources.files("cmip_ref_metrics_esmvaltool").glob("metrics/*.py")
    ignore = {
        "__init__.py",
        "base.py",
    }
    n_metric_modules = len([f for f in metric_modules if f.name not in ignore])
    assert len(provider) == n_metric_modules
