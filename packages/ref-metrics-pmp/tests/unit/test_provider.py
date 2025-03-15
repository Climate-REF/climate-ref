from cmip_ref_metrics_pmp import __version__, provider


def test_provider():
    assert provider.name == "PMP"
    assert provider.slug == "pmp"
    assert provider.version == __version__

    assert len(provider) == 2
