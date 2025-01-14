from cmip_ref_metrics_ilamb import __version__, provider


def test_provider():
    assert provider.name == "ILAMB"
    assert provider.slug == "ilamb"
    assert provider.version == __version__

    assert len(provider) == 1
