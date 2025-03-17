import pytest

from cmip_ref_core.pycmec.controlled_vocabulary import CV


@pytest.fixture
def cv(datadir):
    return CV.load_from_file(datadir / "cv_sample.yaml")


def test_load_from_file(datadir):
    cv = CV.load_from_file(str(datadir / "cv_sample.yaml"))

    assert len(cv.dimensions)


def test_validate(cv, cmec_metric):
    cv.validate_metrics(cmec_metric)


def test_validate_extra_dimension(cv, cmec_metric):
    cmec_metric.merge()
    cv.validate_metrics(cmec_metric)
