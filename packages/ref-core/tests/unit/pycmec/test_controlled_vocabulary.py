import pytest

from cmip_ref_core.exceptions import ResultValidationError
from cmip_ref_core.pycmec.controlled_vocabulary import CV, Dimension
from cmip_ref_core.pycmec.metric import CMECMetric


@pytest.fixture
def cv(datadir):
    return CV.load_from_file(datadir / "cv_sample.yaml")


def test_load_from_file(datadir):
    cv = CV.load_from_file(str(datadir / "cv_sample.yaml"))

    assert len(cv.dimensions)


def test_validate(cv, cmec_metric):
    cv.validate_metrics(cmec_metric)


def test_cv_duplicate_dimension(cv):
    with pytest.raises(ValueError):
        CV(
            dimensions=(
                Dimension(
                    name="model",
                    long_name="Model",
                    description="Model name",
                    allow_extra_values=True,
                    required=True,
                ),
                Dimension(
                    name="model",
                    long_name="Model",
                    description="Model name",
                    allow_extra_values=True,
                    required=True,
                ),
            )
        )


def test_cv_reserved_dimension(cv):
    with pytest.raises(ValueError):
        CV(
            dimensions=(
                Dimension(
                    name="attributes",
                    long_name="Model",
                    description="Model name",
                    allow_extra_values=True,
                    required=True,
                ),
            )
        )


def test_invalid_dimension(cv, cmec_metric):
    cmec_metric = CMECMetric(
        DIMENSIONS={
            "json_structure": ["model", "extra"],
            "model": {
                "E3SM": {"name": "E3SM"},
            },
            "extra": {
                "Ecosystem and Carbon Cycle": {"name": "Ecosystem and Carbon Cycle"},
                "Hydrology Cycle": {"name": "Hydrology Cycle"},
            },
        },
        RESULTS={
            "E3SM": {
                "Ecosystem and Carbon Cycle": {"overall score": 0.11, "bias": 0.56},
                "Hydrology Cycle": {"overall score": 0.26, "bias": 0.70},
            },
        },
    )
    with pytest.raises(ResultValidationError, match="Unknown dimension: 'extra'"):
        cv.validate_metrics(cmec_metric)


def test_missing_value(cv, cmec_metric):
    cmec_metric = CMECMetric(
        DIMENSIONS={
            "json_structure": ["model", "metric"],
            "model": {
                "E3SM": {"name": "E3SM"},
            },
            "metric": {
                "Hydrology Cycle": {"name": "Hydrology Cycle"},
            },
        },
        RESULTS={
            "E3SM": {
                "Hydrology Cycle": {"unknown": 0.26, "bias": 0.70},
            },
        },
    )
    with pytest.raises(ResultValidationError, match="Unknown value 'unknown' for dimension 'statistic'"):
        cv.validate_metrics(cmec_metric)
