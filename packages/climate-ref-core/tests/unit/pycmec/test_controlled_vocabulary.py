import pytest

from climate_ref_core.exceptions import ResultValidationError
from climate_ref_core.pycmec import BUNDLED_AFT_CV
from climate_ref_core.pycmec.controlled_vocabulary import CV, Dimension
from climate_ref_core.pycmec.metric import CMECMetric


@pytest.fixture
def cv(datadir):
    return CV.load_from_file(datadir / "cv_sample.yaml")


def test_load_from_file(datadir):
    cv = CV.load_from_file(str(datadir / "cv_sample.yaml"))

    assert len(cv.dimensions)


def test_load_from_file_failed(tmp_path):
    (tmp_path / "cv_sample.yaml").touch()
    with pytest.raises(AttributeError, match="'NoneType' object has no attribute 'keys'"):
        CV.load_from_file(tmp_path / "cv_sample.yaml")


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
            "json_structure": ["model", "extra", "statistic"],
            "model": {
                "E3SM": {"name": "E3SM"},
            },
            "extra": {
                "Ecosystem and Carbon Cycle": {"name": "Ecosystem and Carbon Cycle"},
                "Hydrology Cycle": {"name": "Hydrology Cycle"},
            },
            "statistic": {
                "overall score": {"name": "overall score", "units": "-"},
                "bias": {"name": "mean bias", "units": "inherit"},
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
            "json_structure": ["model", "metric", "statistic"],
            "model": {
                "E3SM": {"name": "E3SM"},
            },
            "metric": {
                "Hydrology Cycle": {"name": "Hydrology Cycle"},
            },
            "statistic": {
                "unknown": {"name": "unknown", "units": "-"},
                "bias": {"name": "mean bias", "units": "inherit"},
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


def test_load_from_file_round_trips(tmp_path):
    """The path-based loader is kept for callers outside this repository."""
    source = tmp_path / "cv.yaml"
    source.write_text(BUNDLED_AFT_CV.read_text(), encoding="utf-8")

    assert CV.load_from_file(source) == CV.load(BUNDLED_AFT_CV)
