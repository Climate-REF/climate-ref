import json
import pathlib

import pytest

from cmip_ref_core.pycmec.metric import CMECMetric


# Update the original_datadir to specify where the expected values go
@pytest.fixture(scope="session")
def original_datadir():
    return pathlib.Path(__file__).parent / "cmec_testdata"


@pytest.fixture
def cmec_right_metric_dict(datadir):
    with open(datadir / "cmec_metric_sample.json") as fh:
        content = json.loads(fh.read())

    return content


@pytest.fixture
def cmec_metric(cmec_right_metric_dict):
    return CMECMetric(**cmec_right_metric_dict)
