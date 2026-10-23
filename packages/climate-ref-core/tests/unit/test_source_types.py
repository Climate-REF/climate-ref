from climate_ref_core.source_types import SourceDatasetType


def test_is_model_output():
    model = {t for t in SourceDatasetType if t.is_model_output}
    assert model == {SourceDatasetType.CMIP6, SourceDatasetType.CMIP7}
