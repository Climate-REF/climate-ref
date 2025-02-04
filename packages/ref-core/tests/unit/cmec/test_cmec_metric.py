import pytest
from pydantic import ValidationError

from cmip_ref_core.pycmec.metric import (
    CMECMetric,
    MetricDimensions,
    MetricResults,
)


@pytest.fixture
def cmec_right_metric_dict():
    return {
        "SCHEMA": {"name": "cmec", "version": "v1", "package": "ilamb v3"},
        "DIMENSIONS": {
            "json_structure": ["model", "metric"],
            "model": {
                "E3SM": {"name": "E3SM"},
                "CESM2": {"name": "CESM2"},
                "IPSL-CM5A-LR": {"name": "IPSL-CM5A-LR"},
            },
            "metric": {
                "Ecosystem and Carbon Cycle": {"name": "Ecosystem and Carbon Cycle"},
                "Hydrology Cycle": {"name": "Hydrology Cycle"},
            },
        },
        "RESULTS": {
            "E3SM": {
                "Ecosystem and Carbon Cycle": {"overall score": 0.11, "bias": 0.56},
                "Hydrology Cycle": {"overall score": 0.26, "bias": 0.70},
            },
            "CESM2": {
                "Ecosystem and Carbon Cycle": {"overall score": 0.05, "bias": 0.72},
                "Hydrology Cycle": {"overall score": 0.61, "bias": 0.18},
            },
            "IPSL-CM5A-LR": {
                "Ecosystem and Carbon Cycle": {
                    "overall score": 0.08,
                    "bias": 0.92,
                    "rmse": 0.34,
                },
                "Hydrology Cycle": {"overall score": 0.67, "rmse": 0.68},
            },
        },
    }


@pytest.fixture(params=["dict", "CMECMetric"])
def cmec_right_metric_data(request, cmec_right_metric_dict):
    if request.param == "dict":
        return cmec_right_metric_dict
    elif request.param == "CMECMetric":
        return CMECMetric(**cmec_right_metric_dict)


@pytest.fixture(params=["dict", "MetricDimensions"])
def cmec_right_dimen_data(request, cmec_right_metric_dict):
    if request.param == "dict":
        return cmec_right_metric_dict["DIMENSIONS"]
    elif request.param == "MetricDimensions":
        return MetricDimensions(**cmec_right_metric_dict["DIMENSIONS"])


@pytest.fixture
def cmec_right_result_dimobj(cmec_right_metric_dict):
    return cmec_right_metric_dict["RESULTS"], MetricDimensions(**cmec_right_metric_dict["DIMENSIONS"])


@pytest.fixture
def cmec_wrongkw_metric_dict(cmec_right_metric_dict):
    return {
        "DIMENSIONS": {
            "xJSON": cmec_right_metric_dict["DIMENSIONS"]["json_structure"],
            "xmodel": cmec_right_metric_dict["DIMENSIONS"]["model"],
            "xmetric": cmec_right_metric_dict["DIMENSIONS"]["metric"],
        },
        "RESULTS": cmec_right_metric_dict["RESULTS"],
    }


@pytest.fixture
def cmec_wrongattr_metric_dict(cmec_right_metric_dict):
    return {
        "xDIMENSIONS": cmec_right_metric_dict["DIMENSIONS"],
        "xRESULTS": cmec_right_metric_dict["RESULTS"],
    }


@pytest.fixture
def cmec_wrongdim_metric_dict(cmec_right_metric_dict):
    cmec_right_metric_dict["DIMENSIONS"]["json_structure"] = ["model"]
    return cmec_right_metric_dict


def test_metric_right(cmec_right_metric_data):
    CMECMetric.model_validate(cmec_right_metric_data)


def test_metric_right_init(cmec_right_metric_dict):
    CMECMetric(**cmec_right_metric_dict)


def test_dimen_right(cmec_right_dimen_data):
    MetricDimensions.model_validate(cmec_right_dimen_data)


def test_metric_wrongkm(cmec_wrongkw_metric_dict):
    with pytest.raises(ValidationError):
        CMECMetric.model_validate(cmec_wrongkw_metric_dict)
    with pytest.raises(ValidationError):
        CMECMetric(**cmec_wrongkw_metric_dict)


def test_metric_wrongattr(cmec_wrongattr_metric_dict):
    with pytest.raises(ValidationError):
        CMECMetric.model_validate(cmec_wrongattr_metric_dict)
    with pytest.raises(ValidationError):
        CMECMetric(**cmec_wrongattr_metric_dict)


def test_metric_wrongdim(cmec_wrongdim_metric_dict):
    with pytest.raises(ValidationError):
        CMECMetric.model_validate(cmec_wrongdim_metric_dict)
    with pytest.raises(ValidationError):
        CMECMetric(**cmec_wrongdim_metric_dict)


def test_dimen_wrongdim(cmec_wrongdim_metric_dict):
    with pytest.raises(ValidationError):
        MetricDimensions.model_validate(cmec_wrongdim_metric_dict["DIMENSIONS"])
    with pytest.raises(ValidationError):
        MetricDimensions(DIMENSIONS=cmec_wrongdim_metric_dict["DIMENSIONS"])


@pytest.mark.parametrize(
    "dim_name,dim_dict",
    [
        ("model", {"BCC-CSM2-MR": {"name": "BCC-CSM2-MR"}}),
        (
            "metric",
            {
                "Hydrology Cycle::Sensible Heat!!FLUXNET2015": {
                    "Name": "Hydrology Cycle::Latent Heat!!FLUXNET2015",
                    "Abstract": "benchmark score",
                    "URI": ["https://www.osti.gov/biblio/1330803"],
                }
            },
        ),
    ],
)
def test_add_dimensions_exist_dimen(cmec_right_dimen_data, dim_name, dim_dict):
    if isinstance(cmec_right_dimen_data, MetricDimensions):
        cmec_dims = cmec_right_dimen_data
    else:
        cmec_dims = MetricDimensions(cmec_right_dimen_data)

    js_struct_old = cmec_dims.root["json_structure"]
    cmec_dims.add_dimension(dim_name, dim_dict)
    js_struct_new = cmec_dims.root["json_structure"]

    assert js_struct_old == js_struct_new

    if dim_name == "model":
        assert cmec_dims.root["model"]["BCC-CSM2-MR"] == {"name": "BCC-CSM2-MR"}
    if dim_name == "metric":
        assert cmec_dims.root["metric"]["Hydrology Cycle::Sensible Heat!!FLUXNET2015"] == {
            "Name": "Hydrology Cycle::Latent Heat!!FLUXNET2015",
            "Abstract": "benchmark score",
            "URI": ["https://www.osti.gov/biblio/1330803"],
        }


def test_add_dimensions_new_dimen(cmec_right_dimen_data):
    if isinstance(cmec_right_dimen_data, MetricDimensions):
        cmec_dims = cmec_right_dimen_data
    else:
        cmec_dims = MetricDimensions(cmec_right_dimen_data)

    cmec_dims.add_dimension("region", {"global": {"title": "global"}, "NA": {"title": "North America"}})

    assert "region" in cmec_dims.root["json_structure"]

    assert cmec_dims.root["region"] == {
        "global": {"title": "global"},
        "NA": {"title": "North America"},
    }


def test_validate_result_wo_dim(cmec_right_metric_dict):
    with pytest.raises(ValidationError):
        MetricResults(cmec_right_metric_dict["RESULTS"])


def test_metric_merge():
    import json

    dict_pmp = {
        "DIMENSIONS": {
            "json_structure": ["model", "metric"],
            "model": {"GFDL-ESM2M": {"Source": "CMIP5 ESGF"}},
            "metric": {
                "NinoSstDiversity_2": {"Name": "NinoSstDiversity_2"},
                "BiasTauxLonRmse": {"name": "BiasTauxLonRmse"},
            },
        },
        "RESULTS": {
            "GFDL-ESM2M": {
                "NinoSstDiversity_2": -75,
                "BiasTauxLonRmse": 5.959564210237568,
            }
        },
    }
    dict_ilamb = {
        "DIMENSIONS": {
            "json_structure": ["model", "metric"],
            "model": {"E3SM": {"name": "E3SM"}, "CESM": {"name": "CESM"}},
            "metric": {"carbon": {"name": "carbon"}},
        },
        "RESULTS": {
            "E3SM": {"carbon": {"overall score": 0.11, "bias": 0.56}},
            "CESM": {"carbon": {"overall score": 0.05, "bias": 0.72}},
        },
    }

    dict_merged = {
        "SCHEMA": None,
        "DIMENSIONS": {
            "json_structure": ["model", "metric"],
            "model": {
                "GFDL-ESM2M": {"Source": "CMIP5 ESGF"},
                "E3SM": {"name": "E3SM"},
                "CESM": {"name": "CESM"},
            },
            "metric": {
                "NinoSstDiversity_2": {"Name": "NinoSstDiversity_2"},
                "BiasTauxLonRmse": {"name": "BiasTauxLonRmse"},
                "carbon": {"name": "carbon"},
            },
        },
        "RESULTS": {
            "E3SM": {
                "carbon": {"overall score": 0.11, "bias": 0.56},
                "NinoSstDiversity_2": {},
                "BiasTauxLonRmse": {},
            },
            "CESM": {
                "carbon": {"overall score": 0.05, "bias": 0.72},
                "NinoSstDiversity_2": {},
                "BiasTauxLonRmse": {},
            },
            "GFDL-ESM2M": {
                "NinoSstDiversity_2": -75,
                "BiasTauxLonRmse": 5.959564210237568,
                "carbon": {},
            },
        },
        "PROVENANCE": None,
        "DISCLAIMER": None,
        "NOTES": None,
    }

    assert (
        json.loads(CMECMetric.merge(dict_pmp, dict_ilamb, -9999.0).model_dump_json(indent=2)) == dict_merged
    )


def test_gen_json_schema():
    from cmip_ref_core.pycmec.metric import (
        CMECGenerateJsonSchema,
    )

    cmec_model_schema = CMECMetric.model_json_schema(schema_generator=CMECGenerateJsonSchema)

    assert cmec_model_schema == {
        "$defs": {
            "MetricDimensions": {
                "default": {
                    "json_structure": ["model", "metric"],
                    "model": {},
                    "metric": {},
                },
                "description": "CMEC metric bundle DIMENSIONS object",
                "title": "MetricDimensions",
                "type": "object",
            },
            "MetricSchema": {
                "description": "A metric schema used by unified dasbboard, not required by CMEC",
                "properties": {
                    "name": {"title": "Name", "type": "string"},
                    "version": {"title": "Version", "type": "string"},
                    "package": {"title": "Package", "type": "string"},
                },
                "required": ["name", "version", "package"],
                "title": "MetricSchema",
                "type": "object",
            },
        },
        "description": "CMEC metric bundle object",
        "properties": {
            "SCHEMA": {
                "anyOf": [{"$ref": "#/$defs/MetricSchema"}, {"type": "null"}],
                "default": None,
            },
            "DIMENSIONS": {"$ref": "#/$defs/MetricDimensions"},
            "RESULTS": {"title": "Results", "type": "object"},
            "PROVENANCE": {
                "anyOf": [{"type": "object"}, {"type": "null"}],
                "default": None,
                "title": "Provenance",
            },
            "DISCLAIMER": {
                "anyOf": [{"type": "object"}, {"type": "null"}],
                "default": None,
                "title": "Disclaimer",
            },
            "NOTES": {
                "anyOf": [{"type": "object"}, {"type": "null"}],
                "default": None,
                "title": "Notes",
            },
        },
        "required": ["DIMENSIONS", "RESULTS"],
        "title": "CMEC",
        "type": "object",
        "$schema": "https://json-schema.org/draft/2020-12/schema",
    }
