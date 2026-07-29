import json
import pathlib

import pytest
from climate_ref_celery.serialisation import CONTENT_TYPE, SERIALIZER, register_serialisation
from kombu.serialization import dumps, loads

from climate_ref_core.diagnostics import ExecutionResult


@pytest.fixture(autouse=True)
def _registered():
    register_serialisation()


def test_definition_roundtrips_through_kombu(metric_definition):
    content_type, encoding, body = dumps([metric_definition, "INFO"], serializer=SERIALIZER)

    assert content_type == CONTENT_TYPE

    definition, log_level = loads(body, content_type, encoding)

    assert log_level == "INFO"
    assert definition.execution_slug() == metric_definition.execution_slug()
    assert definition.datasets.hash == metric_definition.datasets.hash


def test_result_roundtrips_through_kombu(metric_definition):
    result = ExecutionResult(
        definition=metric_definition,
        metric_bundle_filename=pathlib.Path("diagnostic.json"),
        successful=True,
    )

    content_type, encoding, body = dumps(result, serializer=SERIALIZER)
    decoded = loads(body, content_type, encoding)

    assert decoded.successful is True
    assert decoded.metric_bundle_filename == pathlib.Path("diagnostic.json")
    assert decoded.definition.execution_slug() == metric_definition.execution_slug()


def test_encoded_body_names_no_python_types(metric_definition):
    """A JSON body cannot name a class to import, which is the point of dropping pickle."""
    _, _, body = dumps([metric_definition, "INFO"], serializer=SERIALIZER)

    # Parses as JSON, so there is no opcode stream for a decoder to execute
    definition, _ = json.loads(body)

    assert type(metric_definition.diagnostic).__name__ not in body
    assert definition["diagnostic_full_slug"] == metric_definition.diagnostic_full_slug


def test_json_is_smaller_than_pickle(metric_definition):
    import pickle  # noqa: PLC0415

    _, _, body = dumps([metric_definition, "INFO"], serializer=SERIALIZER)

    assert len(body.encode()) < len(pickle.dumps([metric_definition, "INFO"]))
