"""
Registration of the REF wire format with kombu.

Celery looks codecs up by name, so this has to be imported by both the client sending a
task and the worker receiving it. Importing `climate_ref_celery.app` is enough for both.
"""

import json
from typing import Any

from kombu.serialization import register

from climate_ref_core.serialisation import from_wire, to_wire

SERIALIZER = "ref-json"
CONTENT_TYPE = "application/x-ref-json"


def _encode(payload: Any) -> str:
    # allow_nan would write bare NaN and Infinity, which is not valid JSON.
    # to_wire tags them, so this only fires on a type it missed.
    return json.dumps(to_wire(payload), allow_nan=False)


def _decode(payload: str | bytes) -> Any:
    return from_wire(json.loads(payload))


def register_serialisation() -> None:
    """
    Register the REF wire format so Celery can encode tasks and results with it

    Safe to call more than once.
    """
    register(
        SERIALIZER,
        encoder=_encode,
        decoder=_decode,
        content_type=CONTENT_TYPE,
        content_encoding="utf-8",
    )
