"""Helpers for reading and updating dotted configuration keys."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast, get_args, get_origin

import attrs
from attrs import NOTHING, Attribute

from climate_ref.config import _converter_defaults_relaxed


class ConfigKeyError(KeyError):
    """Raised when a dotted configuration key cannot be resolved."""

    def __init__(self, key: str, segment: str) -> None:
        super().__init__(key)
        self.key = key
        self.segment = segment


def _field_map(obj: object) -> dict[str, Attribute[Any]]:
    if not attrs.has(obj.__class__):
        return {}
    return {field.name: field for field in attrs.fields(obj.__class__)}


def resolve_key(config: object, dotted: str) -> tuple[object, Attribute[Any], Any]:
    """Resolve a dotted key to its parent object, attrs field and current value."""
    if not dotted:
        raise ConfigKeyError(dotted, dotted)

    current = config
    parts = dotted.split(".")
    for index, part in enumerate(parts):
        fields = _field_map(current)
        field = fields.get(part)
        if field is None:
            raise ConfigKeyError(dotted, part)

        value = getattr(current, field.name)
        if index == len(parts) - 1:
            return current, field, value
        current = value

    raise ConfigKeyError(dotted, dotted)  # pragma: no cover


def env_var_for(parent: object, field: Attribute[Any]) -> str | None:
    """Return the environment variable that overrides a field, if any."""
    env_name = field.metadata.get("env")
    if not env_name:
        return None
    prefix = getattr(parent, "_prefix", None)
    if not prefix:
        return None
    return f"{prefix}_{env_name}"


def _is_bool_type(field_type: Any) -> bool:
    if field_type is bool:
        return True
    origin = get_origin(field_type)
    if origin is None:
        return False
    return bool in get_args(field_type)


def _parse_bool(raw: str) -> bool:
    normalised = raw.strip().lower()
    if normalised in {"1", "true", "yes", "y", "on"}:
        return True
    if normalised in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError("expected one of true/false/1/0/yes/no")


def coerce_value(field: Attribute[Any], raw: str) -> Any:
    """Coerce a CLI string to the type required by an attrs field."""
    if _is_bool_type(field.type):
        value: Any = _parse_bool(raw)
    elif field.type is Path:
        value = Path(raw)
    else:
        value = raw

    if field.converter is not None:
        return cast(Callable[[Any], Any], field.converter)(value)

    if field.type is None:
        return value
    return _converter_defaults_relaxed.structure(value, field.type)


def is_structured(value: Any, field: Attribute[Any]) -> bool:
    """Return whether a field is too structured for scalar CLI set/unset."""
    if isinstance(value, dict | list | tuple | set):
        return True
    if attrs.has(value.__class__) and not isinstance(value, Path):
        return True

    origin = get_origin(field.type)
    if origin in {dict, list, tuple, set}:
        return True
    return isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray)


def default_value(parent: object, field: Attribute[Any]) -> Any:
    """Return the attrs default for a field without reading the current value."""
    default = field.default
    if default is NOTHING:
        raise ValueError(f"Configuration key {field.name!r} has no default")

    if hasattr(default, "factory") and hasattr(default, "takes_self"):
        default_factory = cast(Any, default)
        if default_factory.takes_self:
            value = default_factory.factory(parent)
        else:
            value = default_factory.factory()
    else:
        value = default

    if field.converter is not None:
        return cast(Callable[[Any], Any], field.converter)(value)
    return value


def available_keys(config: object, prefix: str = "") -> list[str]:
    """Return all dotted scalar configuration keys."""
    keys: list[str] = []
    for field in _field_map(config).values():
        value = getattr(config, field.name)
        dotted = f"{prefix}.{field.name}" if prefix else field.name
        if is_structured(value, field):
            if attrs.has(value.__class__) and not isinstance(value, Path):
                keys.extend(available_keys(value, dotted))
        else:
            keys.append(dotted)
    return keys
