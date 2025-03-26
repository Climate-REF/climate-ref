import typing
from collections.abc import Callable
from typing import Any, TypeVar, overload

import attr
from attrs import fields
from cattrs import ClassValidationError, ForbiddenExtraKeysError, IterableValidationError
from cattrs.v import format_exception as default_format_exception
from environs.exceptions import EnvError
from loguru import logger

from cmip_ref_core.env import env

T = TypeVar("T")
C = TypeVar("C", bound=type)


def _pop_empty(d: dict[str, Any]) -> None:
    keys = list(d.keys())
    for key in keys:
        value = d[key]
        if isinstance(value, dict):
            _pop_empty(value)
            if not value:
                d.pop(key)


def _format_key_exception(exc: BaseException, _: type | None) -> str | None:
    """Format a n error exception."""
    if isinstance(exc, ForbiddenExtraKeysError):
        return f"extra fields found ({', '.join(exc.extra_fields)})"
    else:
        return None


def _format_exception(exc: BaseException, type: type | None) -> str:  # noqa: A002
    """Format an exception into a string description of the error.

    Parameters
    ----------
    exc
        The exception to format into an error message.
    type
        The type that the value was expected to be.
    """
    # Any custom handling of error goes here before falling back to the default
    if isinstance(exc, EnvError):  # pragma: no cover
        return str(exc)

    return default_format_exception(exc, type)


def transform_error(
    exc: ClassValidationError | IterableValidationError | BaseException,
    path: str = "$",
    format_exception: Callable[[BaseException, type | None], str | None] = _format_exception,
) -> list[str]:
    """Transform an exception into a list of error messages.

    This is based on [cattrs.transform_error][cattrs.transform_error],
     but modified to be able to ignore errors

    To get detailed error messages, the exception should be produced by a converter
    with `detailed_validation` set.

    By default, the error messages are in the form of `{description} @ {path}`.

    While traversing the exception and subexceptions, the path is formed:

    * by appending `.{field_name}` for fields in classes
    * by appending `[{int}]` for indices in iterables, like lists
    * by appending `[{str}]` for keys in mappings, like dictionaries

    Parameters
    ----------
    exc
        The exception to transform into error messages.
    path
        The root path to use.
    format_exception
        A callable to use to transform `Exceptions` into string descriptions of errors.
    """
    errors = []

    def _maybe_append_error(exc: BaseException, _: type | None, path: str) -> str | None:
        error_message = format_exception(exc, None)
        if error_message:
            errors.append(f"{error_message} @ {path}")
        return None

    if isinstance(exc, IterableValidationError):
        iterable_validation_notes, without = exc.group_exceptions()
        for inner_exc, iterable_note in iterable_validation_notes:
            p = f"{path}[{iterable_note.index!r}]"
            if isinstance(inner_exc, ClassValidationError | IterableValidationError):
                errors.extend(transform_error(inner_exc, p, format_exception))
            else:
                _maybe_append_error(inner_exc, iterable_note.type, p)
        for inner_exc in without:
            _maybe_append_error(inner_exc, None, path)
    elif isinstance(exc, ClassValidationError):
        class_validation_notes, without = exc.group_exceptions()
        for inner_exc, class_note in class_validation_notes:
            p = f"{path}.{class_note.name}"
            if isinstance(inner_exc, ClassValidationError | IterableValidationError):
                errors.extend(transform_error(inner_exc, p, format_exception))
            else:
                _maybe_append_error(inner_exc, class_note.type, p)
        for inner_exc in without:
            _maybe_append_error(inner_exc, None, path)
    else:
        _maybe_append_error(exc, None, path)

    return errors


def _environment_override(value: T, env_name: str, convertor: Callable[[Any], T] | None = None) -> T:
    try:
        env_value = env.str(env_name)
    except EnvError:
        return value

    logger.debug(f"Overriding {env_name} with {env_value}")
    if convertor:
        return convertor(env_value)

    return typing.cast(T, env_value)


def _environ_post_init(cls: Any) -> None:
    for f in fields(cls.__class__):
        if f.metadata.get("env"):
            env_name = f"{cls._prefix}_{f.metadata['env']}"

            # This is a bit of a hack to get around the fact that we can't update values on frozen classes
            # https://www.attrs.org/en/stable/how-does-it-work.html#how-frozen
            object.__setattr__(
                cls, f.name, _environment_override(getattr(cls, f.name), env_name, f.converter)
            )


@overload
def config(
    *,
    prefix: str,
    frozen: bool = False,
) -> Callable[[C], C]: ...


@overload
def config(
    maybe_cls: C,
    *,
    prefix: str,
    frozen: bool = False,
) -> C: ...


def config(
    maybe_cls: C | None = None,
    *,
    prefix: str,
    frozen: bool = False,
) -> C | Callable[[C], C]:
    """
    Make a class a configuration class.

    Parameters
    ----------
    prefix:
        The prefix that is used for the env variables.

        This is be prepended to all the fields that use an `env_field`.
    frozen:
        The configuration will be immutable after instantiation, if `True`.
    """

    def wrap(cls: C) -> C:
        setattr(cls, "_prefix", prefix)
        setattr(cls, "__attrs_post_init__", _environ_post_init)

        return attr.s(cls, frozen=frozen, slots=True)

    # maybe_cls's type depends on the usage of the decorator.  It's a class
    # if it's used as `@attrs` but `None` if used as `@attrs()`.
    if maybe_cls is None:
        return wrap

    return wrap(maybe_cls)


def env_field(name: str, **kwargs: Any) -> Any:
    """
    Create a field with an environment variable override.

    This field will use a value from the environment if it is set.

    This field requires the class to be decorated with `config` to work.
    The environment variable name is constructed by prefixing the field name with the class prefix.

    Parameters
    ----------
    name
        Name of the environment variable to use without a prefix
    kwargs
        Additional arguments to pass to the field

    Returns
    -------
    field
        The field with the environment variable override

    """
    return attr.attrib(**kwargs, metadata={"env": name})
