"""
Generate virtual config documentation

This splits the configuration into sections, based on the class structure of the Config class.
Each field is documented with its name, description, defaults, type, and environment variable
that can be used to override it.

This script can also be run directly to actually write out those files,
as a preview.
"""

from __future__ import annotations

import ast
import importlib.resources
import inspect
import os
import textwrap
from collections.abc import Sequence
from itertools import pairwise
from typing import Any, get_origin

import attrs
import mkdocs_gen_files

from climate_ref.config import Config, env_prefix

# Load a set of default values for the config
# Replace the REF_CONFIGURATION location with a dummy value for the docs
old_ref_config = os.environ.get("REF_CONFIGURATION")
# This needs to be an absolute path, so we can use it in the docs
os.environ["REF_CONFIGURATION"] = "/__REF_CONFIGURATION__"
defualt_config = Config.default()
os.environ["REF_CONFIGURATION"] = old_ref_config


def _get_default_value(items: Sequence[str]) -> Any:
    """Get a default value from the config class."""
    value = defualt_config
    for item in items:
        value = getattr(value, item)
        if isinstance(value, list):
            value = value[0]

    # We need to replace a path within the installation directory with a dummy values
    ref_install_directory = str(importlib.resources.files("climate_ref_core.pycmec"))
    if ref_install_directory in str(value):
        return str(value).replace(ref_install_directory, "$REF_INSTALL_DIRECTORY")

    if "__REF_CONFIGURATION__" in str(value):
        return str(value).replace("/__REF_CONFIGURATION__", "$REF_CONFIGURATION")

    return value


def _is_attrs_class(cls: type[Any]) -> bool:
    """
    Check if a class is an attrs class or a list of attrs classes
    """
    if get_origin(cls) is list:
        # Handle list types
        return hasattr(cls.__args__[0], "__attrs_attrs__") and hasattr(cls.__args__[0], "__init_subclass__")

    return hasattr(cls, "__attrs_attrs__") and hasattr(cls, "__init_subclass__")


def _get_attr_docs(cls: type[Any]) -> dict[str, str]:
    """
    Get any docstrings placed after attribute assignments in a class body.

    This snippet was sourced from https://davidism.com/attribute-docstrings/ under an MIT license.
    """
    cls_node = ast.parse(textwrap.dedent(inspect.getsource(cls))).body[0]

    if not isinstance(cls_node, ast.ClassDef):
        raise TypeError("Given object was not a class.")

    out = {}

    # Consider each pair of nodes.
    for a, b in pairwise(cls_node.body):
        # Must be an assignment then a constant string.
        if (
            not isinstance(a, ast.Assign | ast.AnnAssign)
            or not isinstance(b, ast.Expr)
            or not isinstance(b.value, ast.Constant)
            or not isinstance(b.value.value, str)
        ):
            continue

        doc = inspect.cleandoc(b.value.value)

        if isinstance(a, ast.Assign):
            # An assignment can have multiple targets (a = b = v).
            targets = a.targets
        else:
            # An annotated assignment only has one target.
            targets = [a.target]

        for target in targets:
            # Must be assigning to a plain name.
            if not isinstance(target, ast.Name):
                continue

            out[target.id] = doc

    return out


def write_field_set(fh, field_parent_names: list[str], target: type[Any]) -> None:
    """
    Write a section of the configuration.

    This corresponds to a class in the config.
    """
    if not field_parent_names:  # Top-level
        fh.write("### Top-level\n\n")
    else:
        # Construct full path like "parent.child"
        title = ".".join(field_parent_names)
        fh.write(f"#### `{title}`\n\n")

    # Write the clean class docstring
    fh.write(f"{inspect.cleandoc(target.__doc__)}\n\n")

    fields = attrs.fields(target)
    field_descriptions = _get_attr_docs(target)

    scalar_fields = [f for f in fields if not _is_attrs_class(f.type) and not f.name.startswith("_")]
    nested_fields = [f for f in fields if _is_attrs_class(f.type) and not f.name.startswith("_")]

    for field in sorted(scalar_fields, key=lambda f: f.name):
        description = field_descriptions.get(field.name, "No description provided.")
        write_field(fh, field_parent_names, field, description)
        fh.write("\n---\n")

    for field in sorted(nested_fields, key=lambda f: f.name):
        field_type = field.type
        if get_origin(field.type) is list:
            field_type = field.type.__args__[0]

        write_field_set(fh, [*field_parent_names, field.name], field_type)


def write_field(fh, field_parent_names: list[str], field, description: str) -> None:
    """
    Write a single field of the configuration.
    """
    header_level = "#####" if field_parent_names else "####"

    # --- Anchor generation (matches Rust logic) ---
    # e.g., parent1_parent2_fieldname
    parents_anchor_part = "_".join(p for p in field_parent_names if p)

    if not parents_anchor_part:
        anchor_id = field.name
    else:
        anchor_id = f"{parents_anchor_part}_{field.name}"

    fh.write(f"{header_level} [`{field.name}`](#{anchor_id}) {{: #{anchor_id} }}\n\n")
    fh.write(description)
    fh.write("\n\n")
    if field.default is not None:
        default_value = _get_default_value([*field_parent_names, field.name])
        fh.write(f"**Default**: {default_value!r}\n\n")

    fh.write(f"**Type**: `{field.type.__name__}`\n\n")

    if field.metadata.get("env"):
        env_variable = f"{env_prefix}_{field.metadata.get('env')}"
        fh.write(f"**Environment Variable**: `{env_variable}`\n\n")


def write_config_page() -> None:
    """
    Write the docs pages for a module/package
    """
    with mkdocs_gen_files.open("configuration.md", "a") as fh:
        write_field_set(fh, [], Config)


write_config_page()
