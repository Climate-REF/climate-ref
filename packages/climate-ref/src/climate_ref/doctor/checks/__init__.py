"""
The checks that ship with `climate_ref`.

Importing this package registers them, so every module of checks must be imported here.
"""

from climate_ref.doctor.checks import data

__all__ = ["data"]
