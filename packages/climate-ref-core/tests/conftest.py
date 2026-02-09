"""
Core test configuration.

Import core modules to ensure they are loaded during test collection.
"""

import climate_ref_core.constraints
import climate_ref_core.dataset_registry
import climate_ref_core.datasets
import climate_ref_core.diagnostics
import climate_ref_core.env
import climate_ref_core.esgf
import climate_ref_core.exceptions
import climate_ref_core.executor
import climate_ref_core.logging
import climate_ref_core.metric_values
import climate_ref_core.providers
import climate_ref_core.pycmec
import climate_ref_core.source_types
import climate_ref_core.testing  # noqa: F401
