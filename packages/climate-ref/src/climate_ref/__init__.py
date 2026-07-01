"""
Rapid evaluating CMIP data
"""

import importlib.metadata

__version__ = importlib.metadata.version("climate-ref")

# Version of sample data used for testing - defined here to avoid importing
# the heavy climate_ref.testing module (which pulls in pandas, xarray, etc.)
SAMPLE_DATA_VERSION = "v0.7.7"

from climate_ref_core.dataset_registry import DATASET_URL, dataset_registry_manager  # noqa: E402

# Register the obs4REF data registry
dataset_registry_manager.register(
    "obs4ref",
    base_url=DATASET_URL,
    package="climate_ref.dataset_registry",
    resource="obs4ref_reference.txt",
)
# Register the quickstart data registry -- a tiny curated subset of obs4REF used by the
# five-minute quickstart so users can fetch a single reference dataset rather than the
# full (multi-gigabyte) obs4REF collection.
dataset_registry_manager.register(
    "quickstart",
    base_url=DATASET_URL,
    package="climate_ref.dataset_registry",
    resource="quickstart.txt",
)
# Register the sample data registry -- used for testing
dataset_registry_manager.register(
    "sample-data",
    base_url="https://raw.githubusercontent.com/Climate-REF/ref-sample-data/refs/tags/{version}/data/",
    package="climate_ref.dataset_registry",
    resource="sample_data.txt",
    version=SAMPLE_DATA_VERSION,
)


__all__ = ["__version__"]
