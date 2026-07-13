"""
Rapid evaluating CMIP data
"""

import importlib.metadata

__version__ = importlib.metadata.version("climate-ref")

# Version of sample data used for testing - defined here to avoid importing
# the heavy climate_ref.testing module (which pulls in pandas, xarray, etc.)
SAMPLE_DATA_VERSION = "v0.7.7"

from climate_ref_core.dataset_registry import (  # noqa: E402
    DATASET_URL,
    RegistryUseCase,
    dataset_registry_manager,
)
from climate_ref_core.source_types import SourceDatasetType  # noqa: E402

# Register the obs4REF data registry
dataset_registry_manager.register(
    "obs4ref",
    base_url=DATASET_URL,
    package="climate_ref.dataset_registry",
    resource="obs4ref_reference.txt",
    source_type=SourceDatasetType.obs4REF,
    use_case=RegistryUseCase.reference,
)
# Register the quickstart data registry -- a tiny curated subset of obs4REF used by the
# five-minute quickstart so users can fetch a single reference dataset rather than the
# full (multi-gigabyte) obs4REF collection.
dataset_registry_manager.register(
    "quickstart",
    base_url=DATASET_URL,
    package="climate_ref.dataset_registry",
    resource="quickstart.txt",
    source_type=SourceDatasetType.obs4REF,
    use_case=RegistryUseCase.reference,
)
# Register the sample data registry, used for testing.
#
# This is genuinely multi-type: its manifest mixes CMIP6, obs4MIPs and obs4REF content, so it
# cannot satisfy "exactly one source type" and is left unannotated (source_type=None). It is
# ingested by explicit `--source-type` in tests, never by registry-driven ingest.
dataset_registry_manager.register(
    "sample-data",
    base_url="https://raw.githubusercontent.com/Climate-REF/ref-sample-data/refs/tags/{version}/data/",
    package="climate_ref.dataset_registry",
    resource="sample_data.txt",
    version=SAMPLE_DATA_VERSION,
    use_case=RegistryUseCase.support,
)


__all__ = ["__version__"]
