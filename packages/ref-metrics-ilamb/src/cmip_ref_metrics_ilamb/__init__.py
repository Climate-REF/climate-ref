"""
Rapid evaluating CMIP data
"""

import importlib.metadata
import importlib.resources

import yaml

from cmip_ref_core.dataset_registry import data_registry
from cmip_ref_core.providers import MetricsProvider
from cmip_ref_metrics_ilamb.datasets import ILAMB_DATA_VERSION
from cmip_ref_metrics_ilamb.standard import ILAMBStandard

__version__ = importlib.metadata.version("cmip_ref_metrics_ilamb")

provider = MetricsProvider("ILAMB", __version__)

# Register some datasets
data_registry.register(
    "ilamb-test",
    base_url="https://www.ilamb.org/ILAMB-Data/DATA",
    package="cmip_ref_metric_ilamb.dataset_registry",
    resource="test.txt",
    cache_name="ilamb3",
    version=ILAMB_DATA_VERSION,
)
data_registry.register(
    "ilamb",
    base_url="https://www.ilamb.org/ILAMB-Data/DATA",
    package="cmip_ref_metric_ilamb.dataset_registry",
    resource="ilamb3.txt",
    cache_name="ilamb3",
    version=ILAMB_DATA_VERSION,
)
data_registry.register(
    "ilamb",
    base_url="https://www.ilamb.org/IOMB-Data/DATA",
    package="cmip_ref_metric_ilamb.dataset_registry",
    resource="iomb.txt",
    cache_name="ilamb3",
    version=ILAMB_DATA_VERSION,
)

# Dynamically register ILAMB metrics
for yaml_file in importlib.resources.files("cmip_ref_metrics_ilamb.configure").iterdir():
    with open(str(yaml_file)) as fin:
        metrics = yaml.safe_load(fin)
    registry_filename = metrics.pop("registry")
    for metric, options in metrics.items():
        provider.register(ILAMBStandard(registry_filename, metric, options.pop("sources"), **options))
