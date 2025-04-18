"""
Rapid evaluating CMIP data with ESMValTool.
"""

import cmip_ref_metrics_esmvaltool.metrics
from cmip_ref_core.dataset_registry import dataset_registry_manager
from cmip_ref_core.providers import CondaMetricsProvider
from cmip_ref_metrics_esmvaltool._version import __version__
from cmip_ref_metrics_esmvaltool.recipe import _ESMVALTOOL_COMMIT

# Initialise the metrics manager.
provider = CondaMetricsProvider(
    "ESMValTool",
    __version__,
    repo="https://github.com/ESMValGroup/ESMValTool.git",
    tag_or_commit=_ESMVALTOOL_COMMIT,
)

# Register the metrics.
for _metric_cls_name in cmip_ref_metrics_esmvaltool.metrics.__all__:
    _metric_cls = getattr(cmip_ref_metrics_esmvaltool.metrics, _metric_cls_name)
    provider.register(_metric_cls())

# Register OBS, OBS6, and raw data
dataset_registry_manager.register(
    "esmvaltool",
    "https://pub-b093171261094c4ea9adffa01f94ee06.r2.dev/",
    package="cmip_ref_metrics_esmvaltool.dataset_registry",
    resource="data.txt",
)
