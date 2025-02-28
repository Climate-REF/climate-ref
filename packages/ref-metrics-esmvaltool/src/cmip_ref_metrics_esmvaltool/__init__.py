"""
Rapid evaluating CMIP data with ESMValTool.
"""

import cmip_ref_metrics_esmvaltool.metrics
from cmip_ref_core.providers import CondaMetricsProvider
from cmip_ref_metrics_esmvaltool._version import __version__

# Initialise the metrics manager and register the metrics
provider = CondaMetricsProvider("ESMValTool", __version__)
for _metric_cls_name in cmip_ref_metrics_esmvaltool.metrics.__all__:
    _metric_cls = getattr(cmip_ref_metrics_esmvaltool.metrics, _metric_cls_name)
    provider.register(_metric_cls())
