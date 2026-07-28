"""
CMEC python package
"""

from climate_ref_core.data import PackagedResource

BUNDLED_AFT_CV = PackagedResource("climate_ref_core.pycmec", "cv_cmip7_aft.yaml")
"""
The controlled vocabulary for the CMIP7 Assessment Fast Track diagnostics.

This ships inside `climate_ref_core` and is used unless an operator overrides it.
It is declared here rather than alongside `CV` so that reading the configuration
does not pull in the parsing machinery.
"""
