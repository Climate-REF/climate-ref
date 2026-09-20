#
#  OPTIONS ARE SET BY USER IN THIS FILE AS INDICATED BELOW BY:
#
#

# LIST OF MODEL VERSIONS TO BE TESTED
modnames = ["CanCM4"]

# ROOT PATH FOR MODELS CLIMATOLOGIES
test_data_path = (
    "demo_data_tmp/CMIP5_demo_clims/cmip5.historical.%(model).r1i1p1.mon.pr.198101-200512.AC.v20200426.nc"
)

# ROOT PATH FOR OBSERVATIONS
reference_data_path = "NOAA-NCEI/GPCP-2-3/mon/pr/gn/v20210727/pr_mon_GPCP-2-3_PCMDI_gn_197901-201907.nc"

# DIRECTORY WHERE TO PUT RESULTS
results_dir = "demo_output_tmp/monsoon_wang"

# Threshold
threshold = 2.5 / 86400
