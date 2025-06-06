import datetime
import os

# =================================================
# Background Information
# -------------------------------------------------
mip = "cmip6"
exp = "historical"
frequency = "mo"
realm = "atm"

# =================================================
# Analysis Options
# -------------------------------------------------
variability_mode = "NAM"  # Available domains: NAM, NAO, SAM, PNA, PDO
seasons = [
    "DJF",
    "MAM",
    "JJA",
    "SON",
]  # Available seasons: DJF, MAM, JJA, SON, monthly, yearly

ConvEOF = True  # Calculate conventioanl EOF for model
CBF = True  # Calculate Common Basis Function (CBF) for model

# =================================================
# Miscellaneous
# -------------------------------------------------
update_json = False
debug = False

# =================================================
# Observation
# -------------------------------------------------
reference_data_name = "NOAA-CIRES_20CR"
reference_data_path = os.path.join(
    "/p/user_pub/PCMDIobs/obs4MIPs/NOAA-ESRL-PSD/20CR/mon/psl/gn/latest",
    "psl_mon_20CR_PCMDI_gn_187101-201212.nc",
)

varOBS = "psl"
ObsUnitsAdjust = (True, "divide", 100.0)  # Pa to hPa; or (False, 0, 0)

osyear = 1900
oeyear = 2005
eofn_obs = 1

# =================================================
# Models
# -------------------------------------------------
modpath = os.path.join(
    "/p/css03/cmip5_css02/data/cmip5/output1/CSIRO-BOM/ACCESS1-0/historical/mon/atmos/Amon/r1i1p1/psl/1",
    "psl_Amon_ACCESS1-0_historical_r1i1p1_185001-200512.nc",
)

modnames = ["ACCESS1-0"]

realization = "r1i1p1f1"

varModel = "psl"
ModUnitsAdjust = (True, "divide", 100.0)  # Pa to hPa

msyear = 1900
meyear = 2005
eofn_mod = 1

# =================================================
# Output
# -------------------------------------------------
case_id = f"{datetime.datetime.now():v%Y%m%d}"
pmprdir = "/p/user_pub/pmp/pmp_results/"

results_dir = os.path.join(
    pmprdir,
    "%(output_type)",
    "variability_modes",
    "%(mip)",
    "%(exp)",
    "%(case_id)",
    "%(variability_mode)",
    "%(reference_data_name)",
)

# Output for obs
plot_obs = True  # Create map graphics
nc_out_obs = True  # Write output in NetCDF

# Output for models
nc_out = True
plot = True
