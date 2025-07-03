#!/usr/bin/env bash

# run this script under the REF venv!

export REF_DATASET_CACHE_DIR=/gpfs/wolf2/cades/cli185/scratch/mfx/wk_climate_ref/cache
export REF_CONFIGURATION=/gpfs/wolf2/cades/cli185/scratch/mfx/wk_climate_ref/config
export REF_INSTALLATION_DIR=/ccsopen/home/mfx/MyGit/climate-ref/

# set env var to avoid ssl error in the uv venv on some machines
export SSL_CERT_FILE=$(python -m certifi)

# Default behavior: Run all steps if no args given
RUN_CREATE_ENV=false
RUN_PREFETCH=false
RUN_INGEST=false

# Parse command-line options
while getopts "cpiah" opt; do
  case $opt in
    c) RUN_CREATE_ENV=true;  RUN_PREFETCH=false; RUN_INGEST=false ;;
    p) RUN_CREATE_ENV=false; RUN_PREFETCH=true;  RUN_INGEST=false ;;
    i) RUN_CREATE_ENV=false; RUN_PREFETCH=false; RUN_INGEST=true ;;
    a) RUN_CREATE_ENV=true;  RUN_PREFETCH=true;  RUN_INGEST=true ;;
    h) echo "Usage: $0 [-c (create-env only)] [-p (pre-fetch only)] [-i (ingest only)] [-a (do all)]"
       exit 1 ;;
  esac
done

# 1. Create conda environments (if enabled)
if [ "$RUN_CREATE_ENV" = true ]; then
  echo "=== Creating conda environments ==="
  ref providers create-env || { echo "Conda env creation failed"; exit 1; }
fi

# 2. Pre-fetch data (if enabled)
if [ "$RUN_PREFETCH" = true ]; then
  echo "=== Pre-fetching datasets ==="

  # PMP climatology
  ref datasets fetch-data --registry pmp-climatology \
    --output-directory "${REF_DATASET_CACHE_DIR}/datasets/pmp-climatology" || exit 1

  # obs4mips
  ref datasets fetch-data --registry obs4ref \
    --output-directory "${REF_DATASET_CACHE_DIR}/datasets/obs4ref" || exit 1

  # Diagnostic packages
  ref datasets fetch-data --registry ilamb || exit 1
  ref datasets fetch-data --registry iomb || exit 1
  ref datasets fetch-data --registry esmvaltool || exit 1

  # Cartopy data
  python ${REF_INSTALLATION_DIR}/scripts/download-cartopy-data.py || exit 1
fi

# 3. Ingest data (if enabled)
if [ "$RUN_INGEST" = true ]; then
  echo "=== Ingesting datasets ==="
  ref datasets ingest --source-type obs4mips "${REF_DATASET_CACHE_DIR}/datasets/obs4ref" || exit 1
  ref datasets ingest --source-type pmp-climatology "${REF_DATASET_CACHE_DIR}/datasets/pmp-climatology" || exit 1
  # need to run make fetch-test-data under REF directory and check the version under REF cache directory
  ref datasets ingest --source-type cmip6 ${REF_DATASET_CACHE_DIR}/v0.6.3/CMIP6 || exit 1
fi

echo "=== Operation completed ==="
