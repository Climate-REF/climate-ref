#!/usr/bin/env bash

# Run this script under the REF venv on a login node with internet access!
# This prepares everything needed for offline execution on compute nodes.

export REF_DATASET_CACHE_DIR=/gpfs/wolf2/cades/cli185/scratch/mfx/wk_climate_ref/cache
export REF_CONFIGURATION=/gpfs/wolf2/cades/cli185/scratch/mfx/wk_climate_ref/config
export REF_INSTALLATION_DIR=/ccsopen/home/mfx/MyGit/climate-ref/

# Set env var to avoid ssl error in the uv venv on some machines
export SSL_CERT_FILE=$(python -m certifi)

# Default behavior: Run all steps if no args given
RUN_SETUP=false
RUN_FETCH=false
RUN_INGEST=false

# Parse command-line options
while getopts "sfiah" opt; do
  case $opt in
    s) RUN_SETUP=true;  RUN_FETCH=false; RUN_INGEST=false ;;
    f) RUN_SETUP=false; RUN_FETCH=true;  RUN_INGEST=false ;;
    i) RUN_SETUP=false; RUN_FETCH=false; RUN_INGEST=true ;;
    a) RUN_SETUP=true;  RUN_FETCH=true;  RUN_INGEST=true ;;
    h) echo "Usage: $0 [-s (provider setup only)] [-f (fetch datasets only)] [-i (ingest only)] [-a (do all)]"
       exit 1 ;;
  esac
done

# 1. Provider setup: conda environments + provider reference data (if enabled)
if [ "$RUN_SETUP" = true ]; then
  echo "=== Setting up providers (conda envs + reference data) ==="
  ref providers setup || { echo "Provider setup failed"; exit 1; }

  # Cartopy data (not handled by providers)
  python ${REF_INSTALLATION_DIR}/scripts/download-cartopy-data.py || exit 1
fi

# 2. Fetch datasets that need ingestion (if enabled)
if [ "$RUN_FETCH" = true ]; then
  echo "=== Fetching datasets for ingestion ==="

  # obs4REF - observational datasets not yet in obs4MIPs
  ref datasets fetch-data --registry obs4ref \
    --output-directory "${REF_DATASET_CACHE_DIR}/datasets/obs4ref" || exit 1

  # PMP climatology - pre-computed climatology for PMP diagnostics
  ref datasets fetch-data --registry pmp-climatology \
    --output-directory "${REF_DATASET_CACHE_DIR}/datasets/pmp-climatology" || exit 1
fi

# 3. Ingest datasets into the catalog (if enabled)
if [ "$RUN_INGEST" = true ]; then
  echo "=== Ingesting datasets ==="
  ref datasets ingest --source-type obs4mips "${REF_DATASET_CACHE_DIR}/datasets/obs4ref" || exit 1
  ref datasets ingest --source-type pmp-climatology "${REF_DATASET_CACHE_DIR}/datasets/pmp-climatology" || exit 1
  # Need to run make fetch-test-data under REF directory and check the version under REF cache directory
  ref datasets ingest --source-type cmip6 ${REF_DATASET_CACHE_DIR}/v0.6.3/CMIP6 || exit 1
fi

echo "=== Operation completed ==="
