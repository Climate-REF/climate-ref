#!/usr/bin/env bash
# Drive a full REF run: ingest CMIP6 + obs4MIPs, then solve every diagnostic.
#
# This mirrors a production-scale local run. It caps the OpenMP/BLAS thread
# pools (the numerical libraries under the diagnostics each spawn core-count
# sized pools, which oversubscribe the CPU once the LocalExecutor fans out --
# see docs/how-to-guides/control-memory-use.md) and raises the open-file limit,
# which large solves routinely exceed.
#
# Configure via environment variables:
#   REF_CONFIGURATION   REF configuration directory (required)
#   REF_CMIP6_DATA      directory of CMIP6 datasets to ingest (required)
#   REF_OBS4MIPS_DATA   directory of obs4MIPs datasets to ingest (required)
#   REF_THREADS         per-process OpenMP/BLAS thread cap (default: 4)
#   REF_NOFILE          open-file ulimit to request (default: 524288)
#   REF_SOLVE_TIMEOUT   overall solve timeout in seconds, 0 = no limit (default: 0)
#
# Example:
#   REF_CONFIGURATION=/data/projects/climate-ref/2026-07-07 \
#   REF_CMIP6_DATA=/mnt/datasets/ESGF/CMIP6 \
#   REF_OBS4MIPS_DATA=/data/projects/climate-ref/cache/obs4ref/obs4REF \
#   scripts/full-run.sh
set -euo pipefail

: "${REF_CONFIGURATION:?set REF_CONFIGURATION to the REF configuration directory}"
: "${REF_CMIP6_DATA:?set REF_CMIP6_DATA to the CMIP6 dataset directory to ingest}"
: "${REF_OBS4MIPS_DATA:?set REF_OBS4MIPS_DATA to the obs4MIPs dataset directory to ingest}"
export REF_CONFIGURATION

threads="${REF_THREADS:-4}"
nofile="${REF_NOFILE:-524288}"
solve_timeout="${REF_SOLVE_TIMEOUT:-0}"

# Cap the OpenMP / BLAS backends so worker processes do not oversubscribe the CPU.
export OMP_NUM_THREADS="$threads"
export OPENBLAS_NUM_THREADS="$threads"
export MKL_NUM_THREADS="$threads"
export NUMEXPR_NUM_THREADS="$threads"
export VECLIB_MAXIMUM_THREADS="$threads"
export BLIS_NUM_THREADS="$threads"

# Raise the open-file limit (best effort -- may be capped by the hard limit).
ulimit -n "$nofile" || echo "warning: could not raise open-file limit to $nofile" >&2

echo "REF_CONFIGURATION=$REF_CONFIGURATION"
echo "threads=$threads ulimit -n=$(ulimit -n) solve_timeout=$solve_timeout"

# 1. Ingest CMIP6 model output.
ref datasets ingest --source-type cmip6 --n-jobs 1 --chunk-size 5000 "$REF_CMIP6_DATA"

# 2. Ingest obs4MIPs reference data.
ref datasets ingest --source-type obs4mips --n-jobs 1 "$REF_OBS4MIPS_DATA"

# 3. Solve every applicable diagnostic across all configured providers.
ref --log-level INFO solve --timeout "$solve_timeout"

echo "Full run complete."
