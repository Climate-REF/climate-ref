#!/usr/bin/env bash
# Script to initialise the docker compose stack
# This should be run to set up the docker compose stack

set -euo pipefail

# Build the climate-ref container locally
docker compose build

# Run any migrations
docker compose run --rm climate-ref config list

# Initialise the conda containers
docker compose run --rm climate-ref providers setup --skip-data --skip-validate

# Check each provider can launch a diagnostic
docker compose run --rm climate-ref providers check-health
