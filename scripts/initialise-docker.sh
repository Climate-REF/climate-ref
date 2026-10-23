#!/usr/bin/env bash
# Script to initialise the docker compose stack
# This should be run to set up the docker compose stack

# Build the climate-ref container locally
docker compose build

# Run any migrations
docker compose run --rm climate-ref config list

# Initialise the conda containers
docker compose run --rm climate-ref providers setup --skip-data --skip-validate
