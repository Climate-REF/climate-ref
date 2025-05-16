#!/usr/bin/env bash
# Test script for ingesting and solving the CMIP7 Assessment Fast Track set of providers
# Uses the sample data
# WARNING this deletes your existing database

export REF_CONFIGURATION=$PWD/.ref

rm $REF_CONFIGURATION/db/climate_ref.db

# Ingest datasets
ref datasets ingest --source-type cmip6 tests/test-data/sample-data/CMIP6
ref datasets ingest --source-type obs4mips tests/test-data/sample-data/obs4REF

# Run everything
ref --verbose solve --timeout 360

# Display the executions
ref executions list-groups
