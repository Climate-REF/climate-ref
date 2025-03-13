#!/usr/bin/env bash
# Test script for ingesting and solving the AR7 FT set of providers
# Uses the sample data
# WARNING this deletes your existing database

export REF_CONFIGURATION=$PWD/.ref

rm -r $REF_CONFIGURATION/db/*

uv run ref datasets ingest --source-type obs4mips $PWD/tests/test-data/sample-data/obs4MIPs
uv run ref datasets ingest --source-type cmip6 $PWD/tests/test-data/sample-data/cmip6

uv run ref --verbose solve --timeout 3600

uv run ref executions list
