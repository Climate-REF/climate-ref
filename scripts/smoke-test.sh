#!/usr/bin/env bash
set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m' # No Color

echo "Starting smoke test for climate-ref docker stack..."

# Function to check if a service is healthy
check_service() {
    local service=$1
    local max_attempts=30
    local attempt=1

    echo "Checking service: $service"
    while [ $attempt -le $max_attempts ]; do
        if docker compose ps $service | grep -q "Up"; then
            echo -e "${GREEN}✓ $service is up${NC}"
            return 0
        fi
        echo "Waiting for $service to be ready... (attempt $attempt/$max_attempts)"
        sleep 2
        attempt=$((attempt + 1))
    done

    echo -e "${RED}✗ $service failed to start${NC}"
    return 1
}

# Initialize the stack
echo "Initializing docker stack..."
bash ./scripts/initialise-docker.sh

# Fetch the test data
docker compose run --rm climate-ref datasets fetch-data --registry sample-data --output-directory /ref/sample-data

# Start the stack
echo "Starting docker stack..."
docker compose up -d

# Check if all services are running
echo "Checking service health..."
services=("redis" "postgres" "flower" "climate-ref" "climate-ref-esmvaltool" "climate-ref-ilamb" "climate-ref-pmp")
for service in "${services[@]}"; do
    check_service $service || exit 1
done

# Sleep to allow services to stabilize
echo "Sleeping to wait for services to stabilize..."
sleep 5

# Log out the started containers
docker compose ps

## Test Celery worker status
#echo "Testing Celery worker status..."
#if docker compose exec climate-ref celery -A climate_ref_celery.app inspect ping | grep -q "3 nodes online"; then
#    echo -e "${GREEN} Celery worker is responsive${NC}"
#else
#    echo -e "${RED} Celery worker is not responsive${NC}"
#    exit 1
#fi

# Log the available data
docker compose exec climate-ref ls -alR /ref/sample-data/CMIP6
docker compose exec climate-ref ls -alR /ref/sample-data/obs4REF

# Ingest sample data
echo "Ingesting sample data..."
if docker compose run --rm climate-ref -v datasets ingest --source-type cmip6 /ref/sample-data/CMIP6; then
    echo -e "${GREEN} CMIP6 data ingestion successful${NC}"
else
    echo -e "${RED} CMIP6 data ingestion failed${NC}"
    exit 1
fi

if docker compose run --rm climate-ref datasets ingest --source-type obs4mips /ref/sample-data/obs4REF; then
    echo -e "${GREEN} Obs4MIPs data ingestion successful${NC}"
else
    echo -e "${RED} Obs4MIPs data ingestion failed${NC}"
    exit 1
fi


# Run a simple solve
if docker compose run --rm climate-ref -v solve --timeout 180 --one-per-provider \
    --diagnostic global-mean-timeseries \
    --diagnostic annual-cycle \
    --diagnostic gpp-wecann; then
    echo -e "${GREEN} Solving completed before timeout${NC}"
else
    echo -e "${RED} Solving failed${NC}"
    exit 1
fi

echo -e "${GREEN} All smoke tests passed!${NC}"
echo "The docker stack is healthy and ready for use."
