# Makefile to help automate key steps

.DEFAULT_GOAL := help
# Will likely fail on Windows, but Makefiles are in general not Windows
# compatible so we're not too worried
TEMP_FILE := $(shell mktemp)

# A helper script to get short descriptions of each target in the Makefile
define PRINT_HELP_PYSCRIPT
import re, sys

for line in sys.stdin:
	match = re.match(r'^([\$$\(\)a-zA-Z_-]+):.*?## (.*)$$', line)
	if match:
		target, help = match.groups()
		print("%-30s %s" % (target, help))
endef
export PRINT_HELP_PYSCRIPT


.PHONY: help
help:  ## print short description of each target
	@python3 -c "$$PRINT_HELP_PYSCRIPT" < $(MAKEFILE_LIST)

.PHONY: pre-commit
pre-commit:  ## run all the linting checks of the codebase
	uv run pre-commit run --all-files


.PHONY: mypy
mypy:  ## run mypy on the codebase
	uv run --package cmip_ref_core mypy packages/ref-core
	uv run --package cmip_ref mypy packages/ref
	uv run --package cmip_ref_metrics_example mypy packages/ref-metrics-example
	uv run --package cmip_ref_metrics_esmvaltool mypy packages/ref-metrics-esmvaltool
	uv run --package cmip_ref_metrics_ilamb mypy packages/ref-metrics-ilamb
	uv run --package cmip_ref_metrics_pmp mypy packages/ref-metrics-pmp

.PHONY: clean
clean:  ## clean up temporary files
	rm -rf site dist build
	rm -rf .coverage

.PHONY: build
build: clean  ## build the packages to be deployed to PyPI
	cp LICENCE NOTICE packages/ref
	cp LICENCE NOTICE packages/ref-core
	cp LICENCE NOTICE packages/ref-metrics-example
	cp LICENCE NOTICE packages/ref-metrics-esmvaltool
	cp LICENCE NOTICE packages/ref-metrics-ilamb
	cp LICENCE NOTICE packages/ref-metrics-pmp
	uv build --package cmip_ref --no-sources
	uv build --package cmip_ref_core --no-sources
	uv build --package cmip_ref_metrics_example --no-sources
	uv build --package cmip_ref_metrics_esmvaltool --no-sources
	uv build --package cmip_ref_metrics_ilamb --no-sources
	uv build --package cmip_ref_metrics_pmp --no-sources

.PHONY: ruff-fixes
ruff-fixes:  ## fix the code using ruff
	uv run ruff check --fix
	uv run ruff format

.PHONY: test-ref
test-ref:  ## run the tests
	uv run --package cmip_ref \
		pytest packages/ref \
		-r a -v --doctest-modules --cov=packages/ref/src --cov-report=term --cov-append

.PHONY: test-core
test-core:  ## run the tests
	uv run --package cmip_ref_core \
		pytest packages/ref-core \
		-r a -v --doctest-modules --cov=packages/ref-core/src --cov-report=term --cov-append

.PHONY: test-metrics-example
test-metrics-example:  ## run the tests
	uv run --package cmip_ref_metrics_example \
		pytest packages/ref-metrics-example \
		-r a -v --doctest-modules --cov=packages/ref-metrics-example/src --cov-report=term --cov-append

.PHONY: test-metrics-esmvaltool
test-metrics-esmvaltool:  ## run the tests
	uv run --package cmip_ref_metrics_esmvaltool \
		pytest packages/ref-metrics-esmvaltool \
		-r a -v --doctest-modules --cov=packages/ref-metrics-esmvaltool/src --cov-report=term --cov-append

.PHONY: test-metrics-ilamb
test-metrics-ilamb:  ## run the tests
	uv run --package cmip_ref_metrics_ilamb \
		pytest packages/ref-metrics-ilamb \
		-r a -v --doctest-modules --cov=packages/ref-metrics-ilamb/src --cov-report=term --cov-append

.PHONY: test-metrics-pmp
test-metrics-pmp:  ## run the tests
	uv run --package cmip_ref_metrics_pmp \
		pytest packages/ref-metrics-pmp \
		-r a -v --doctest-modules --cov=packages/ref-metrics-pmp/src --cov-report=term --cov-append

.PHONY: test-integration
test-integration:  ## run the integration tests
	uv run \
		pytest tests \
		-r a -v

.PHONY: test
test: clean test-core test-ref test-metrics-example test-metrics-esmvaltool test-metrics-ilamb test-metrics-pmp test-integration ## run the tests

# Note on code coverage and testing:
# If you want to debug what is going on with coverage, we have found
# that adding COVERAGE_DEBUG=trace to the front of the below command
# can be very helpful as it shows you if coverage is tracking the coverage
# of all of the expected files or not.

.PHONY: docs
docs:  ## build the docs
	uv run mkdocs build

.PHONY: docs-strict
docs-strict:  ## build the docs strictly (e.g. raise an error on warnings, this most closely mirrors what we do in the CI)
	uv run mkdocs build --strict

.PHONY: docs-serve
docs-serve: ## serve the docs locally
	uv run mkdocs serve

.PHONY: changelog-draft
changelog-draft:  ## compile a draft of the next changelog
	uv run towncrier build --draft

.PHONY: licence-check
licence-check:  ## Check that licences of the dependencies are suitable
	uv export --no-dev > $(TEMP_FILE)
	uv run liccheck -r $(TEMP_FILE) -R licence-check.txt
	rm -f $(TEMP_FILE)

.PHONY: virtual-environment
virtual-environment:  ## update virtual environment, create a new one if it doesn't already exist
	uv sync
	uv run pre-commit install

.PHONY: fetch-test-data
fetch-test-data:  ## Download any data needed by the test suite
	uv run python ./scripts/fetch-sample-data.py

.PHONY: update-test-data-registry
update-test-data-registry:  ## Update the test data registry
	curl --output packages/ref/src/cmip_ref/datasets/sample_data.txt https://raw.githubusercontent.com/CMIP-REF/ref-sample-data/refs/heads/main/registry.txt
