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
	uv run mypy packages

.PHONY: clean
clean:  ## clean up temporary files
	rm -rf site dist build
	rm -rf .coverage

.PHONY: clean-sample-data
clean-sample-data:  ## clean up the sample data
	rm -rf tests/test-data/sample-data

.PHONY: build
build: clean  ## build the packages to be deployed to PyPI
	cp LICENCE NOTICE packages/ref
	cp LICENCE NOTICE packages/ref-core
	cp LICENCE NOTICE packages/ref-celery
	cp LICENCE NOTICE packages/ref-metrics-example
	cp LICENCE NOTICE packages/ref-metrics-esmvaltool
	cp LICENCE NOTICE packages/ref-metrics-ilamb
	cp LICENCE NOTICE packages/ref-metrics-pmp
	uv build --package cmip_ref --no-sources
	uv build --package cmip_ref_core --no-sources
	uv build --package cmip_ref_celery --no-sources
	uv build --package cmip_ref_metrics_esmvaltool --no-sources
	uv build --package cmip_ref_metrics_ilamb --no-sources
	uv build --package cmip_ref_metrics_pmp --no-sources
	uv build --package cmip_ref_metrics_example --no-sources

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

.PHONY: test-celery
test-celery:  ## run the tests
	uv run --package cmip_ref_celery \
		pytest packages/ref-celery \
		-r a -v --doctest-modules --cov=packages/ref-celery/src

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
	uv run --package cmip_ref_metrics_ilamb python ./scripts/fetch-ilamb-data.py test.txt
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
		pytest tests -m "not slow" \
		-r a -v

.PHONY: test-integration-slow
test-integration-slow:  ## run the integration tests, including the slow tests which may take a while
	uv run \
		pytest tests\
		-r a -v

.PHONY: test-metrics-packages
test-metrics-packages: test-metrics-example test-metrics-esmvaltool test-metrics-ilamb test-metrics-pmp

.PHONY: test-executors
test-executors: test-celery

.PHONY: test
test: clean test-core test-ref test-executors test-metrics-packages test-integration ## run the tests

.PHONY: test-quick
test-quick: clean  ## run all the tests at once
	# This is a quicker way of running all the tests
	# It doesn't execute each test using the target package as above
	uv run \
		pytest tests packages \
		-r a -v  --cov-report=term

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
	uv run ref datasets fetch-sample-data
	uv run python ./scripts/fetch-ilamb-data.py test.txt

.PHONY: update-sample-data-registry
update-sample-data-registry:  ## Update the sample data registry
	curl --output packages/ref/src/cmip_ref/datasets/sample_data.txt https://raw.githubusercontent.com/Climate-REF/ref-sample-data/refs/heads/main/registry.txt
