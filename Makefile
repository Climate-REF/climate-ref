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
	cp LICENCE NOTICE packages/climate-ref
	cp LICENCE NOTICE packages/climate-ref-core
	cp LICENCE NOTICE packages/climate-ref-celery
	cp LICENCE NOTICE packages/climate-ref-esmvaltool
	cp LICENCE NOTICE packages/climate-ref-ilamb
	cp LICENCE NOTICE packages/climate-ref-pmp
	uv build --package climate-ref --no-sources
	uv build --package climate-ref-core --no-sources
	uv build --package climate-ref-celery --no-sources
	uv build --package climate-ref-esmvaltool --no-sources
	uv build --package climate-ref-ilamb --no-sources
	uv build --package climate-ref-pmp --no-sources

.PHONY: ruff-fixes
ruff-fixes:  ## fix the code using ruff
	uv run ruff check --fix
	uv run ruff format

# Coverage is collected with `coverage run` (rather than pytest-cov) so that
# module-level code in entry-point plugins is tracked from process start.
# Each target appends to .coverage; the `test` target runs a final report.

.PHONY: test-ref
test-ref:  ## run the tests
	uv run --package climate-ref \
		coverage run --append --source=packages/climate-ref/src \
		-m pytest packages/climate-ref \
		-r a -v --doctest-modules --durations=20

.PHONY: test-core
test-core:  ## run the tests
	uv run --package climate-ref-core \
		coverage run --append --source=packages/climate-ref-core/src \
		-m pytest packages/climate-ref-core \
		-r a -v --doctest-modules --durations=20

.PHONY: test-celery
test-celery:  ## run the tests
	uv run --package climate-ref-celery \
		coverage run --append --source=packages/climate-ref-celery/src \
		-m pytest packages/climate-ref-celery \
		-r a -v --doctest-modules --durations=20

.PHONY: test-diagnostic-example
test-diagnostic-example:  ## run the tests
	uv run --package climate-ref-example \
		coverage run --append --source=packages/climate-ref-example/src \
		-m pytest packages/climate-ref-example \
		-r a -v --doctest-modules --durations=20

.PHONY: test-diagnostic-esmvaltool
test-diagnostic-esmvaltool:  ## run the tests
	uv run --package climate-ref-esmvaltool \
		coverage run --append --source=packages/climate-ref-esmvaltool/src \
		-m pytest packages/climate-ref-esmvaltool \
		-r a -v --doctest-modules --durations=20

.PHONY: test-diagnostic-ilamb
test-diagnostic-ilamb:  ## run the tests
	uv run ref datasets fetch-data --registry ilamb-test
	uv run --package climate-ref-ilamb \
		coverage run --append --source=packages/climate-ref-ilamb/src \
		-m pytest packages/climate-ref-ilamb \
		-r a -v --doctest-modules --durations=20

.PHONY: test-diagnostic-pmp
test-diagnostic-pmp:  ## run the tests
	uv run --package climate-ref-pmp \
		coverage run --append --source=packages/climate-ref-pmp/src \
		-m pytest packages/climate-ref-pmp \
		-r a -v --doctest-modules --durations=20

.PHONY: test-integration
test-integration:  ## run the integration tests
	uv run \
		pytest tests \
		-r a -v --durations=20

.PHONY: test-integration-slow
test-integration-slow:  ## run the integration tests, including the slow tests which may take a while
	uv run \
		pytest tests --slow \
		-r a -v --durations=20

.PHONY: test-diagnostics
test-diagnostics: test-diagnostic-example test-diagnostic-esmvaltool test-diagnostic-ilamb test-diagnostic-pmp

.PHONY: test-executors
test-executors: test-celery

.PHONY: test
test: clean test-core test-ref test-executors test-diagnostics test-integration ## run the tests
	uv run coverage report

.PHONY: test-quick
test-quick: clean  ## run all the tests at once
	# This is a quicker way of running all the tests
	# It doesn't execute each test using the target package as above
	uv run \
		pytest tests packages \
		-r a -v  --cov-report=term -n auto

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
docs-serve: ## serve the docs locally to http://localhost:8001
	uv run mkdocs serve -a localhost:8001

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
	uv run ref datasets fetch-data --registry ilamb-test

.PHONY: fetch-ref-data
fetch-ref-data:  ## Download reference data needed by providers and (temporarily) not in obs4mips
	uv run ref datasets fetch-data --registry esmvaltool
	uv run ref datasets fetch-data --registry ilamb
	uv run ref datasets fetch-data --registry iomb

.PHONY: update-sample-data-registry
update-sample-data-registry:  ## Update the sample data registry
	curl --output packages/climate-ref/src/climate_ref/dataset_registry/sample_data.txt https://raw.githubusercontent.com/Climate-REF/ref-sample-data/refs/heads/main/registry.txt
