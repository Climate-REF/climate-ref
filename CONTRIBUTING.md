# Contributing to Climate REF

For detailed contribution guidelines, see <https://climate-ref.readthedocs.io/en/latest/development/>.

## Repository Structure

Climate REF is a monorepo managed with [uv workspaces](https://docs.astral.sh/uv/concepts/workspaces/).
All packages live under `packages/`:

| Package                  | Description                                   |
| ------------------------ | --------------------------------------------- |
| `climate-ref-core`       | Core library with base classes and interfaces |
| `climate-ref`            | Main application (CLI, database, solver)      |
| `climate-ref-celery`     | Celery executor for distributed execution     |
| `climate-ref-example`    | Reference implementation for new providers    |
| `climate-ref-esmvaltool` | ESMValTool diagnostic provider                |
| `climate-ref-pmp`        | PCMDI Metrics Package diagnostic provider     |
| `climate-ref-ilamb`      | ILAMB diagnostic provider                     |

## Cloning the Repository

### Full Clone (recommended for most contributors)

```bash
git clone https://github.com/Climate-REF/climate-ref.git
cd climate-ref
```

### Partial Clone (large repos / CI)

If you only need recent history or want to reduce clone size, use a partial clone:

```bash
# Blobless clone: fetches all commits and trees, but downloads file contents on demand
git clone --filter=blob:none https://github.com/Climate-REF/climate-ref.git

# Treeless clone: fetches only commits, downloads trees and blobs on demand (fastest)
git clone --filter=tree:0 https://github.com/Climate-REF/climate-ref.git

# Shallow clone: only fetch the last N commits (not recommended for development)
git clone --depth=1 https://github.com/Climate-REF/climate-ref.git
```

**Blobless clone** (`--filter=blob:none`) is the best trade-off for development:
it has full commit history for `git log` and `git blame`, but only downloads
file contents when you check them out.

### Sparse Checkout (work on a single package)

If you only need to work on one package, combine partial clone with sparse checkout:

```bash
git clone --filter=blob:none --sparse https://github.com/Climate-REF/climate-ref.git
cd climate-ref

# Check out only the package you need plus shared config
git sparse-checkout set packages/climate-ref-core packages/climate-ref-esmvaltool tests
```

## Setting Up the Development Environment

```bash
# Install uv if you haven't already
# See https://docs.astral.sh/uv/getting-started/installation/

# Create the virtual environment and install all packages
make virtual-environment

# Or manually:
uv sync
```

## Running Tests

```bash
# Run all tests
make test

# Run tests for a specific package
make test-core          # climate-ref-core
make test-ref           # climate-ref

# Run a single test file
uv run pytest packages/climate-ref-core/tests/unit/test_diagnostics.py -v
```

## Code Quality

```bash
# Run all pre-commit checks (ruff, mypy, etc.)
make pre-commit

# Auto-fix linting issues
make ruff-fixes

# Type checking
make mypy
```
