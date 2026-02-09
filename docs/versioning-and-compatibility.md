# Versioning and Compatibility

This document describes the versioning strategy for Climate REF packages
and how compatibility is managed between the core framework and diagnostic
provider packages.

## Package Versioning

### Core packages (in the monorepo)

The following packages are versioned together and released from the main
`Climate-REF/climate-ref` repository:

| Package               | Description                              |
| --------------------- | ---------------------------------------- |
| `climate-ref-core`    | Core interfaces and base classes         |
| `climate-ref`         | Main application (CLI, database, solver) |
| `climate-ref-celery`  | Celery distributed executor              |
| `climate-ref-example` | Reference provider implementation        |

These packages follow [Semantic Versioning](https://semver.org/) and share a
single version number managed by `.bumpversion.toml`. A version bump in one
package bumps all of them.

**Semver rules for core packages:**

- **MAJOR**: Breaking changes to `climate-ref-core` public API
- **MINOR**: New features, new public API additions
- **PATCH**: Bug fixes, documentation, internal changes

### Provider packages (separate repositories)

Diagnostic provider packages are versioned independently:

| Package                  | Repository                           |
| ------------------------ | ------------------------------------ |
| `climate-ref-esmvaltool` | `Climate-REF/climate-ref-esmvaltool` |
| `climate-ref-pmp`        | `Climate-REF/climate-ref-pmp`        |
| `climate-ref-ilamb`      | `Climate-REF/climate-ref-ilamb`      |

Each provider follows its own semver cadence. Providers declare their
minimum compatible `climate-ref-core` version via a dependency pin:

```toml
dependencies = [
    "climate-ref-core>=0.9.0,<1.0.0",
]
```

## Compatibility Matrix

| Provider version | Required climate-ref-core | Status  |
| ---------------- | ------------------------- | ------- |
| 0.9.x            | >=0.9.0,<1.0.0            | Current |

This matrix will be updated as versions evolve.

## Compatibility CI

Every provider repository runs a **compatibility CI job** that tests
against multiple versions of `climate-ref-core`:

1. **Minimum supported version** -- the lower bound in the dependency pin
2. **Latest release** -- the newest published version on PyPI
3. **Development (`main`)** -- the latest commit on `climate-ref-core` main branch

This catches breaking changes before they reach users. See
`.github/workflows/provider-compat.yml` for the workflow definition.

## API Stability

The public API surface of `climate-ref-core` is documented in
[`docs/api-surface.md`](api-surface.md). Each symbol is classified into
stability tiers:

| Tier            | Meaning                                                |
| --------------- | ------------------------------------------------------ |
| **Stable**      | Covered by semver; breaking changes require major bump |
| **Provisional** | May change in minor releases with deprecation notice   |
| **Internal**    | Prefixed with `_`; no stability guarantee              |

### Deprecation Policy

When a stable API needs to change:

1. Add a deprecation warning in a **minor** release
2. Document the migration path in the changelog
3. Remove the deprecated API in the next **major** release
4. Maintain at least one minor release with the deprecation warning

## Provider Dependency Guidelines

### For provider authors

- Pin `climate-ref-core` with a compatible release constraint: `>=X.Y.0,<X+1.0.0`
- Only import from modules listed in the [API surface](api-surface.md)
- Avoid importing from `climate_ref` (the app package) in production code;
  use `try/except ImportError` for optional integration (e.g., data ingestion)
- Run the compatibility CI workflow to catch issues early

### For core maintainers

- Run the provider compatibility CI before releasing a new `climate-ref-core` version
- Any change to a **Stable** API requires updating the API surface doc
- Follow the deprecation policy for breaking changes
