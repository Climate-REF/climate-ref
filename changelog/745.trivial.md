Raised the ruff target to py312 and applied the resulting pyupgrade
modernisations (`datetime.UTC`, `Self`/`StrEnum` imports, PEP 695 generics,
and removal of dead `sys.version_info` compatibility blocks).
