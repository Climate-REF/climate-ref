from pathlib import Path

class Env:
    def __init__(self, expand_vars: bool) -> None: ...
    def read_env(
        self,
        path: str | None = None,
        recurse: bool = True,
        verbose: bool = False,
        override: bool = False,
    ) -> bool: ...
    def path(self, name: str, default: str | None = None) -> Path: ...
    def list(self, name: str, default: list[str] | None) -> list[str]: ...
    def str(self, name: str, default: str | None = None) -> str: ...
