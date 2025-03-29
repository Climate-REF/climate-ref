import logging
from pathlib import Path
from typing import IO, Any

def get_logger() -> logging.Logger: ...

class Pooch:
    registry: dict[str, dict[str, str]]

    def fetch(
        self, fname: str, processor: Any = None, downloader: Any = None, progressbar: bool = False
    ) -> str: ...
    def load_registry(self, file: str | Path | IO[bytes]) -> None: ...

def create(
    path: Path | str,
    base_url: str,
    version: str | None = None,
    version_dev: str | None = "master",
    env: str | None = None,
    registry: dict[str, str] | None = None,
    urls: dict[str, str] | None = None,
    retry_if_failed: int = 0,
    allow_updates: bool | str = True,
) -> Pooch: ...
def os_cache(project: str) -> Path: ...
