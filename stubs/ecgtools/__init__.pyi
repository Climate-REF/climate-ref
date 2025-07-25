from typing import Any

import pandas as pd

from climate_ref.datasets.base import DatasetParsingFunction

class Builder:
    df = pd.DataFrame()

    def __init__(
        self,
        paths: list[str],
        depth: int,
        include_patterns: list[str],
        joblib_parallel_kwargs: dict[str, Any],
    ) -> None: ...
    def build(self, *, parsing_func: DatasetParsingFunction) -> Builder: ...
