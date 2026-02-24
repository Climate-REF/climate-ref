from collections.abc import Sequence
from typing import Any

class datetime:
    year: int
    month: int
    day: int
    hour: int
    minute: int
    second: int
    microsecond: int
    def strftime(self, fmt: str = ...) -> str: ...
    def __str__(self) -> str: ...

def num2date(
    num_dates: float | Sequence[Any],
    units: str,
    calendar: str = "standard",
    only_use_cftime_datetimes: bool = True,
    only_use_python_datetimes: bool = False,
    has_year_zero: bool | None = None,
) -> Any: ...
