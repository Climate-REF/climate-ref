"""
Progress reporting for long-running loops

A tqdm bar only renders on a TTY,
so [LoggedProgress][climate_ref.progress.LoggedProgress] also logs the same progress periodically.
This keeps long solves and ingests visible in collected logs such as Loki.
"""

import datetime
import threading
import time
import weakref
from collections.abc import Iterator
from typing import Any

from loguru import logger
from tqdm import tqdm

from climate_ref_core.env import env

PROGRESS_LOG_INTERVAL_ENV = "REF_LOG_PROGRESS_INTERVAL"
"""
Environment variable setting how often progress is logged, in seconds

A value of 0 turns the progress log off.
"""

DEFAULT_PROGRESS_LOG_INTERVAL = 300.0


def progress_log_interval() -> float:
    """Get how often progress is logged, in seconds, from the environment"""
    return env.float(PROGRESS_LOG_INTERVAL_ENV, DEFAULT_PROGRESS_LOG_INTERVAL)


def _log_periodically(bar_ref: "weakref.ref[LoggedProgress]", stop: threading.Event, interval: float) -> None:
    # A weak reference so a bar that is never closed can still be collected, which ends this thread
    while not stop.wait(interval):
        bar = bar_ref()
        if bar is None:
            return
        logger.info(bar.progress_message())
        del bar


class LoggedProgress(tqdm):  # type: ignore[type-arg]
    """
    A tqdm progress bar that also logs its progress at INFO every ``log_interval`` seconds

    The bar is hidden when stderr is not a TTY, where its carriage returns corrupt collected logs.
    The log line is written from a background thread,
    so a loop that stops making progress keeps reporting how long it has been stuck.
    """

    def __init__(self, *args: Any, log_interval: float | None = None, **kwargs: Any) -> None:
        kwargs.setdefault("disable", None)
        # A hidden bar stops counting and drops its labels, so they are tracked here
        self._hidden_count = kwargs.get("initial", 0)
        self._label = kwargs.get("desc") or "Progress"
        self._unit = kwargs.get("unit", "it")
        self._started = time.monotonic()
        self._stop = threading.Event()
        super().__init__(*args, **kwargs)
        # tqdm sets ``disable`` on close too, so whether the bar is shown is captured now
        self._hidden = bool(self.disable)

        interval = progress_log_interval() if log_interval is None else log_interval
        if interval > 0:
            threading.Thread(
                target=_log_periodically, args=(weakref.ref(self), self._stop, interval), daemon=True
            ).start()

    def update(self, n: float | None = 1) -> bool | None:
        """Record ``n`` more completed items"""
        if self._hidden:
            self._hidden_count += n or 0
        return super().update(n)

    def __iter__(self) -> Iterator[Any]:
        if not self._hidden:
            yield from super().__iter__()
            return
        try:
            for obj in self.iterable:
                self._hidden_count += 1
                yield obj
        finally:
            self.close()

    def close(self) -> None:
        """Stop logging progress and close the bar"""
        self._stop.set()
        super().close()

    def progress_message(self) -> str:
        """Describe the progress so far"""
        elapsed = time.monotonic() - self._started
        took = datetime.timedelta(seconds=int(elapsed))
        done = int(self._hidden_count if self._hidden else self.n)
        if not self.total:
            return f"{self._label}: {done} {self._unit}s done after {took}"

        fraction = done / self.total
        message = f"{self._label}: {done}/{int(self.total)} {self._unit}s done ({fraction:.0%}) after {took}"
        if 0 < done < self.total:
            remaining = elapsed / done * (self.total - done)
            message += f", about {datetime.timedelta(seconds=int(remaining))} left"
        return message
