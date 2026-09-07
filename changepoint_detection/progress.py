"""Consistent elapsed-time and completion logging for experiment batches."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
import logging
import math
import time
from typing import Callable, Iterator, TypeVar, cast

Job = TypeVar("Job")
Result = TypeVar("Result")

LOGGER = logging.getLogger(__name__)
_MISSING = object()


def _duration(seconds: float) -> str:
    """Format a nonnegative duration as ``HH:MM:SS``."""
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


class ProgressReporter:
    """Log approximately twenty progress updates for a finite batch.

    Args:
        label: Human-readable batch name.
        total: Positive number of jobs in the batch.
        logger: Logger receiving progress messages.
    """

    def __init__(self, label: str, total: int, *, logger: logging.Logger = LOGGER):
        """Initialise timing and emit the batch-start message."""
        if total < 1:
            raise ValueError("progress total must be positive")
        self.label = label
        self.total = total
        self.logger = logger
        self.started = time.monotonic()
        self.interval = max(1, math.ceil(total / 20))
        self.logger.info("%s: started (%d items)", self.label, self.total)

    def update(self, completed: int, *, detail: str | None = None) -> None:
        """Record a completed-job count and log it when an interval is reached.

        Args:
            completed: Completed jobs, between one and ``total`` inclusive.
            detail: Optional label for the most recently completed job.
        """
        if completed < 1 or completed > self.total:
            raise ValueError("completed count lies outside the progress total")
        if not (
            completed == 1
            or completed == self.total
            or completed % self.interval == 0
        ):
            return
        elapsed = max(time.monotonic() - self.started, 1e-9)
        rate = completed / elapsed
        remaining = (self.total - completed) / rate
        suffix = f" | latest={detail}" if detail else ""
        self.logger.info(
            "%s: %d/%d (%.1f%%) | elapsed=%s | rate=%.2f items/s | ETA=%s%s",
            self.label,
            completed,
            self.total,
            100.0 * completed / self.total,
            _duration(elapsed),
            rate,
            _duration(remaining),
            suffix,
        )


@contextmanager
def logged_stage(
    label: str, *, logger: logging.Logger = LOGGER
) -> Iterator[None]:
    """Log the start, successful completion, or failure of one pipeline stage.

    Args:
        label: Human-readable stage name.
        logger: Logger receiving lifecycle messages.

    Yields:
        Control to the enclosed stage.
    """
    started = time.monotonic()
    logger.info("%s: started", label)
    try:
        yield
    except Exception:
        logger.error(
            "%s: failed after %s", label, _duration(time.monotonic() - started)
        )
        raise
    logger.info(
        "%s: completed in %s", label, _duration(time.monotonic() - started)
    )


def map_jobs(
    function: Callable[[Job], Result],
    jobs: list[Job],
    workers: int,
    *,
    label: str,
    logger: logging.Logger = LOGGER,
) -> list[Result]:
    """Evaluate jobs concurrently, log progress, and preserve input ordering.

    Args:
        function: Picklable single-job callable.
        jobs: Ordered job values.
        workers: Maximum worker-process count; one selects sequential execution.
        label: Human-readable batch name.
        logger: Logger receiving progress messages.

    Returns:
        Results in the same order as ``jobs``.
    """
    if not jobs:
        logger.info("%s: no items", label)
        return []
    reporter = ProgressReporter(label, len(jobs), logger=logger)
    if workers <= 1:
        results = []
        for completed, job in enumerate(jobs, start=1):
            results.append(function(job))
            reporter.update(completed)
        return results

    results: list[Result | object] = [_MISSING] * len(jobs)
    worker_count = min(workers, len(jobs))
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(function, job): index
            for index, job in enumerate(jobs)
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            results[futures[future]] = future.result()
            reporter.update(completed)
    if any(result is _MISSING for result in results):
        raise RuntimeError(f"{label}: incomplete result collection")
    return cast(list[Result], results)
