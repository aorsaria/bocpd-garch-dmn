"""Classical control-chart baselines and alarm rules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

ChartKind = Literal["cusum-mean", "cusum-variance", "cusum-combined", "ewma"]


@dataclass(frozen=True)
class ChartRun:
    """Output of one sequential control-chart run.

    Attributes:
        statistic: Chart statistic at each observation.
        alarms: Integer positions at which the threshold was crossed.
        monitored: Boolean mask identifying observations monitored after burn-in.
    """

    statistic: np.ndarray
    alarms: np.ndarray
    monitored: np.ndarray


@dataclass(frozen=True)
class ControlChart:
    """Sequential CUSUM or EWMA comparator.

    Attributes:
        kind: Statistic to compute.
        burn_in: Number of initial observations used to estimate location and scale.
        reference_value: CUSUM drift allowance in standard-deviation units.
        smoothing: EWMA weight assigned to the current observation.
    """

    kind: ChartKind
    burn_in: int = 50
    reference_value: float = 0.5
    smoothing: float = 0.2

    def __post_init__(self) -> None:
        """Validate the chart configuration after dataclass construction."""
        if self.kind not in {
            "cusum-mean",
            "cusum-variance",
            "cusum-combined",
            "ewma",
        }:
            raise ValueError(f"unknown chart kind: {self.kind}")
        if self.burn_in < 2 or self.reference_value < 0:
            raise ValueError("invalid chart settings")
        if not (0 < self.smoothing <= 1):
            raise ValueError("smoothing must lie in (0, 1]")

    def statistic_trajectory(self, values: np.ndarray) -> np.ndarray:
        """Compute a chart statistic without resetting after threshold crossings.

        Args:
            values: One-dimensional finite observation sequence.

        Returns:
            Statistic values aligned with ``values``; burn-in entries are zero.
        """
        values = _validate_values(values, self.burn_in)
        mean, scale = _burn_estimates(values[: self.burn_in])
        statistic = np.zeros(len(values), dtype=float)
        upper_mean = lower_mean = upper_variance = lower_variance = 0.0
        ewma = mean
        for index in range(self.burn_in, len(values)):
            z_score = (values[index] - mean) / scale
            square_score = (z_score**2 - 1.0) / np.sqrt(2.0)
            upper_mean = max(0.0, upper_mean + z_score - self.reference_value)
            lower_mean = max(0.0, lower_mean - z_score - self.reference_value)
            upper_variance = max(
                0.0, upper_variance + square_score - self.reference_value
            )
            lower_variance = max(
                0.0, lower_variance - square_score - self.reference_value
            )
            if self.kind == "cusum-mean":
                statistic[index] = max(upper_mean, lower_mean)
            elif self.kind == "cusum-variance":
                statistic[index] = max(upper_variance, lower_variance)
            elif self.kind == "cusum-combined":
                statistic[index] = max(
                    upper_mean, lower_mean, upper_variance, lower_variance
                )
            else:
                ewma = (1.0 - self.smoothing) * ewma + self.smoothing * values[index]
                age = index - self.burn_in + 1
                ewma_sd = scale * np.sqrt(
                    self.smoothing
                    / (2.0 - self.smoothing)
                    * (1.0 - (1.0 - self.smoothing) ** (2 * age))
                )
                statistic[index] = abs(ewma - mean) / ewma_sd
        return statistic

    def run(
        self,
        values: np.ndarray,
        threshold: float,
        *,
        reestimate_after_alarm: bool,
    ) -> ChartRun:
        """Monitor an observation sequence and record threshold crossings.

        Args:
            values: One-dimensional finite observation sequence.
            threshold: Strict positive alarm threshold.
            reestimate_after_alarm: Whether to use a new burn-in sample after
                each alarm before monitoring resumes.

        Returns:
            Statistics, alarm positions, and the monitored-observation mask.
        """
        values = _validate_values(values, self.burn_in)
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError("threshold must be positive and finite")
        statistic = np.zeros(len(values), dtype=float)
        monitored = np.zeros(len(values), dtype=bool)
        alarms: list[int] = []
        position = self.burn_in
        mean, scale = _burn_estimates(values[: self.burn_in])
        while position < len(values):
            upper_mean = lower_mean = upper_variance = lower_variance = 0.0
            ewma, age = mean, 0
            alarmed = False
            while position < len(values):
                monitored[position] = True
                z_score = (values[position] - mean) / scale
                square_score = (z_score**2 - 1.0) / np.sqrt(2.0)
                upper_mean = max(0.0, upper_mean + z_score - self.reference_value)
                lower_mean = max(0.0, lower_mean - z_score - self.reference_value)
                upper_variance = max(
                    0.0, upper_variance + square_score - self.reference_value
                )
                lower_variance = max(
                    0.0, lower_variance - square_score - self.reference_value
                )
                if self.kind == "cusum-mean":
                    current = max(upper_mean, lower_mean)
                elif self.kind == "cusum-variance":
                    current = max(upper_variance, lower_variance)
                elif self.kind == "cusum-combined":
                    current = max(
                        upper_mean, lower_mean, upper_variance, lower_variance
                    )
                else:
                    ewma = (1.0 - self.smoothing) * ewma + self.smoothing * values[position]
                    age += 1
                    ewma_sd = scale * np.sqrt(
                        self.smoothing
                        / (2.0 - self.smoothing)
                        * (1.0 - (1.0 - self.smoothing) ** (2 * age))
                    )
                    current = abs(ewma - mean) / ewma_sd
                statistic[position] = current
                if current > threshold:
                    alarms.append(position)
                    position += 1
                    alarmed = True
                    break
                position += 1
            if not alarmed:
                break
            if reestimate_after_alarm:
                if position + self.burn_in > len(values):
                    break
                mean, scale = _burn_estimates(values[position : position + self.burn_in])
                position += self.burn_in
        return ChartRun(statistic, np.asarray(alarms, dtype=int), monitored)


def _validate_values(values: np.ndarray, burn_in: int) -> np.ndarray:
    """Return a finite vector long enough to exceed the specified burn-in."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or len(values) <= burn_in:
        raise ValueError("chart input must be one-dimensional and exceed burn-in")
    if np.any(~np.isfinite(values)):
        raise ValueError("chart input must be finite")
    return values


def _burn_estimates(values: np.ndarray) -> tuple[float, float]:
    """Estimate burn-in mean and sample standard deviation from ``values``."""
    mean = float(np.mean(values))
    scale = float(np.std(values, ddof=1))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("burn-in sample must have positive finite standard deviation")
    return mean, scale


def upcrossing_alarms(
    statistic: np.ndarray, threshold: float, *, start: int = 50
) -> np.ndarray:
    """Return positions where a statistic crosses a threshold from below.

    Args:
        statistic: Sequential statistic values.
        threshold: Strict alarm threshold.
        start: First position eligible for an alarm.

    Returns:
        Integer positions of threshold upcrossings.
    """
    statistic = np.asarray(statistic, dtype=float)
    above = statistic > threshold
    previous = np.concatenate(([False], above[:-1]))
    upcrossing = above & ~previous
    upcrossing[:start] = False
    return np.flatnonzero(upcrossing)


def first_crossing(
    statistic: np.ndarray, threshold: float, *, start: int = 50
) -> int | None:
    """Return the first eligible threshold crossing, or ``None`` if absent.

    Args:
        statistic: Sequential statistic values.
        threshold: Strict alarm threshold.
        start: First position eligible for an alarm.

    Returns:
        Position of the first crossing, or ``None``.
    """
    positions = np.flatnonzero(np.asarray(statistic)[start:] > threshold)
    return None if len(positions) == 0 else int(start + positions[0])
