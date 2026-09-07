"""Synthetic generators and calibrated detector experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging

import numpy as np
import pandas as pd

from .baselines import ControlChart, first_crossing, upcrossing_alarms
from .config import ExperimentConfig
from .models import RegimePrior, SYNTHETIC_PRIOR
from .particle import make_detector
from .progress import map_jobs
from .storage import atomic_write_csv, atomic_write_json, atomic_write_npz

LOGGER = logging.getLogger(__name__)

CHARTS = {
    "CUSUM": ControlChart("cusum-mean"),
    "CUSUM-var": ControlChart("cusum-variance"),
    "CUSUM-mv": ControlChart("cusum-combined"),
    "EWMA": ControlChart("ewma"),
}
BAYES_SYNTHETIC = ("BOCPD", "BOCPD-GARCH")
SYNTHETIC_ALGORITHMS = (*CHARTS, *BAYES_SYNTHETIC)

MONITORING_EXAMPLE_RULE = (
    "first stream in fixed seed order on which BOCPD-GARCH has strictly "
    "higher CCD and DNF than every comparator; otherwise the first stream"
)

PRE_CHANGE = {
    "mu": 0.0,
    "omega": 0.05,
    "alpha": 0.08,
    "beta": 0.87,
    "hbar": 1.0,
}


def simulate_garch_stream(
    rng: np.random.Generator,
    length: int,
    *,
    regimes: list[tuple[int, dict[str, float]]] | None = None,
    hazard: float = 1.0 / 250.0,
    prior: RegimePrior = SYNTHETIC_PRIOR,
    minimum_duration: int | None = None,
) -> tuple[np.ndarray, list[int], np.ndarray]:
    """Simulate regime-switching GARCH with complete variance reset.

    Args:
        rng: NumPy random generator.
        length: Number of observations.
        regimes: Optional fixed ``(start, parameter_mapping)`` regimes. When
            omitted, regimes arrive according to ``hazard``.
        hazard: Changepoint probability per observation for random regimes.
        prior: Parameter prior for randomly generated regimes.
        minimum_duration: Optional minimum age before a random regime can end.

    Returns:
        Simulated observations, changepoint positions, and conditional variances.
    """
    if length < 1:
        raise ValueError("length must be positive")
    values = np.empty(length)
    variances = np.empty(length)
    changepoints: list[int] = []
    regime_index = 1
    starts = [start for start, _ in regimes] if regimes else []

    def draw() -> dict[str, float]:
        """Draw and scalarise one set of GARCH regime parameters."""
        sampled = prior.sample(rng, (), True)
        return {key: float(value) for key, value in sampled.items()}

    parameters = regimes[0][1].copy() if regimes else draw()
    variance = parameters["hbar"]
    previous_residual = 0.0
    first_in_regime = True
    age = 0
    for index in range(length):
        if not first_in_regime:
            variance = (
                parameters["omega"]
                + parameters["alpha"] * previous_residual**2
                + parameters["beta"] * variance
            )
        values[index] = parameters["mu"] + np.sqrt(variance) * rng.standard_normal()
        previous_residual = values[index] - parameters["mu"]
        variances[index] = variance
        first_in_regime = False
        age += 1
        change = False
        replacement = None
        if regimes and regime_index < len(regimes) and index + 1 == starts[regime_index]:
            replacement = regimes[regime_index][1].copy()
            regime_index += 1
            change = True
        elif not regimes and rng.random() < hazard and (
            minimum_duration is None or age >= minimum_duration
        ):
            replacement = draw()
            change = True
        if change and index + 1 < length:
            parameters = replacement
            variance = parameters["hbar"]
            previous_residual = 0.0
            first_in_regime = True
            age = 0
            changepoints.append(index + 1)
    return values, changepoints, variances


def _detector_settings(config: ExperimentConfig, seed: int) -> dict[str, object]:
    """Extract particle-detector settings from an experiment profile."""
    return {
        "particles": config.particles,
        "hazard": config.hazard,
        "max_run_lengths": config.max_run_lengths,
        "young_window": config.young_window,
        "seed": seed,
    }


def _all_trajectories(values: np.ndarray, config: ExperimentConfig, seed: int) -> dict[str, np.ndarray]:
    """Compute every reported synthetic detector statistic on one stream."""
    trajectories = {
        name: chart.statistic_trajectory(values) for name, chart in CHARTS.items()
    }
    for offset, name in enumerate(BAYES_SYNTHETIC):
        key = name.lower()
        detector = make_detector(
            key,
            SYNTHETIC_PRIOR,
            **_detector_settings(config, seed + 1000 * (offset + 1)),
        )
        trajectories[name] = detector.run(values)["young_prob"].to_numpy()
    return trajectories


def _null_job(arguments: tuple[int, ExperimentConfig]) -> dict[str, np.ndarray]:
    """Simulate and analyse one fixed-regime null stream."""
    seed, config = arguments
    values, _, _ = simulate_garch_stream(
        np.random.default_rng(seed),
        config.null_horizon,
        regimes=[(0, PRE_CHANGE)],
    )
    return _all_trajectories(values, config, seed + 100_000)


def calibrate_threshold(
    trajectories: list[np.ndarray], config: ExperimentConfig
) -> tuple[float, float, float, float]:
    """Select a threshold nearest the target mean null run length.

    Args:
        trajectories: Independent null statistic trajectories.
        config: Profile providing burn-in, target ARL0, and grid resolution.

    Returns:
        Threshold, calibration mean run length, standard deviation, and
        right-censored fraction.
    """
    matrix = np.asarray(trajectories)
    candidates = np.unique(
        np.quantile(
            matrix[:, config.burn_in :].ravel(),
            np.linspace(0.50, 0.99999, 1000 if config.full_scale else 100),
        )
    )
    best: tuple[float, float, float, float] | None = None
    for threshold in candidates:
        run_lengths = []
        censored = 0
        for trajectory in matrix:
            crossing = first_crossing(
                trajectory, float(threshold), start=config.burn_in
            )
            if crossing is None:
                run_lengths.append(len(trajectory) - config.burn_in)
                censored += 1
            else:
                run_lengths.append(crossing - config.burn_in + 1)
        mean = float(np.mean(run_lengths))
        row = (
            float(threshold),
            mean,
            float(np.std(run_lengths, ddof=1)),
            censored / len(matrix),
        )
        if best is None or (abs(mean - config.target_arl0), -threshold) < (
            abs(best[1] - config.target_arl0),
            -best[0],
        ):
            best = row
    return best


def _arl_summary(
    trajectories: list[np.ndarray], threshold: float, config: ExperimentConfig
) -> tuple[float, float, float, float, float]:
    """Evaluate a calibrated threshold on independent null trajectories."""
    run_lengths = []
    censored = 0
    for trajectory in trajectories:
        crossing = first_crossing(trajectory, threshold, start=config.burn_in)
        if crossing is None:
            run_lengths.append(len(trajectory) - config.burn_in)
            censored += 1
        else:
            run_lengths.append(crossing - config.burn_in + 1)
    values = np.asarray(run_lengths, dtype=float)
    rng = np.random.default_rng(config.seed + 888_001)
    boot = np.mean(
        rng.choice(values, size=(2000, len(values)), replace=True), axis=1
    )
    return (
        float(values.mean()),
        float(values.std(ddof=1)),
        censored / len(values),
        float(np.quantile(boot, 0.025)),
        float(np.quantile(boot, 0.975)),
    )


def _scenario_parameters(name: str) -> list[tuple[int, dict[str, float]]]:
    """Return pre- and post-change parameters for scenario S1--S4."""
    post = dict(PRE_CHANGE)
    if name in {"S1", "S4"}:
        post["mu"] = 0.75
    if name in {"S2", "S4"}:
        post["hbar"] = 3.0
    elif name == "S3":
        post["hbar"] = 1.0 / 3.0
    post["omega"] = post["hbar"] * (1.0 - post["alpha"] - post["beta"])
    return [(0, dict(PRE_CHANGE)), (None, post)]


def _single_change_job(
    arguments: tuple[str, int, ExperimentConfig, dict[str, float]]
) -> tuple[str, dict[str, int | None]]:
    """Simulate one single-change replication and obtain detector delays."""
    scenario, seed, config, thresholds = arguments
    regimes = _scenario_parameters(scenario)
    regimes[1] = (config.single_change_time, regimes[1][1])
    values, _, _ = simulate_garch_stream(
        np.random.default_rng(seed), config.single_change_horizon, regimes=regimes
    )
    alarms: dict[str, np.ndarray] = {}
    for name, chart in CHARTS.items():
        alarms[name] = chart.run(
            values, thresholds[name], reestimate_after_alarm=False
        ).alarms
    trajectories = _all_trajectories(values, config, seed + 200_000)
    for name in BAYES_SYNTHETIC:
        alarms[name] = upcrossing_alarms(
            trajectories[name], thresholds[name], start=config.burn_in
        )
    delays: dict[str, int | None] = {}
    for name, positions in alarms.items():
        post_change = positions[positions >= config.single_change_time]
        delays[name] = (
            None if len(post_change) == 0 else int(post_change[0] - config.single_change_time)
        )
    return scenario, delays


def summarise_detection_delays(
    delays: list[int | None], detection_window: int
) -> dict[str, float | int]:
    """Separate short-window misses from end-of-follow-up censoring.

    Args:
        delays: First post-change delays, with ``None`` for no alarm by horizon.
        detection_window: Delay cutoff used for the short-window miss rate.

    Returns:
        Conditional delay statistics and two distinct miss rates.
    """
    observed = np.asarray([delay for delay in delays if delay is not None], dtype=float)
    return {
        "mean_delay_detected": float(observed.mean()) if len(observed) else np.nan,
        "sd_delay_detected": (
            float(observed.std(ddof=1)) if len(observed) > 1 else np.nan
        ),
        "median_delay_detected": (
            float(np.median(observed)) if len(observed) else np.nan
        ),
        "miss_rate_50": float(
            np.mean(
                [delay is None or delay >= detection_window for delay in delays]
            )
        ),
        "miss_rate_horizon": float(np.mean([delay is None for delay in delays])),
        "trials": len(delays),
    }


def match_alarms(
    positions: np.ndarray,
    changepoints: list[int] | np.ndarray,
    detection_window: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match each change to its earliest unused alarm in the detection window.

    Args:
        positions: Sorted alarm positions.
        changepoints: Sorted true changepoint positions.
        detection_window: Length of the half-open post-change matching window.

    Returns:
        Matched alarms, unmatched alarms, and detection delays.
    """
    if detection_window < 1:
        raise ValueError("detection_window must be positive")
    alarms = np.asarray(positions, dtype=int)
    changes = np.asarray(changepoints, dtype=int)
    if alarms.ndim != 1 or changes.ndim != 1:
        raise ValueError("alarms and changepoints must be one-dimensional")
    if np.any(np.diff(alarms) < 0) or np.any(np.diff(changes) < 0):
        raise ValueError("alarms and changepoints must be sorted")

    matched_alarm_indices: set[int] = set()
    delays: list[int] = []
    for changepoint in changes:
        candidates = [
            alarm_index
            for alarm_index, position in enumerate(alarms)
            if alarm_index not in matched_alarm_indices
            and changepoint <= position < changepoint + detection_window
        ]
        if candidates:
            alarm_index = candidates[0]
            matched_alarm_indices.add(alarm_index)
            delays.append(int(alarms[alarm_index] - changepoint))

    matched_mask = np.zeros(len(alarms), dtype=bool)
    if matched_alarm_indices:
        matched_mask[np.fromiter(sorted(matched_alarm_indices), dtype=int)] = True
    return (
        alarms[matched_mask],
        alarms[~matched_mask],
        np.asarray(delays, dtype=int),
    )


def _monitoring_ratios(counts: dict[str, object]) -> tuple[float, float]:
    """Calculate correct-change detection and detection-not-false ratios."""
    correct = int(counts["correct"])
    false = int(counts["false"])
    missed = int(counts["missed"])
    ccd = correct / (correct + missed) if correct + missed else -np.inf
    dnf = correct / (correct + false) if correct + false else -np.inf
    return ccd, dnf


def select_monitoring_example(
    results: list[dict[str, object]],
) -> tuple[dict[str, object], bool]:
    """Select a favourable but non-extreme BOCPD-GARCH illustration.

    Results are inspected in their fixed seed order. The first stream on
    which BOCPD-GARCH has strictly greater CCD and DNF than every comparator
    is selected. If no stream satisfies both conditions, the first stream is
    retained. This rule affects only the illustration, never aggregate
    estimates or confidence intervals.

    Args:
        results: Stream-level counts and illustration arrays.

    Returns:
        Selected result and whether it satisfied the stated comparison rule.
    """
    if not results:
        raise ValueError("at least one monitoring result is required")
    ordered = sorted(results, key=lambda result: int(result["stream_index"]))
    for result in ordered:
        counts = result["counts"]
        if not isinstance(counts, dict) or "BOCPD-GARCH" not in counts:
            raise ValueError("monitoring result has an invalid counts mapping")
        target_ccd, target_dnf = _monitoring_ratios(counts["BOCPD-GARCH"])
        comparators = [
            _monitoring_ratios(values)
            for name, values in counts.items()
            if name != "BOCPD-GARCH"
        ]
        if not comparators:
            raise ValueError("monitoring result contains no comparator")
        if target_ccd > max(row[0] for row in comparators) and target_dnf > max(
            row[1] for row in comparators
        ):
            return result, True
    return ordered[0], False


def save_monitoring_example(
    selected: dict[str, object],
    criterion_met: bool,
    config: ExperimentConfig,
    build_dir: Path,
    output_dir: Path,
) -> None:
    """Persist the selected stream and the provenance of its selection.

    Args:
        selected: Stream-level result selected for illustration.
        criterion_met: Whether the preferred selection criterion was satisfied.
        config: BOCPD experiment profile.
        build_dir: Intermediate-output directory.
        output_dir: Final numerical-output directory.
    """
    example = dict(selected["example"])
    example.update(
        {
            "stream_index": np.asarray(int(selected["stream_index"])),
            "stream_seed": np.asarray(int(selected["stream_seed"])),
            "monitoring_streams": np.asarray(config.monitoring_streams),
            "detection_window": np.asarray(config.detection_window),
            "selection_criterion_met": np.asarray(criterion_met),
            "selection_rule": np.asarray(MONITORING_EXAMPLE_RULE),
        }
    )
    atomic_write_npz(build_dir / "continuous_example.npz", example)
    atomic_write_npz(output_dir / "continuous_example.npz", example)


def _continuous_job(
    arguments: tuple[int, int, ExperimentConfig, dict[str, float]]
) -> dict[str, object]:
    """Simulate and score one continuous-monitoring stream."""
    stream_index, seed, config, thresholds = arguments
    values, changepoints, variances = simulate_garch_stream(
        np.random.default_rng(seed),
        config.monitoring_horizon,
        hazard=config.hazard,
        minimum_duration=config.minimum_regime_duration,
    )
    alarms: dict[str, np.ndarray] = {}
    for name, chart in CHARTS.items():
        alarms[name] = chart.run(
            values, thresholds[name], reestimate_after_alarm=True
        ).alarms
    trajectories = _all_trajectories(values, config, seed + 300_000)
    for name in BAYES_SYNTHETIC:
        alarms[name] = upcrossing_alarms(
            trajectories[name], thresholds[name], start=config.burn_in
        )
    counts: dict[str, dict[str, object]] = {}
    for name, positions in alarms.items():
        matched, unmatched, delays = match_alarms(
            positions, changepoints, config.detection_window
        )
        counts[name] = {
            "correct": len(matched),
            "false": len(unmatched),
            "missed": len(changepoints) - len(matched),
            "delays": delays.tolist(),
        }
    return {
        "stream_index": stream_index,
        "stream_seed": seed,
        "counts": counts,
        "example": {
            "values": values,
            "variances": variances,
            "changepoints": np.asarray(changepoints, dtype=int),
            **{f"alarms_{name}": positions for name, positions in alarms.items()},
        },
    }


def run_simulation(
    config: ExperimentConfig,
    output_dir: Path,
    build_dir: Path,
    *,
    workers: int = 1,
) -> dict[str, object]:
    """Run all reported synthetic changepoint experiments.

    Args:
        config: BOCPD experiment profile.
        output_dir: Destination for numerical summaries.
        build_dir: Destination for resumable intermediate arrays.
        workers: Maximum number of independent stream jobs.

    Returns:
        Calibrated thresholds and ARL0 rows.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    build_dir.mkdir(parents=True, exist_ok=True)
    calibration_jobs = [
        (config.seed + 10_000 + index, config)
        for index in range(config.null_calibration_streams)
    ]
    validation_jobs = [
        (config.seed + 20_000 + index, config)
        for index in range(config.null_validation_streams)
    ]
    calibration = map_jobs(
        _null_job,
        calibration_jobs,
        workers,
        label="Null calibration streams",
        logger=LOGGER,
    )
    validation = map_jobs(
        _null_job,
        validation_jobs,
        workers,
        label="Independent null evaluation streams",
        logger=LOGGER,
    )
    thresholds: dict[str, float] = {}
    arl_rows = []
    for detector_index, name in enumerate(SYNTHETIC_ALGORITHMS, start=1):
        LOGGER.info(
            "Threshold and ARL0 summaries: %d/%d (%s)",
            detector_index,
            len(SYNTHETIC_ALGORITHMS),
            name,
        )
        threshold, cal_mean, cal_sd, cal_censored = calibrate_threshold(
            [row[name] for row in calibration], config
        )
        thresholds[name] = threshold
        val_mean, val_sd, val_censored, ci_low, ci_high = _arl_summary(
            [row[name] for row in validation], threshold, config
        )
        arl_rows.append(
            {
                "detector": name,
                "threshold": threshold,
                "calibration_arl0": cal_mean,
                "calibration_sdrl0": cal_sd,
                "calibration_censored_fraction": cal_censored,
                "validation_arl0": val_mean,
                "validation_sdrl0": val_sd,
                "validation_censored_fraction": val_censored,
                "validation_ci_low": ci_low,
                "validation_ci_high": ci_high,
            }
        )
    arl_frame = pd.DataFrame(arl_rows)
    atomic_write_csv(arl_frame, output_dir / "simulation_arl0.csv")
    atomic_write_json(output_dir / "synthetic_thresholds.json", thresholds)

    jobs = [
        (scenario, config.seed + 30_000 + 10_000 * offset + trial, config, thresholds)
        for offset, scenario in enumerate(("S1", "S2", "S3", "S4"))
        for trial in range(config.single_change_trials)
    ]
    trial_results = map_jobs(
        _single_change_job,
        jobs,
        workers,
        label="Single-change simulation trials",
        logger=LOGGER,
    )
    single_rows = []
    for scenario in ("S1", "S2", "S3", "S4"):
        scenario_results = [result for label, result in trial_results if label == scenario]
        for detector in SYNTHETIC_ALGORITHMS:
            delays = [row[detector] for row in scenario_results]
            single_rows.append(
                {
                    "scenario": scenario,
                    "detector": detector,
                    **summarise_detection_delays(
                        delays, config.detection_window
                    ),
                }
            )
    atomic_write_csv(
        pd.DataFrame(single_rows), output_dir / "simulation_single_change.csv"
    )

    continuous_jobs = [
        (index, config.seed + 80_000 + index, config, thresholds)
        for index in range(config.monitoring_streams)
    ]
    continuous_results = map_jobs(
        _continuous_job,
        continuous_jobs,
        workers,
        label="Continuous-monitoring streams",
        logger=LOGGER,
    )
    continuous_rows = []
    for detector_index, detector in enumerate(SYNTHETIC_ALGORITHMS):
        LOGGER.info(
            "Continuous-monitoring summaries: %d/%d (%s)",
            detector_index + 1,
            len(SYNTHETIC_ALGORITHMS),
            detector,
        )
        correct = sum(result["counts"][detector]["correct"] for result in continuous_results)
        false = sum(result["counts"][detector]["false"] for result in continuous_results)
        missed = sum(result["counts"][detector]["missed"] for result in continuous_results)
        delays = [
            delay
            for result in continuous_results
            for delay in result["counts"][detector]["delays"]
        ]
        bootstrap_rng = np.random.default_rng(
            config.seed + 900_000 + detector_index
        )
        ccd_bootstrap = []
        dnf_bootstrap = []
        for _ in range(2000):
            sampled = bootstrap_rng.integers(
                0, len(continuous_results), len(continuous_results)
            )
            boot_correct = sum(
                continuous_results[index]["counts"][detector]["correct"]
                for index in sampled
            )
            boot_false = sum(
                continuous_results[index]["counts"][detector]["false"]
                for index in sampled
            )
            boot_missed = sum(
                continuous_results[index]["counts"][detector]["missed"]
                for index in sampled
            )
            if boot_correct + boot_missed:
                ccd_bootstrap.append(
                    boot_correct / (boot_correct + boot_missed)
                )
            if boot_correct + boot_false:
                dnf_bootstrap.append(boot_correct / (boot_correct + boot_false))
        continuous_rows.append(
            {
                "detector": detector,
                "ccd": correct / (correct + missed) if correct + missed else np.nan,
                "ccd_ci_low": float(np.quantile(ccd_bootstrap, 0.025))
                if ccd_bootstrap
                else np.nan,
                "ccd_ci_high": float(np.quantile(ccd_bootstrap, 0.975))
                if ccd_bootstrap
                else np.nan,
                "dnf": correct / (correct + false) if correct + false else np.nan,
                "dnf_ci_low": float(np.quantile(dnf_bootstrap, 0.025))
                if dnf_bootstrap
                else np.nan,
                "dnf_ci_high": float(np.quantile(dnf_bootstrap, 0.975))
                if dnf_bootstrap
                else np.nan,
                "mean_delay": float(np.mean(delays)) if delays else np.nan,
                "median_delay": float(np.median(delays)) if delays else np.nan,
                "correct": correct,
                "false": false,
                "missed": missed,
                "changepoints": correct + missed,
            }
        )
    atomic_write_csv(
        pd.DataFrame(continuous_rows), output_dir / "simulation_continuous.csv"
    )
    selected, criterion_met = select_monitoring_example(continuous_results)
    save_monitoring_example(
        selected, criterion_met, config, build_dir, output_dir
    )
    LOGGER.info(
        "Continuous-monitoring illustration: stream %d/%d (seed=%d; "
        "selection criterion met=%s)",
        int(selected["stream_index"]) + 1,
        config.monitoring_streams,
        int(selected["stream_seed"]),
        criterion_met,
    )
    return {"thresholds": thresholds, "arl0": arl_rows}
