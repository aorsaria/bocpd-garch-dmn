"""Training-only selection and out-of-sample market event study."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_data import TICKERS, UNIVERSE

from .baselines import ControlChart, upcrossing_alarms
from .config import ExperimentConfig
from .data import CLASS_ORDER, TEST_START, TRAIN_END, TrainingInputs
from .models import RegimePrior
from .particle import make_detector
from .progress import map_jobs
from .simulation import CHARTS
from .storage import atomic_write_csv, atomic_write_json, atomic_write_parquet

LOGGER = logging.getLogger(__name__)

NU_VALUES: tuple[float | None, ...] = (4.0, 6.0, 10.0, None)
EVENTS = (
    ("GFC", pd.Timestamp("2008-09-15"), frozenset(CLASS_ORDER)),
    ("China 2015", pd.Timestamp("2015-08-24"), frozenset({"EQ"})),
    ("Volmageddon", pd.Timestamp("2018-02-05"), frozenset({"EQ"})),
    ("COVID-19", pd.Timestamp("2020-02-24"), frozenset(CLASS_ORDER)),
    ("Ukraine", pd.Timestamp("2022-02-24"), frozenset({"CM", "EQ", "FX"})),
)
REAL_ALGORITHMS = (*CHARTS, "BOCPD", "BOCPD-GARCH", "BOCPD-GARCH-t")


def nu_label(nu: float | None) -> str:
    """Return the table label for an innovation law.

    Args:
        nu: Student-t degrees of freedom, or ``None`` for Gaussian innovations.

    Returns:
        Stable innovation-law label.
    """
    return "Gaussian" if nu is None else f"t({nu:g})"


def _rank_innovation_scores(
    pooled: pd.Series,
) -> tuple[str, str, float, bool]:
    """Select the overall winner and best finite Student-t candidate."""
    ordered_labels = [nu_label(nu) for nu in NU_VALUES]
    winner_label = max(
        ordered_labels,
        key=lambda label: (pooled[label], -ordered_labels.index(label)),
    )
    finite_labels = [nu_label(nu) for nu in NU_VALUES if nu is not None]
    selected_t_label = max(
        finite_labels,
        key=lambda label: (pooled[label], -finite_labels.index(label)),
    )
    selected_nu = {nu_label(nu): nu for nu in NU_VALUES}[selected_t_label]
    if selected_nu is None:
        raise AssertionError("finite Student-t candidate unexpectedly mapped to None")
    return (
        winner_label,
        selected_t_label,
        float(selected_nu),
        winner_label == "Gaussian",
    )


def _detector_settings(config: ExperimentConfig, seed: int) -> dict[str, object]:
    """Extract particle-detector settings from an experiment profile."""
    return {
        "particles": config.particles,
        "hazard": config.hazard,
        "max_run_lengths": config.max_run_lengths,
        "young_window": config.young_window,
        "seed": seed,
    }


def _nu_job(
    arguments: tuple[str, np.ndarray, dict[str, float], float | None, int, ExperimentConfig]
) -> dict[str, object]:
    """Evaluate one contract, innovation candidate, and particle-filter seed."""
    ticker, values, prior_values, nu, seed, config = arguments
    detector_name = "bocpd-garch" if nu is None else "bocpd-garch-t"
    detector = make_detector(
        detector_name,
        RegimePrior.from_dict(prior_values),
        nu=nu,
        **_detector_settings(config, seed),
    )
    output = detector.run(values)
    score = float(output["predictive_log_likelihood"].iloc[config.burn_in :].mean())
    return {"ticker": ticker, "nu": nu_label(nu), "score": score}


def select_nu(
    inputs: TrainingInputs,
    config: ExperimentConfig,
    output_dir: Path,
    *,
    workers: int = 1,
) -> float:
    """Select Student-t degrees of freedom using pre-2007 predictive density.

    Args:
        inputs: Market returns, standardisation scales, and regime priors.
        config: BOCPD experiment profile.
        output_dir: Directory receiving selection tables and metadata.
        workers: Maximum number of independent detector jobs.

    Returns:
        Degrees of freedom of the best finite Student-t candidate. The output
        metadata separately records whether the Gaussian law ranked first.
    """
    tickers = tuple(TICKERS) if config.full_scale else config.representative_tickers
    jobs = []
    for ticker_index, ticker in enumerate(tickers):
        training = inputs.returns[ticker].loc[:TRAIN_END].to_numpy(dtype=float)
        values = (training / inputs.scales[ticker])[-config.selection_observations :]
        asset_class = UNIVERSE[ticker][0]
        for nu_index, nu in enumerate(NU_VALUES):
            for replicate in range(config.selection_seeds):
                seed = (
                    config.seed
                    + 110_000
                    + 10_000 * ticker_index
                    + 100 * replicate
                    + nu_index
                )
                # Common random numbers across nu: remove nu_index from seed.
                seed -= nu_index
                jobs.append(
                    (
                        ticker,
                        values,
                        inputs.priors[asset_class].to_dict(),
                        nu,
                        seed,
                        config,
                    )
                )
    results = pd.DataFrame(
        map_jobs(
            _nu_job,
            jobs,
            workers,
            label="Innovation-selection detector runs",
            logger=LOGGER,
        )
    )
    per_ticker = (
        results.groupby(["ticker", "nu"], as_index=False)["score"].mean()
    )
    per_ticker["asset_class"] = per_ticker["ticker"].map(
        lambda ticker: UNIVERSE[ticker][0]
    )
    class_scores = (
        per_ticker.groupby(["asset_class", "nu"], as_index=False)["score"].mean()
    )
    pooled = class_scores.groupby("nu")["score"].mean()
    winner_label, selected_t_label, selected, gaussian_preferred = (
        _rank_innovation_scores(pooled)
    )
    LOGGER.info(
        "Innovation selection complete: pooled winner=%s; Student-t row=%s",
        winner_label,
        selected_t_label,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_csv(per_ticker, output_dir / "nu_selection_contracts.csv")
    atomic_write_csv(class_scores, output_dir / "nu_selection_classes.csv")
    summary = pd.DataFrame(
        {"nu": pooled.index, "pooled_score": pooled.values}
    )
    gaussian = float(pooled["Gaussian"])
    summary["gain_over_gaussian"] = summary["pooled_score"] - gaussian
    atomic_write_csv(summary, output_dir / "nu_selection.csv")
    atomic_write_json(
        output_dir / "nu_selected.json",
        {
            "selected_label": winner_label,
            "selected_t_label": selected_t_label,
            "selected_nu": selected,
            "gaussian_preferred": gaussian_preferred,
            "training_end": str(TRAIN_END.date()),
            "tickers": list(tickers),
            "selection_observations": config.selection_observations,
            "replicates": config.selection_seeds,
        },
    )
    return float(selected)


def _rate_calibrate(
    count_alarms_and_exposure,
    candidates: np.ndarray,
    target_rate: float,
) -> float:
    """Choose the threshold whose empirical alarm rate is nearest a target."""
    best = None
    for threshold in np.unique(candidates):
        count, monitored_observations = count_alarms_and_exposure(float(threshold))
        count = int(count)
        monitored_observations = int(monitored_observations)
        if monitored_observations <= 0:
            raise ValueError("threshold calibration has no monitored exposure")
        gap = abs(count / monitored_observations - target_rate)
        row = (gap, -float(threshold), float(threshold))
        if best is None or row < best:
            best = row
    return best[2]


def _chart_alarm_count_and_exposure(
    chart: ControlChart, values: np.ndarray, threshold: float
) -> tuple[int, int]:
    """Return reset-chart alarm count and actual monitored exposure."""
    run = chart.run(values, threshold, reestimate_after_alarm=True)
    return len(run.alarms), int(run.monitored.sum())


def event_awareness(
    alarms: np.ndarray,
    event_position: int,
    *,
    statistic_on_event: float | None = None,
    threshold: float | None = None,
    lookback: int = 63,
) -> bool:
    """Evaluate prior-quarter awareness at an event date.

    For every detector, the alarm leg includes positions in
    ``[event_position - lookback, event_position)`` and therefore excludes the
    event date. For Bayesian detectors, callers additionally provide the
    event-date posterior statistic and threshold; equality counts as
    awareness. Chart rows use the alarm-only definition by omitting both.

    Args:
        alarms: Detector alarm positions.
        event_position: Integer position of the event date.
        statistic_on_event: Optional Bayesian posterior statistic on that date.
        threshold: Threshold paired with ``statistic_on_event``.
        lookback: Number of pre-event observations included in the alarm test.

    Returns:
        Whether either the alarm or supplied posterior criterion is satisfied.
    """
    if event_position < 0 or lookback < 1:
        raise ValueError("event position and lookback are invalid")
    if (statistic_on_event is None) != (threshold is None):
        raise ValueError("event statistic and threshold must be supplied together")
    positions = np.asarray(alarms, dtype=int)
    recent = positions[
        (positions >= max(0, event_position - lookback))
        & (positions < event_position)
    ]
    aware = len(recent) > 0
    if statistic_on_event is not None:
        aware = aware or bool(statistic_on_event >= float(threshold))
    return aware


def combined_event_delay(
    alarms: np.ndarray,
    event_position: int,
    *,
    statistic_on_event: float | None = None,
    threshold: float | None = None,
    pre_event_window: int = 15,
    detection_window: int = 50,
) -> int | None:
    """Return the delay under the combined alarm/posterior event criterion.

    An alarm qualifies in the half-open interval
    ``[event_position - pre_event_window, event_position + detection_window)``.
    Its signed delay is returned.  If there is no qualifying alarm, a supplied
    event-date Bayesian statistic at or above its threshold produces a
    posterior-only detection with delay zero.

    Args:
        alarms: Detector alarm positions.
        event_position: Integer position of the event date.
        statistic_on_event: Optional Bayesian posterior statistic on that date.
        threshold: Threshold paired with ``statistic_on_event``.
        pre_event_window: Number of pre-event positions accepted as anticipatory.
        detection_window: Number of post-event positions in the half-open window.

    Returns:
        Signed delay of the first qualifying alarm, zero for posterior-only
        detection, or ``None`` when the event is not detected.
    """
    if event_position < 0 or pre_event_window < 0 or detection_window < 1:
        raise ValueError("combined event-window settings are invalid")
    if (statistic_on_event is None) != (threshold is None):
        raise ValueError("event statistic and threshold must be supplied together")
    positions = np.asarray(alarms, dtype=int)
    hits = positions[
        (positions >= max(0, event_position - pre_event_window))
        & (positions < event_position + detection_window)
    ]
    if len(hits):
        return int(hits[0] - event_position)
    if statistic_on_event is not None and statistic_on_event >= float(threshold):
        return 0
    return None


@dataclass(frozen=True)
class _AssetEventResult:
    """Contract-level detector outputs returned by a parallel event-study job."""

    ticker: str
    dates: np.ndarray
    thresholds: dict[str, float]
    alarms: dict[str, np.ndarray]
    features: dict[str, pd.DataFrame]
    standardised_returns: np.ndarray


def _event_job(
    arguments: tuple[
        str,
        pd.Series,
        float,
        dict[str, float],
        float,
        int,
        ExperimentConfig,
    ]
) -> _AssetEventResult:
    """Calibrate detectors and analyse events for one contract."""
    ticker, returns, scale, prior_values, nu, ticker_index, config = arguments
    training = returns.loc[:TRAIN_END]
    calibration = (training.to_numpy(dtype=float) / scale)[
        -config.real_calibration_observations :
    ]
    testing_series = returns.loc[TEST_START:]
    testing = testing_series.to_numpy(dtype=float) / scale
    thresholds: dict[str, float] = {}
    alarms: dict[str, np.ndarray] = {}
    target_rate = 1.0 / config.target_arl0
    chart_grid = {
        "EWMA": np.linspace(0.5, 10.0, 192 if config.full_scale else 48),
        "other": np.linspace(1.0, 60.0, 472 if config.full_scale else 80),
    }
    for name, chart in CHARTS.items():
        grid = chart_grid["EWMA" if name == "EWMA" else "other"]
        thresholds[name] = _rate_calibrate(
            lambda threshold, chart=chart: _chart_alarm_count_and_exposure(
                chart, calibration, threshold
            ),
            grid,
            target_rate,
        )
        alarms[name] = chart.run(
            testing, thresholds[name], reestimate_after_alarm=True
        ).alarms

    prior = RegimePrior.from_dict(prior_values)
    features: dict[str, pd.DataFrame] = {}
    specifications = (
        ("BOCPD", "bocpd", None),
        ("BOCPD-GARCH", "bocpd-garch", None),
        ("BOCPD-GARCH-t", "bocpd-garch-t", nu),
    )
    for detector_index, (label, name, degrees) in enumerate(specifications):
        seed = (
            config.seed
            + 1_210_000
            + 10_000 * ticker_index
            + 1000 * detector_index
        )
        calibration_detector = make_detector(
            name,
            prior,
            nu=degrees,
            **_detector_settings(config, seed),
        )
        calibration_frame = calibration_detector.run(calibration)
        calibration_statistic = calibration_frame["young_prob"].to_numpy()
        candidates = np.quantile(
            calibration_statistic[config.burn_in :],
            np.linspace(0.5, 0.9999, 1000 if config.full_scale else 100),
        )
        thresholds[label] = _rate_calibrate(
            lambda threshold: (
                len(
                    upcrossing_alarms(
                        calibration_statistic, threshold, start=config.burn_in
                    )
                ),
                len(calibration) - config.burn_in,
            ),
            candidates,
            target_rate,
        )
        test_detector = make_detector(
            name,
            prior,
            nu=degrees,
            **_detector_settings(config, seed + 500),
        )
        frame = test_detector.run(pd.Series(testing, index=testing_series.index))
        features[label] = frame
        alarms[label] = upcrossing_alarms(
            frame["young_prob"].to_numpy(),
            thresholds[label],
            start=config.burn_in,
        )
    return _AssetEventResult(
        ticker=ticker,
        dates=testing_series.index.to_numpy(),
        thresholds=thresholds,
        alarms=alarms,
        features=features,
        standardised_returns=testing,
    )


def _event_windows(
    ticker: str, dates: pd.DatetimeIndex, detection_window: int
) -> list[tuple[int, int, str]]:
    """Return applicable labelled event windows for one contract."""
    asset_class = UNIVERSE[ticker][0]
    result = []
    for name, date, classes in EVENTS:
        if asset_class not in classes:
            continue
        position = int(dates.searchsorted(date))
        if 0 < position < len(dates):
            result.append((position, min(position + detection_window, len(dates)), name))
    return result


def outside_event_counts(
    alarms: np.ndarray,
    observation_count: int,
    windows: list[tuple[int, int, str]],
    burn_in: int,
) -> tuple[int, int]:
    """Count alarms and exposure strictly outside labelled event windows.

    Args:
        alarms: Detector alarm positions.
        observation_count: Total number of observations.
        windows: Half-open event windows as ``(low, high, label)`` tuples.
        burn_in: First monitored position.

    Returns:
        Off-event alarm count and off-event monitored exposure.
    """
    monitored_positions = np.arange(burn_in, observation_count)
    outside_mask = np.ones(len(monitored_positions), dtype=bool)
    for low, high, _ in windows:
        outside_mask &= ~(
            (monitored_positions >= low) & (monitored_positions < high)
        )
    alarm_count = sum(
        not any(low <= position < high for low, high, _ in windows)
        for position in alarms
        if position >= burn_in
    )
    return int(alarm_count), int(outside_mask.sum())


def run_event_study(
    inputs: TrainingInputs,
    config: ExperimentConfig,
    selected_nu: float,
    output_dir: Path,
    *,
    workers: int = 1,
) -> None:
    """Run the 2007--2024 market event study and write numerical outputs.

    Args:
        inputs: Market returns, scales, GARCH fits, and regime priors.
        config: BOCPD experiment profile.
        selected_nu: Pre-2007 selected finite Student-t degrees of freedom.
        output_dir: Destination for thresholds, alarms, summaries, and features.
        workers: Maximum number of contract jobs.
    """
    tickers = tuple(TICKERS) if config.full_scale else config.representative_tickers
    jobs = [
        (
            ticker,
            inputs.returns[ticker],
            inputs.scales[ticker],
            inputs.priors[UNIVERSE[ticker][0]].to_dict(),
            selected_nu,
            index,
            config,
        )
        for index, ticker in enumerate(tickers)
    ]
    results = map_jobs(
        _event_job,
        jobs,
        workers,
        label="Out-of-sample contract event studies",
        logger=LOGGER,
    )
    by_ticker = {result.ticker: result for result in results}
    output_dir.mkdir(parents=True, exist_ok=True)

    threshold_rows = []
    alarm_rows = []
    feature_frames = []
    for result in results:
        dates = pd.DatetimeIndex(result.dates)
        for detector, threshold in result.thresholds.items():
            threshold_rows.append(
                {"ticker": result.ticker, "detector": detector, "threshold": threshold}
            )
            for position in result.alarms[detector]:
                alarm_rows.append(
                    {
                        "ticker": result.ticker,
                        "detector": detector,
                        "date": dates[position],
                        "position": int(position),
                    }
                )
        for detector, frame in result.features.items():
            long = frame.reset_index(names="date")
            long.insert(1, "ticker", result.ticker)
            long.insert(2, "asset_class", UNIVERSE[result.ticker][0])
            long.insert(3, "detector", detector)
            long["standardised_return"] = result.standardised_returns
            feature_frames.append(long)
    atomic_write_csv(pd.DataFrame(threshold_rows), output_dir / "event_thresholds.csv")
    atomic_write_csv(pd.DataFrame(alarm_rows), output_dir / "event_alarms.csv")
    atomic_write_parquet(
        pd.concat(feature_frames, ignore_index=True),
        output_dir / "daily_detector_features.parquet",
    )

    summary_rows = []
    covid_rows = []
    awareness_rows = []
    combined_rows = []
    for detector_index, detector in enumerate(REAL_ALGORITHMS, start=1):
        LOGGER.info(
            "Event-study summaries: %d/%d (%s)",
            detector_index,
            len(REAL_ALGORITHMS),
            detector,
        )
        row: dict[str, object] = {"detector": detector}
        combined_row: dict[str, object] = {"detector": detector}
        for event_name, event_date, classes in EVENTS:
            delays = []
            combined_delays = []
            applicable = 0
            aware = 0
            for ticker, result in by_ticker.items():
                if UNIVERSE[ticker][0] not in classes:
                    continue
                dates = pd.DatetimeIndex(result.dates)
                position = int(dates.searchsorted(event_date))
                if not (0 < position < len(dates)):
                    continue
                applicable += 1
                positions = result.alarms[detector]
                hits = positions[
                    (positions >= position)
                    & (positions < position + config.detection_window)
                ]
                if len(hits):
                    delays.append(int(hits[0] - position))
                statistic_on_event = None
                threshold = None
                if detector.startswith("BOCPD"):
                    statistic_on_event = float(
                        result.features[detector]["young_prob"].iloc[position]
                    )
                    threshold = result.thresholds[detector]
                is_aware = event_awareness(
                    positions,
                    position,
                    statistic_on_event=statistic_on_event,
                    threshold=threshold,
                )
                aware += int(is_aware)
                combined_delay = combined_event_delay(
                    positions,
                    position,
                    statistic_on_event=statistic_on_event,
                    threshold=threshold,
                    detection_window=config.detection_window,
                )
                if combined_delay is not None:
                    combined_delays.append(combined_delay)
            row[f"{event_name}_detections"] = len(delays)
            row[f"{event_name}_applicable"] = applicable
            row[f"{event_name}_mean_delay"] = (
                float(np.mean(delays)) if delays else np.nan
            )
            row[f"{event_name}_median_delay"] = (
                float(np.median(delays)) if delays else np.nan
            )
            combined_row[f"{event_name}_detections"] = len(combined_delays)
            combined_row[f"{event_name}_applicable"] = applicable
            combined_row[f"{event_name}_mean_delay"] = (
                float(np.mean(combined_delays)) if combined_delays else np.nan
            )
            awareness_rows.append(
                {
                    "detector": detector,
                    "event": event_name,
                    "aware": aware,
                    "applicable": applicable,
                }
            )

        outside_alarms = outside_exposure = 0
        for ticker, result in by_ticker.items():
            dates = pd.DatetimeIndex(result.dates)
            windows = _event_windows(ticker, dates, config.detection_window)
            alarm_count, exposure = outside_event_counts(
                result.alarms[detector], len(dates), windows, config.burn_in
            )
            outside_alarms += alarm_count
            outside_exposure += exposure
        row["off_event_alarms"] = outside_alarms
        row["off_event_observations"] = outside_exposure
        row["off_event_alarms_per_asset_year"] = (
            252.0 * outside_alarms / outside_exposure
        )
        combined_row["off_event_alarms"] = outside_alarms
        combined_row["off_event_observations"] = outside_exposure
        combined_row["off_event_alarms_per_asset_year"] = (
            row["off_event_alarms_per_asset_year"]
        )
        summary_rows.append(row)
        combined_rows.append(combined_row)

        covid_date = pd.Timestamp("2020-02-24")
        for asset_class in (*CLASS_ORDER, "All"):
            delays = []
            applicable = 0
            for ticker, result in by_ticker.items():
                if asset_class != "All" and UNIVERSE[ticker][0] != asset_class:
                    continue
                dates = pd.DatetimeIndex(result.dates)
                position = int(dates.searchsorted(covid_date))
                if not (0 < position < len(dates)):
                    continue
                applicable += 1
                hits = result.alarms[detector]
                hits = hits[
                    (hits >= position) & (hits < position + config.detection_window)
                ]
                if len(hits):
                    delays.append(int(hits[0] - position))
            covid_rows.append(
                {
                    "detector": detector,
                    "asset_class": asset_class,
                    "detections": len(delays),
                    "applicable": applicable,
                    "median_delay": float(np.median(delays)) if delays else np.nan,
                }
            )
    atomic_write_csv(pd.DataFrame(summary_rows), output_dir / "event_summary.csv")
    atomic_write_csv(
        pd.DataFrame(combined_rows), output_dir / "event_summary_combined.csv"
    )
    atomic_write_csv(pd.DataFrame(covid_rows), output_dir / "covid_by_class.csv")
    atomic_write_csv(pd.DataFrame(awareness_rows), output_dir / "event_awareness.csv")
