"""Per-window BOCPD-GARCH-t and combined-CUSUM feature caches."""

from __future__ import annotations

from dataclasses import asdict
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from changepoint_detection.baselines import ControlChart
from changepoint_detection.models import (
    GarchFit,
    RegimePrior,
    fit_garch11,
    prior_from_fits,
)
from changepoint_detection.particle import make_detector
from changepoint_detection.progress import map_jobs
from experiment_data import STUDY_START, TICKERS, UNIVERSE

from .config import Profile
from .data import load_closes, log_returns_percent, selected_tickers, study_closes
from .utils import (
    assert_cache_fingerprint,
    atomic_csv,
    atomic_json,
    atomic_parquet,
    file_sha256,
    object_sha256,
    stable_seed,
)


LOGGER = logging.getLogger(__name__)
NU_VALUES: tuple[float | None, ...] = (4.0, 6.0, 10.0, None)


def nu_label(nu: float | None) -> str:
    """Return the table label for an innovation law.

    Args:
        nu: Student-t degrees of freedom, or ``None`` for Gaussian innovations.

    Returns:
        Stable innovation-law label.
    """
    return "Gaussian" if nu is None else f"t({nu:g})"


def _fit_job(arguments):
    """Fit one contract GARCH model, retrying a nonconverged initial fit."""
    ticker, values, starts, seed = arguments
    candidates = []
    try:
        candidates.append(fit_garch11(values, starts=starts, seed=seed))
    except RuntimeError:
        pass
    if not candidates or not candidates[0].converged:
        try:
            candidates.append(
                fit_garch11(values, starts=3 * starts, seed=seed + 1_000_003)
            )
        except RuntimeError:
            pass
    usable = [fit for fit in candidates if fit.has_finite_interior_solution]
    if not usable:
        raise RuntimeError(f"{ticker}: no usable GARCH fit")
    fit = min(usable, key=lambda item: item.negative_log_likelihood)
    return ticker, fit.to_dict()


def _garch_from_dict(values: dict[str, object]) -> GarchFit:
    """Reconstruct a typed GARCH fit from cached scalar values."""
    return GarchFit(
        mean=float(values["mean"]),
        omega=float(values["omega"]),
        alpha=float(values["alpha"]),
        beta=float(values["beta"]),
        hbar=float(values["hbar"]),
        persistence=float(values["persistence"]),
        negative_log_likelihood=float(values["negative_log_likelihood"]),
        converged=bool(values["converged"]),
        iterations=int(values["iterations"]),
    )


def prepare_window_calibration(
    archive: Path,
    profile: Profile,
    build_dir: Path,
    year: int,
    returns: dict[str, pd.Series],
    *,
    workers: int,
) -> dict[str, object]:
    """Fit or reuse expanding-window GARCH models and regime priors.

    Args:
        archive: Prepared archive used to fingerprint the cache.
        profile: Deep-momentum experiment profile.
        build_dir: Profile-specific cache directory.
        year: Test-window start year; calibration ends in the previous year.
        returns: Study-period percentage log returns keyed by ticker.
        workers: Maximum number of contract GARCH jobs.

    Returns:
        Calibration metadata containing scales, priors, and eligible contracts.
    """
    root = build_dir / "detectors" / f"w{year}"
    metadata_path = root / "calibration.json"
    tickers = selected_tickers(profile.tickers)
    train_end = pd.Timestamp(f"{year - 1}-12-31")
    fingerprint = object_sha256(
        {
            "stage": "window-calibration-v2",
            "archive": file_sha256(archive),
            "profile": profile.to_dict(),
            "year": year,
        }
    )
    if (
        assert_cache_fingerprint(metadata_path, fingerprint)
        and (root / "garch_fits.csv").exists()
    ):
        return json.loads(metadata_path.read_text())

    scales: dict[str, float] = {}
    jobs = []
    for ticker in tickers:
        training = returns[ticker].loc[:train_end].to_numpy(dtype=float)
        if len(training) < profile.minimum_prior_observations:
            continue
        scale = float(np.std(training, ddof=0))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"{ticker}: invalid pre-{year} return scale")
        scales[ticker] = scale
        jobs.append(
            (
                ticker,
                training / scale,
                profile.garch_fit_starts,
                stable_seed(profile.seed, "garch", year, ticker),
            )
        )
    fitted = map_jobs(
        _fit_job, jobs, workers, label=f"Window {year} GARCH fits", logger=LOGGER
    )
    fits = {ticker: _garch_from_dict(values) for ticker, values in fitted}
    pooled = tuple(fits.values())
    if len(pooled) < 2:
        raise ValueError(f"window {year}: fewer than two usable GARCH fits")
    priors: dict[str, RegimePrior] = {}
    prior_sources: dict[str, str] = {}
    classes = sorted({UNIVERSE[ticker][0] for ticker in fits})
    for asset_class in classes:
        class_fits = tuple(
            fit for ticker, fit in fits.items() if UNIVERSE[ticker][0] == asset_class
        )
        source = "asset_class"
        if len(class_fits) < 2:
            class_fits, source = pooled, "pooled_smoke_fallback"
        priors[asset_class] = prior_from_fits(class_fits)
        prior_sources[asset_class] = source
    fit_rows = []
    for ticker, fit in fits.items():
        fit_rows.append(
            {
                "ticker": ticker,
                "asset_class": UNIVERSE[ticker][0],
                "training_scale": scales[ticker],
                **fit.to_dict(),
            }
        )
    atomic_csv(root / "garch_fits.csv", pd.DataFrame(fit_rows))
    metadata = {
        "fingerprint": fingerprint,
        "archive_sha256": file_sha256(archive),
        "window": year,
        "training_end": str(train_end.date()),
        "eligible_tickers": list(fits),
        "scales": scales,
        "priors": {key: asdict(value) for key, value in priors.items()},
        "prior_sources": prior_sources,
        "nonconverged_fits": [ticker for ticker, fit in fits.items() if not fit.converged],
    }
    atomic_json(metadata_path, metadata)
    return metadata


def _nu_job(arguments):
    """Score one innovation law for one contract and filter seed."""
    ticker, values, prior_values, nu, seed, settings, burn_in = arguments
    name = "bocpd-garch" if nu is None else "bocpd-garch-t"
    detector = make_detector(
        name,
        RegimePrior.from_dict(prior_values),
        nu=nu,
        seed=seed,
        **settings,
    )
    frame = detector.run(values)
    scored = frame["predictive_log_likelihood"].iloc[int(burn_in) :]
    if scored.empty or not np.isfinite(scored.to_numpy(dtype=float)).all():
        raise RuntimeError(f"{ticker}: non-finite innovation-selection log score")
    return ticker, nu_label(nu), float(scored.mean())


def select_window_nu(
    profile: Profile,
    build_dir: Path,
    year: int,
    returns: dict[str, pd.Series],
    calibration: dict[str, object],
    *,
    workers: int,
) -> float:
    """Select finite Student-t degrees of freedom for one expanding window.

    Args:
        profile: Deep-momentum experiment profile.
        build_dir: Profile-specific cache directory.
        year: Test-window start year.
        returns: Study-period percentage log returns keyed by ticker.
        calibration: Window calibration metadata.
        workers: Maximum number of independent filter jobs.

    Returns:
        Degrees of freedom of the highest-scoring finite Student-t candidate.
    """
    root = build_dir / "detectors" / f"w{year}"
    output_path = root / "nu_selected.json"
    fingerprint = object_sha256(
        {
            "stage": "window-nu-selection-v2",
            "calibration": calibration["fingerprint"],
            "nu_values": [nu_label(value) for value in NU_VALUES],
            "selection_observations": profile.selection_observations,
            "selection_burn_in": profile.selection_burn_in,
            "selection_seeds": profile.selection_seeds,
            "particles": profile.particles,
            "max_run_lengths": profile.max_run_lengths,
        }
    )
    selection_outputs = (
        root / "nu_selection_contracts.csv",
        root / "nu_selection_classes.csv",
        root / "nu_selection.csv",
    )
    if assert_cache_fingerprint(output_path, fingerprint) and all(
        path.exists() for path in selection_outputs
    ):
        return float(json.loads(output_path.read_text())["selected_nu"])
    settings = {
        "particles": profile.particles,
        "hazard": profile.hazard,
        "max_run_lengths": profile.max_run_lengths,
        "young_window": profile.young_window,
    }
    train_end = pd.Timestamp(calibration["training_end"])
    jobs = []
    for ticker in calibration["eligible_tickers"]:
        values = returns[ticker].loc[:train_end].to_numpy(dtype=float)
        values = values[-profile.selection_observations :] / float(
            calibration["scales"][ticker]
        )
        asset_class = UNIVERSE[ticker][0]
        for nu in NU_VALUES:
            for replicate in range(profile.selection_seeds):
                jobs.append(
                    (
                        ticker,
                        values,
                        calibration["priors"][asset_class],
                        nu,
                        stable_seed(profile.seed, "nu", year, ticker, replicate),
                        settings,
                        profile.selection_burn_in,
                    )
                )
    raw = map_jobs(
        _nu_job,
        jobs,
        workers,
        label=f"Window {year} innovation selection",
        logger=LOGGER,
    )
    frame = pd.DataFrame(raw, columns=["ticker", "nu", "score"])
    per_ticker = frame.groupby(["ticker", "nu"], as_index=False)["score"].mean()
    per_ticker["asset_class"] = per_ticker["ticker"].map(
        lambda ticker: UNIVERSE[ticker][0]
    )
    per_class = per_ticker.groupby(["asset_class", "nu"], as_index=False)[
        "score"
    ].mean()
    pooled = per_class.groupby("nu")["score"].mean()
    order = [nu_label(value) for value in NU_VALUES]
    winner = max(order, key=lambda label: (float(pooled[label]), -order.index(label)))
    finite = order[:-1]
    selected_label = max(
        finite, key=lambda label: (float(pooled[label]), -finite.index(label))
    )
    selected_nu = {nu_label(value): value for value in NU_VALUES}[selected_label]
    atomic_csv(root / "nu_selection_contracts.csv", per_ticker)
    atomic_csv(root / "nu_selection_classes.csv", per_class)
    atomic_csv(
        root / "nu_selection.csv",
        pd.DataFrame({"nu": pooled.index, "pooled_score": pooled.values}),
    )
    atomic_json(
        output_path,
        {
            "fingerprint": fingerprint,
            "window": year,
            "selected_label": winner,
            "selected_t_label": selected_label,
            "selected_nu": float(selected_nu),
            "gaussian_preferred": winner == "Gaussian",
            "selection_observations": profile.selection_observations,
            "selection_burn_in": profile.selection_burn_in,
            "replicates": profile.selection_seeds,
        },
    )
    return float(selected_nu)


def _bocpd_job(arguments):
    """Generate and persist causal BOCPD-GARCH-t features for one contract."""
    ticker, values, scale, prior_values, nu, settings, seed, output_path = arguments
    detector = make_detector(
        "bocpd-garch-t",
        RegimePrior.from_dict(prior_values),
        nu=nu,
        seed=seed,
        **settings,
    )
    frame = detector.run(values / scale).reset_index(names="date")
    frame.insert(1, "ticker", ticker)
    atomic_parquet(Path(output_path), frame)
    return ticker, len(frame)


def _calibrate_chart(chart: ControlChart, values: np.ndarray, target_rate: float) -> float:
    """Calibrate a reset chart to the nearest empirical alarm rate."""
    best = None
    for threshold in np.linspace(1.0, 60.0, 472):
        run = chart.run(values, float(threshold), reestimate_after_alarm=True)
        exposure = int(run.monitored.sum())
        if exposure <= 0:
            continue
        row = (abs(len(run.alarms) / exposure - target_rate), -threshold, threshold)
        if best is None or row < best:
            best = row
    if best is None:
        raise RuntimeError("CUSUM threshold calibration has no monitored exposure")
    return float(best[2])


def _cusum_job(arguments):
    """Generate and persist combined mean/variance CUSUM features for one contract."""
    ticker, values, scale, n_training, profile_values, output_path = arguments
    burn = int(profile_values["burn"])
    chart = ControlChart("cusum-combined", burn_in=burn)
    tail = int(profile_values["calibration_observations"])
    calibration = values[max(0, n_training - tail) : n_training] / scale
    threshold = _calibrate_chart(
        chart, calibration, 1.0 / float(profile_values["target_arl"])
    )
    run = chart.run(values / scale, threshold, reestimate_after_alarm=True)
    alarm = np.zeros(len(values), dtype=np.int8)
    alarm[run.alarms] = 1
    positions = np.arange(len(values))
    last = np.where(alarm == 1, positions, -1)
    last = np.maximum.accumulate(last)
    age = np.where(last >= 0, positions - last, np.inf)
    frame = pd.DataFrame(
        {
            "date": profile_values["dates"],
            "ticker": ticker,
            "cmv_stat": np.clip(run.statistic / threshold, 0.0, 2.0),
            "cmv_recent": np.exp(-age / 21.0),
            "alarm": alarm,
            "monitored": run.monitored,
            "threshold": threshold,
        }
    )
    atomic_parquet(Path(output_path), frame)
    return ticker, len(frame), threshold


def prepare_window_detector_features(
    archive: Path,
    profile: Profile,
    build_dir: Path,
    year: int,
    returns: dict[str, pd.Series],
    calibration: dict[str, object],
    nu: float,
    *,
    workers: int,
) -> None:
    """Build BOCPD-GARCH-t and CUSUM feature caches for one test window.

    Args:
        archive: Prepared archive used to fingerprint the cache.
        profile: Deep-momentum experiment profile.
        build_dir: Profile-specific cache directory.
        year: Test-window start year.
        returns: Study-period percentage log returns keyed by ticker.
        calibration: Window calibration metadata.
        nu: Selected Student-t degrees of freedom.
        workers: Maximum number of independent contract jobs.
    """
    root = build_dir / "detectors" / f"w{year}"
    metadata_path = root / "features.json"
    test_end = pd.Timestamp(f"{year + profile.test_span - 1}-12-31")
    archive_sha256 = file_sha256(archive)
    fingerprint = object_sha256(
        {
            "stage": "window-detector-features-v2",
            "archive": archive_sha256,
            "calibration": calibration["fingerprint"],
            "nu": nu,
            "particles": profile.particles,
            "max_run_lengths": profile.max_run_lengths,
            "hazard": profile.hazard,
            "young_window": profile.young_window,
            "test_end": str(test_end.date()),
        }
    )
    tickers = tuple(calibration["eligible_tickers"])
    expected = [
        root / kind / f"{ticker}.parquet"
        for kind in ("bocpd", "cusum")
        for ticker in tickers
    ]
    metadata_existed = metadata_path.exists()
    complete = assert_cache_fingerprint(metadata_path, fingerprint)
    if complete:
        metadata = json.loads(metadata_path.read_text())
        complete = metadata.get("status", "complete") == "complete"
    if complete and all(path.exists() for path in expected):
        LOGGER.info("Window %d detector caches are complete", year)
        return
    rebuild_all = not metadata_existed and any(path.exists() for path in expected)
    if rebuild_all:
        LOGGER.warning(
            "Window %d detector files exist without metadata; rebuilding all %d files",
            year,
            len(expected),
        )
    atomic_json(
        metadata_path,
        {
            "fingerprint": fingerprint,
            "status": "building",
            "archive_sha256": archive_sha256,
            "window": year,
            "selected_nu": nu,
            "tickers": list(tickers),
        },
    )
    settings = {
        "particles": profile.particles,
        "hazard": profile.hazard,
        "max_run_lengths": profile.max_run_lengths,
        "young_window": profile.young_window,
    }
    bocpd_jobs, cusum_jobs = [], []
    train_end = pd.Timestamp(calibration["training_end"])
    for ticker in tickers:
        detector_start = max(
            pd.Timestamp(profile.data_start), pd.Timestamp(STUDY_START)
        )
        series = returns[ticker].loc[detector_start:test_end]
        bocpd_path = root / "bocpd" / f"{ticker}.parquet"
        if rebuild_all or not bocpd_path.exists():
            bocpd_jobs.append(
                (
                    ticker,
                    series,
                    float(calibration["scales"][ticker]),
                    calibration["priors"][UNIVERSE[ticker][0]],
                    nu,
                    settings,
                    stable_seed(profile.seed, "bocpd-features", year, ticker),
                    str(bocpd_path),
                )
            )
        cusum_path = root / "cusum" / f"{ticker}.parquet"
        if rebuild_all or not cusum_path.exists():
            n_training = int((series.index <= train_end).sum())
            cusum_jobs.append(
                (
                    ticker,
                    series.to_numpy(dtype=float),
                    float(calibration["scales"][ticker]),
                    n_training,
                    {
                        "burn": profile.chart_burn_in,
                        "calibration_observations": profile.chart_calibration_observations,
                        "target_arl": profile.chart_target_arl,
                        "dates": series.index.to_numpy(),
                    },
                    str(cusum_path),
                )
            )
    map_jobs(
        _bocpd_job,
        bocpd_jobs,
        workers,
        label=f"Window {year} BOCPD-GARCH-t features",
        logger=LOGGER,
    )
    map_jobs(
        _cusum_job,
        cusum_jobs,
        workers,
        label=f"Window {year} CUSUM-mv features",
        logger=LOGGER,
    )
    bocpd_rows = {
        ticker: len(
            pd.read_parquet(root / "bocpd" / f"{ticker}.parquet", columns=["date"])
        )
        for ticker in tickers
    }
    cusum_rows, thresholds = {}, {}
    for ticker in tickers:
        frame = pd.read_parquet(
            root / "cusum" / f"{ticker}.parquet",
            columns=["date", "threshold"],
        )
        cusum_rows[ticker] = len(frame)
        thresholds[ticker] = float(frame["threshold"].iloc[0])
    atomic_json(
        metadata_path,
        {
            "fingerprint": fingerprint,
            "status": "complete",
            "archive_sha256": archive_sha256,
            "window": year,
            "selected_nu": nu,
            "tickers": list(tickers),
            "bocpd_rows": bocpd_rows,
            "cusum_rows": cusum_rows,
            "cusum_thresholds": thresholds,
        },
    )


def prepare_detector_features(
    archive: Path, profile: Profile, build_dir: Path, *, workers: int
) -> dict[int, dict[str, object]]:
    """Prepare all expanding-window detector features required by a profile.

    Args:
        archive: Prepared continuous-futures ZIP archive.
        profile: Deep-momentum experiment profile.
        build_dir: Profile-specific cache directory.
        workers: Maximum number of independent contract/filter jobs.

    Returns:
        Calibration metadata keyed by test-window start year.
    """
    tickers = selected_tickers(profile.tickers)
    closes = study_closes(load_closes(archive, tickers))
    returns = {ticker: log_returns_percent(close) for ticker, close in closes.items()}
    calibrations = {}
    for index, year in enumerate(profile.windows, start=1):
        LOGGER.info("Detector window %d/%d: %d", index, len(profile.windows), year)
        calibration = prepare_window_calibration(
            archive, profile, build_dir, year, returns, workers=workers
        )
        nu = select_window_nu(
            profile, build_dir, year, returns, calibration, workers=workers
        )
        prepare_window_detector_features(
            archive,
            profile,
            build_dir,
            year,
            returns,
            calibration,
            nu,
            workers=workers,
        )
        calibrations[year] = calibration
    return calibrations


def load_bocpd_frame(build_dir: Path, year: int, ticker: str) -> pd.DataFrame:
    """Load cached BOCPD features for tensor construction.

    Args:
        build_dir: Profile-specific detector cache directory.
        year: Test-window start year.
        ticker: Contract identifier.

    Returns:
        Normalised BOCPD feature frame indexed by date.
    """
    path = build_dir / "detectors" / f"w{year}" / "bocpd" / f"{ticker}.parquet"
    frame = pd.read_parquet(path).set_index("date")
    return pd.DataFrame(
        {
            "bocpd_score": frame["young_prob"],
            "bocpd_rl": (frame["expected_run_length"] / 21.0).clip(0.0, 1.0),
            "bocpd_cp_prob": frame["cp_prob"],
            "bocpd_exp_vol": frame["expected_volatility"],
        }
    )


def load_cusum_frame(build_dir: Path, year: int, ticker: str) -> pd.DataFrame:
    """Load cached combined-CUSUM features for tensor construction.

    Args:
        build_dir: Profile-specific detector cache directory.
        year: Test-window start year.
        ticker: Contract identifier.

    Returns:
        Combined-CUSUM feature frame indexed by date.
    """
    path = build_dir / "detectors" / f"w{year}" / "cusum" / f"{ticker}.parquet"
    frame = pd.read_parquet(path).set_index("date")
    return frame[["cmv_stat", "cmv_recent"]]
