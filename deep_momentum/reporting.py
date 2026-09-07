"""Numerical reductions, structural validation, and provenance manifests."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from experiment_data import TICKERS, UNIVERSE

from .config import MODEL_NAMES, PRIMARY_WINDOWS, Profile
from .data import selected_tickers
from .training import load_search_records
from .utils import atomic_csv, atomic_json, atomic_parquet, file_sha256


LOGGER = logging.getLogger(__name__)
COST_GRID = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0)


def performance_metrics(daily: pd.Series) -> dict[str, float]:
    """Calculate the dissertation's annualised daily-return performance measures.

    Args:
        daily: Daily strategy returns in decimal units.

    Returns:
        Return, risk, drawdown, and gain/loss statistics used in result tables.
    """
    daily = daily.dropna()
    if len(daily) < 2:
        return {name: np.nan for name in (
            "Returns", "Vol.", "Sharpe", "Downside Dev.", "Sortino", "MDD",
            "Calmar", "% +ve Ret.", "Ave. P / Ave. L"
        )}
    mean, standard_deviation = float(daily.mean()), float(daily.std(ddof=1))
    downside = float(daily[daily < 0].std(ddof=1))
    wealth = (1.0 + daily).cumprod()
    drawdown = float((1.0 - wealth / wealth.cummax()).max())
    annual_return = mean * 252.0
    positive = daily > 0
    negative = daily < 0
    profit_mean = float(daily[positive].mean())
    loss_mean = float(daily[negative].mean())
    profit_loss_ratio = (
        -profit_mean / loss_mean
        if positive.any() and negative.any() and loss_mean < 0
        else np.nan
    )
    return {
        "Returns": annual_return * 100.0,
        "Vol.": standard_deviation * np.sqrt(252.0) * 100.0,
        "Sharpe": mean / standard_deviation * np.sqrt(252.0) if standard_deviation else np.nan,
        "Downside Dev.": downside * np.sqrt(252.0) * 100.0,
        "Sortino": annual_return / (downside * np.sqrt(252.0)) if downside else np.nan,
        "MDD": drawdown * 100.0,
        "Calmar": annual_return / drawdown if drawdown else np.nan,
        "% +ve Ret.": float(positive.mean()) * 100.0,
        "Ave. P / Ave. L": profit_loss_ratio,
    }


def _window_metrics(daily: pd.Series, windows: tuple[int, ...], span: int) -> dict[str, float]:
    """Calculate equally weighted mean performance statistics across windows."""
    rows = [
        performance_metrics(daily.loc[f"{year}-01-01" : f"{year + span - 1}-12-31"])
        for year in windows
    ]
    return pd.DataFrame(rows).mean(numeric_only=True).to_dict()


def _ensemble_predictions(records: list[dict[str, object]]) -> pd.DataFrame:
    """Average member positions on their shared contract-date prediction grid."""
    frames = [pd.read_parquet(record["prediction_path"]) for record in records]
    combined = pd.concat(frames, ignore_index=True)
    result = combined.groupby(["date", "ticker"], as_index=False).agg(
        signal_date=("signal_date", "first"),
        target=("target", "mean"),
        lev=("lev", "mean"),
        position=("position", "mean"),
    )
    result["captured_return"] = result["position"] * result["target"]
    return result


def report_ensemble_size(profile: Profile, ensemble: int | None = None) -> int:
    """Resolve and validate the reporting ensemble independently of caches.

    Args:
        profile: Deep-momentum experiment profile.
        ensemble: Explicit member count, or ``None`` for the top-five default.

    Returns:
        Validated ensemble member count.
    """
    count = min(5, profile.trials) if ensemble is None else int(ensemble)
    if count < 1 or count > profile.trials:
        raise ValueError(
            f"ensemble size must be between 1 and {profile.trials}, found {count}"
        )
    return count


def selected_predictions(
    profile: Profile, build_dir: Path, *, ensemble: int | None = None
) -> tuple[dict[str, pd.DataFrame], list[dict[str, object]]]:
    """Select validation-ranked members and aggregate their test positions.

    Args:
        profile: Deep-momentum experiment profile.
        build_dir: Profile-specific training cache directory.
        ensemble: Explicit member count, or ``None`` for the default.

    Returns:
        Ensemble predictions by model and selected-member metadata by window.
    """
    count = report_ensemble_size(profile, ensemble)
    by_model: dict[str, list[pd.DataFrame]] = {name: [] for name in MODEL_NAMES}
    selected = []
    for year in profile.windows:
        for model in MODEL_NAMES:
            valid = [
                record
                for record in load_search_records(build_dir, year, model, profile.trials)
                if record["status"] == "success"
            ]
            if len(valid) < count:
                raise RuntimeError(
                    f"{model} window {year}: {len(valid)} valid trials for top-{count}"
                )
            top = sorted(valid, key=lambda record: -float(record["val_sharpe"]))[:count]
            frame = _ensemble_predictions(top)
            frame["window"] = year
            by_model[model].append(frame)
            best = dict(top[0])
            best["ensemble_members"] = [int(record["index"]) for record in top]
            selected.append(best)
    return {
        model: pd.concat(frames, ignore_index=True) for model, frames in by_model.items()
    }, selected


def _portfolio(frame: pd.DataFrame, divisor: int, *, cost_bps: float = 0.0) -> pd.Series:
    """Aggregate contract predictions into a daily equal-risk portfolio."""
    ordered = frame.sort_values(["ticker", "date"]).copy()
    gross = ordered["position"] * ordered["target"]
    holding = ordered["position"] * ordered["lev"]
    turnover = holding.groupby(ordered["ticker"]).diff().abs().fillna(0.0)
    ordered["net"] = gross - float(cost_bps) * 1e-4 * turnover
    return (ordered.groupby("date")["net"].sum() / divisor).sort_index()


def _rescale(daily: pd.Series, windows: tuple[int, ...], span: int) -> pd.Series:
    """Rescale daily returns to 15% mean annualised window volatility."""
    volatilities = [
        daily.loc[f"{year}-01-01" : f"{year + span - 1}-12-31"].std(ddof=1)
        * np.sqrt(252.0)
        for year in windows
    ]
    mean_volatility = float(np.nanmean(volatilities))
    return daily * 0.15 / mean_volatility if mean_volatility > 0 else daily * np.nan


def _safe_name(name: str) -> str:
    """Convert a model label to its stable lowercase filename component."""
    return name.replace("@", "_at_").replace("-", "_").lower()


def generate_reports(
    profile: Profile,
    build_dir: Path,
    output_dir: Path,
    *,
    ensemble: int | None = None,
) -> None:
    """Generate all reported deep-momentum numerical reductions.

    Args:
        profile: Deep-momentum experiment profile.
        build_dir: Profile-specific training and detector cache directory.
        output_dir: Destination for positions and result tables.
        ensemble: Number of validation-ranked trial members to average.
    """
    ensemble_size = report_ensemble_size(profile, ensemble)
    LOGGER.info("Reducing trial predictions with a top-%d ensemble", ensemble_size)
    output_dir.mkdir(parents=True, exist_ok=True)
    position_root = output_dir / "positions"
    position_root.mkdir(parents=True, exist_ok=True)
    predictions, selected = selected_predictions(
        profile, build_dir, ensemble=ensemble_size
    )
    divisor = len(TICKERS) if profile.full_scale else len(selected_tickers(profile.tickers))
    aggregate_windows = tuple(year for year in profile.windows if year in PRIMARY_WINDOWS)
    if not aggregate_windows:
        aggregate_windows = profile.windows
    portfolios = {
        model: _portfolio(frame, divisor) for model, frame in predictions.items()
    }
    for model, frame in predictions.items():
        atomic_parquet(position_root / f"{_safe_name(model)}.parquet", frame)
    for model, daily in portfolios.items():
        atomic_csv(
            output_dir / f"portfolio_{_safe_name(model)}.csv",
            daily.rename("captured_return").reset_index(),
        )
    raw = pd.DataFrame(
        {
            model: _window_metrics(daily, aggregate_windows, profile.test_span)
            for model, daily in portfolios.items()
        }
    ).T
    rescaled = pd.DataFrame(
        {
            model: _window_metrics(
                _rescale(daily, aggregate_windows, profile.test_span),
                aggregate_windows,
                profile.test_span,
            )
            for model, daily in portfolios.items()
        }
    ).T
    by_window = pd.DataFrame(
        {
            model: {
                str(year): performance_metrics(
                    daily.loc[f"{year}-01-01" : f"{year + profile.test_span - 1}-12-31"]
                )["Sharpe"]
                for year in profile.windows
            }
            for model, daily in portfolios.items()
        }
    ).T
    selected_rows = []
    for record in selected:
        selected_rows.append(
            {
                "model": record["model"],
                "window": record["window"],
                "trial": record["index"],
                "validation_sharpe": record["val_sharpe"],
                "best_epoch": record["best_epoch"],
                **record["hp"],
                "ensemble_members": ",".join(map(str, record["ensemble_members"])),
            }
        )
    selected_table = pd.DataFrame(selected_rows)
    sensitivity_rows = []
    for count in sorted({1, min(5, profile.trials), 10, ensemble_size}):
        if count > profile.trials:
            continue
        alternative, _ = selected_predictions(profile, build_dir, ensemble=count)
        for model, frame in alternative.items():
            daily = _portfolio(frame, divisor)
            sensitivity_rows.append(
                {
                    "model": model,
                    "ensemble": count,
                    "Sharpe": _window_metrics(
                        daily, aggregate_windows, profile.test_span
                    )["Sharpe"],
                }
            )
    sensitivity = pd.DataFrame(sensitivity_rows)
    cost_rows = []
    for model, frame in predictions.items():
        for cost in COST_GRID:
            daily = _portfolio(frame, divisor, cost_bps=cost)
            cost_rows.append(
                {
                    "model": model,
                    "cost_bps": cost,
                    "Sharpe": _window_metrics(
                        daily, aggregate_windows, profile.test_span
                    )["Sharpe"],
                }
            )
    costs = pd.DataFrame(cost_rows)
    seed_rows, seed_ensemble_rows = [], []
    for model in MODEL_NAMES:
        replicate_portfolios = []
        for replicate in range(profile.replicates):
            frames = []
            for year in profile.windows:
                path = (
                    build_dir
                    / "training"
                    / f"w{year}"
                    / model
                    / f"positions_replicate{replicate:03d}.parquet"
                )
                if not path.exists():
                    raise FileNotFoundError(path)
                frame = pd.read_parquet(path)
                frame["window"] = year
                frames.append(frame)
            combined = pd.concat(frames, ignore_index=True)
            daily = _portfolio(combined, divisor)
            replicate_portfolios.append(daily.rename(replicate))
            seed_rows.append(
                {
                    "model": model,
                    "replicate": replicate,
                    "Sharpe": _window_metrics(
                        daily, aggregate_windows, profile.test_span
                    )["Sharpe"],
                }
            )
        wide = pd.concat(replicate_portfolios, axis=1)
        seed_ensemble = wide.mean(axis=1)
        seed_ensemble_rows.append(
            {
                "model": model,
                "seed_ensemble_sharpe": _window_metrics(
                    seed_ensemble, aggregate_windows, profile.test_span
                )["Sharpe"],
            }
        )
    seed_detail = pd.DataFrame(seed_rows)
    seed_summary = seed_detail.groupby("model")["Sharpe"].agg(
        n="count", mean="mean", std="std", minimum="min", maximum="max"
    )
    seed_summary["standard_error"] = seed_summary["std"] / np.sqrt(seed_summary["n"])
    seed_summary = seed_summary.reset_index().merge(
        pd.DataFrame(seed_ensemble_rows), on="model", how="left"
    )
    detector_rows, nu_frames, nu_class_frames, nu_contract_frames = [], [], [], []
    fit_frames = []
    for year in profile.windows:
        detector_root = build_dir / "detectors" / f"w{year}"
        calibration = json.loads((detector_root / "calibration.json").read_text())
        selection = json.loads((detector_root / "nu_selected.json").read_text())
        class_counts = {asset_class: 0 for asset_class in ("CM", "EQ", "FI", "FX")}
        for ticker in calibration["eligible_tickers"]:
            class_counts[UNIVERSE[ticker][0]] += 1
        detector_rows.append(
            {
                "window": year,
                "training_end": calibration["training_end"],
                "eligible_contracts": len(calibration["eligible_tickers"]),
                **{f"eligible_{key}": value for key, value in class_counts.items()},
                "selected_nu": selection["selected_nu"],
                "overall_winner": selection["selected_label"],
                "selected_t_candidate": selection["selected_t_label"],
                "gaussian_preferred": selection["gaussian_preferred"],
                "nonconverged_garch_fits": len(calibration["nonconverged_fits"]),
                "eligible_tickers": ",".join(calibration["eligible_tickers"]),
            }
        )
        nu_frame = pd.read_csv(detector_root / "nu_selection.csv")
        nu_frame.insert(0, "window", year)
        nu_frames.append(nu_frame)
        nu_class_frame = pd.read_csv(detector_root / "nu_selection_classes.csv")
        nu_class_frame.insert(0, "window", year)
        nu_class_frames.append(nu_class_frame)
        nu_contract_frame = pd.read_csv(
            detector_root / "nu_selection_contracts.csv"
        )
        nu_contract_frame.insert(0, "window", year)
        nu_contract_frames.append(nu_contract_frame)
        fit_frame = pd.read_csv(detector_root / "garch_fits.csv")
        fit_frame.insert(0, "window", year)
        fit_frames.append(fit_frame)
    detector_calibration = pd.DataFrame(detector_rows)
    detector_nu_scores = pd.concat(nu_frames, ignore_index=True)
    detector_nu_class_scores = pd.concat(nu_class_frames, ignore_index=True)
    detector_nu_contract_scores = pd.concat(nu_contract_frames, ignore_index=True)
    detector_garch_fits = pd.concat(fit_frames, ignore_index=True)
    for name, table in {
        "metrics_raw": raw,
        "metrics_rescaled": rescaled,
        "sharpe_by_window": by_window,
        "chosen_hyperparameters": selected_table,
        "ensemble_sensitivity": sensitivity,
        "cost_sweep": costs,
        "seed_detail": seed_detail,
        "seed_summary": seed_summary,
        "detector_calibration": detector_calibration,
        "detector_nu_scores": detector_nu_scores,
        "detector_nu_class_scores": detector_nu_class_scores,
        "detector_nu_contract_scores": detector_nu_contract_scores,
        "detector_garch_fits": detector_garch_fits,
    }.items():
        frame = table.reset_index(names="model") if table.index.name is None and name in {
            "metrics_raw", "metrics_rescaled", "sharpe_by_window"
        } else table
        atomic_csv(output_dir / f"{name}.csv", frame)

    LOGGER.info("Deep-momentum reports written to %s", output_dir)


REQUIRED_OUTPUTS = (
    "metrics_raw.csv",
    "metrics_rescaled.csv",
    "sharpe_by_window.csv",
    "chosen_hyperparameters.csv",
    "ensemble_sensitivity.csv",
    "cost_sweep.csv",
    "seed_detail.csv",
    "seed_summary.csv",
    "detector_calibration.csv",
    "detector_nu_scores.csv",
    "detector_nu_class_scores.csv",
    "detector_nu_contract_scores.csv",
    "detector_garch_fits.csv",
    "gating/gating_summary.csv",
    "gating/gating_by_window.csv",
    "gating/gating_cost_sweep.csv",
    "gating/gating_contract_abstention.csv",
    "gating/gate_by_class.csv",
    "gating/gate_by_class_windows.csv",
    "gating/bootstrap.csv",
    "gating/metadata.json",
    "gating/gate_by_class_metadata.json",
    "gating/bootstrap_metadata.json",
) + tuple(f"positions/{_safe_name(model)}.parquet" for model in MODEL_NAMES) + tuple(
    f"portfolio_{_safe_name(model)}.csv" for model in MODEL_NAMES
) + tuple(
    f"gating/gating_daily_{_safe_name(model)}_{kind}.parquet"
    for model in (
        "LSTM",
        "LSTM-BOCPD",
        "LSTM-BOCPD-rich",
        "LSTM-BOCPD-sys",
        "LSTM-BOCPD-full",
    )
    for kind in ("trial", "seed")
)


def _expected_training_records(
    root: Path,
    profile: Profile,
    *,
    mode: str,
) -> list[Path]:
    """Return existing records for the active roster and configured indices.

    Args:
        root: Root directory of training records.
        profile: Deep-momentum experiment profile.
        mode: ``"search"`` or ``"replicate"``.

    Returns:
        Existing record paths over every configured model, window, and index.
    """
    if mode == "search":
        stem, count = "trial", profile.trials
    elif mode == "replicate":
        stem, count = "replicate", profile.replicates
    else:
        raise ValueError(f"unknown training-record mode: {mode}")
    return [
        path
        for year in profile.windows
        for model in MODEL_NAMES
        for index in range(count)
        if (
            path := root
            / f"w{year}"
            / model
            / f"{stem}{index:03d}.json"
        ).exists()
    ]


def validate_outputs(
    profile: Profile,
    build_dir: Path,
    output_dir: Path,
    *,
    ensemble: int | None = None,
) -> dict[str, object]:
    """Validate completeness, schemas, and numerical invariants of all outputs.

    Args:
        profile: Profile used to produce the outputs.
        build_dir: Profile-specific training and detector cache directory.
        output_dir: Completed numerical-output directory.
        ensemble: Reporting ensemble size used for the outputs.

    Returns:
        Compact validation summary suitable for the provenance manifest.
    """
    ensemble_size = report_ensemble_size(profile, ensemble)
    missing = [relative for relative in REQUIRED_OUTPUTS if not (output_dir / relative).exists()]
    if missing:
        raise FileNotFoundError("missing deep-momentum outputs: " + ", ".join(missing))
    expected_search = len(MODEL_NAMES) * len(profile.windows) * profile.trials
    expected_replicates = len(MODEL_NAMES) * len(profile.windows) * profile.replicates
    record_root = build_dir / "training"
    search_records = _expected_training_records(
        record_root, profile, mode="search"
    )
    replicate_records = _expected_training_records(
        record_root, profile, mode="replicate"
    )
    if len(search_records) != expected_search:
        raise RuntimeError(f"expected {expected_search} search records, found {len(search_records)}")
    if len(replicate_records) != expected_replicates:
        raise RuntimeError(
            f"expected {expected_replicates} replicate records, found {len(replicate_records)}"
        )
    failed_replicates = [
        str(path)
        for path in replicate_records
        if json.loads(path.read_text()).get("status") != "success"
    ]
    if failed_replicates:
        raise RuntimeError(f"failed seed replicates: {len(failed_replicates)}")
    raw = pd.read_csv(output_dir / "metrics_raw.csv")
    if len(raw) != len(MODEL_NAMES) or not np.isfinite(
        raw.select_dtypes(include=[np.number]).to_numpy(dtype=float)
    ).all():
        raise ValueError("primary metrics are incomplete or non-finite")
    rescaled = pd.read_csv(output_dir / "metrics_rescaled.csv")
    windows = pd.read_csv(output_dir / "sharpe_by_window.csv")
    costs = pd.read_csv(output_dir / "cost_sweep.csv")
    seed_detail = pd.read_csv(output_dir / "seed_detail.csv")
    seed_summary = pd.read_csv(output_dir / "seed_summary.csv")
    sensitivity = pd.read_csv(output_dir / "ensemble_sensitivity.csv")
    expected_sensitivity = len(MODEL_NAMES) * len(
        {count for count in (1, min(5, profile.trials), 10) if count <= profile.trials}
    )
    expected_rows = {
        "rescaled headline": (rescaled, len(MODEL_NAMES)),
        "window Sharpe": (windows, len(MODEL_NAMES)),
        "cost sweep": (costs, len(MODEL_NAMES) * len(COST_GRID)),
        "seed detail": (seed_detail, len(MODEL_NAMES) * profile.replicates),
        "seed summary": (seed_summary, len(MODEL_NAMES)),
        "ensemble sensitivity": (sensitivity, expected_sensitivity),
    }
    for label, (frame, expected) in expected_rows.items():
        numeric = frame.select_dtypes(include=[np.number]).to_numpy(dtype=float)
        if len(frame) != expected or not np.isfinite(numeric).all():
            raise ValueError(f"{label} output is incomplete or non-finite")
    selected = pd.read_csv(output_dir / "chosen_hyperparameters.csv")
    expected_selected = len(MODEL_NAMES) * len(profile.windows)
    if len(selected) != expected_selected:
        raise ValueError(
            f"expected {expected_selected} selected model-window rows, "
            f"found {len(selected)}"
        )
    member_counts = selected["ensemble_members"].astype(str).map(
        lambda value: len([member for member in value.split(",") if member])
    )
    if not member_counts.eq(ensemble_size).all():
        raise ValueError(
            f"selected positions do not all contain top-{ensemble_size} membership"
        )
    headline_windows = tuple(
        year for year in profile.windows if year in PRIMARY_WINDOWS
    ) or profile.windows
    calibration = pd.read_csv(output_dir / "detector_calibration.csv")
    if len(calibration) != len(profile.windows):
        raise ValueError("detector calibration summary is incomplete")
    fits = pd.read_csv(output_dir / "detector_garch_fits.csv")
    if len(fits) != int(calibration["eligible_contracts"].sum()):
        raise ValueError("consolidated GARCH-fit output is incomplete")
    nu_scores = pd.read_csv(output_dir / "detector_nu_scores.csv")
    if len(nu_scores) != 4 * len(profile.windows):
        raise ValueError("pooled innovation-score output is incomplete")
    nu_contracts = pd.read_csv(output_dir / "detector_nu_contract_scores.csv")
    if len(nu_contracts) != 4 * int(calibration["eligible_contracts"].sum()):
        raise ValueError("contract innovation-score output is incomplete")
    active_classes = calibration[
        ["eligible_CM", "eligible_EQ", "eligible_FI", "eligible_FX"]
    ].gt(0).sum(axis=1).sum()
    nu_classes = pd.read_csv(output_dir / "detector_nu_class_scores.csv")
    if len(nu_classes) != 4 * int(active_classes):
        raise ValueError("asset-class innovation-score output is incomplete")
    gating = pd.read_csv(output_dir / "gating" / "gating_summary.csv")
    gating_required = {
        "model",
        "ensemble",
        "variant",
        "statistic",
        "rule",
        "p",
        "m",
        "sharpe",
        "mdd",
        "abstention_rate",
        "mean_turnover",
        "conditional_sharpe",
        "conditional_contracts",
        "conditional_excluded",
    }
    if not gating_required.issubset(gating.columns) or len(gating) != 5 * 2 * 19:
        raise ValueError("gating summary schema or row count is incomplete")
    if not np.isfinite(
        gating[
            [
                "p",
                "sharpe",
                "mdd",
                "abstention_rate",
                "mean_turnover",
                "conditional_sharpe",
                "conditional_contracts",
                "conditional_excluded",
            ]
        ].to_numpy(dtype=float)
    ).all():
        raise ValueError("gating summary contains non-finite values")
    if not set(gating["variant"]).issubset(
        {"ungated", "rank-hyst", "rank-cwhyst"}
    ):
        raise ValueError("gating summary contains an unsupported policy")
    ungated = gating[gating["variant"] == "ungated"]
    if len(ungated) != 2 * 5 or not (
        ungated["p"].eq(0.0) & ungated["m"].astype(str).isin({"1", "1.0"})
    ).all():
        raise ValueError("gating summary lacks one explicit baseline per model/kind")
    bootstrap = pd.read_csv(output_dir / "gating" / "bootstrap.csv")
    if len(bootstrap) != 2 * 2 * 3 * 6:
        raise ValueError("gating bootstrap output is incomplete")
    if set(bootstrap["B"]) != {profile.bootstrap_samples}:
        raise ValueError("gating bootstrap sample count differs from the profile")
    if not np.isfinite(
        bootstrap[["delta", "p_value"]].to_numpy(dtype=float)
    ).all():
        raise ValueError("gating bootstrap contains non-finite statistics")
    if not bootstrap["p_value"].between(0.0, 1.0).all():
        raise ValueError("gating bootstrap contains an invalid P-value")
    bootstrap_key = ["kind", "policy", "cost_bps", "model"]
    if bootstrap.duplicated(bootstrap_key).any():
        raise ValueError("gating bootstrap contains duplicate experiment cells")
    class_summary = pd.read_csv(output_dir / "gating" / "gate_by_class.csv")
    class_windows = pd.read_csv(
        output_dir / "gating" / "gate_by_class_windows.csv"
    )
    class_required = {
        "model",
        "asset_class",
        "contracts",
        "abstention",
        "cond_ungated",
        "cond_gated",
        "sleeve_ungated",
        "sleeve_hyst",
        "share_baseline",
        "share_confidence",
    }
    active_classes = class_summary["asset_class"].nunique()
    if (
        not class_required.issubset(class_summary.columns)
        or len(class_summary) != 5 * active_classes
        or active_classes < 1
    ):
        raise ValueError("asset-class gating output is incomplete")
    if len(class_windows) != len(class_summary) * 3 * len(profile.windows) * 3:
        raise ValueError("window-level asset-class gating output is incomplete")
    for label, frame in (
        ("asset-class gating", class_summary),
        ("window-level asset-class gating", class_windows),
    ):
        if not np.isfinite(
            frame.select_dtypes(include=[np.number]).to_numpy(dtype=float)
        ).all():
            raise ValueError(f"{label} contains non-finite values")
    for model in MODEL_NAMES:
        positions = pd.read_parquet(
            output_dir / "positions" / f"{_safe_name(model)}.parquet"
        )
        required = {
            "date",
            "signal_date",
            "ticker",
            "target",
            "lev",
            "position",
            "captured_return",
            "window",
        }
        if not required.issubset(positions.columns):
            raise ValueError(f"{model}: ensemble-position schema is incomplete")
        if positions.duplicated(["date", "ticker"]).any():
            raise ValueError(f"{model}: duplicate ensemble positions")
        if (positions["signal_date"] >= positions["date"]).any():
            raise ValueError(f"{model}: signal dates do not precede return dates")
        numeric = positions[["target", "lev", "position", "captured_return"]]
        if not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise ValueError(f"{model}: ensemble positions contain non-finite values")
        portfolio = pd.read_csv(
            output_dir / f"portfolio_{_safe_name(model)}.csv",
            parse_dates=["date"],
        )
        if (
            portfolio.empty
            or portfolio["date"].duplicated().any()
            or not np.isfinite(
                portfolio["captured_return"].to_numpy(dtype=float)
            ).all()
        ):
            raise ValueError(f"{model}: portfolio output is invalid")
    for model in (
        "LSTM",
        "LSTM-BOCPD",
        "LSTM-BOCPD-rich",
        "LSTM-BOCPD-sys",
        "LSTM-BOCPD-full",
    ):
        for kind in ("trial", "seed"):
            daily = pd.read_parquet(
                output_dir
                / "gating"
                / f"gating_daily_{_safe_name(model)}_{kind}.parquet"
            )
            required = {
                "date",
                "variant",
                "p",
                "m",
                "gross",
                "net",
                "turnover",
                "n_available",
                "n_active",
            }
            if not required.issubset(daily.columns):
                raise ValueError(f"{model} {kind}: gating daily schema is incomplete")
            if not np.isfinite(
                daily[["gross", "net", "turnover"]].to_numpy(dtype=float)
            ).all():
                raise ValueError(f"{model} {kind}: non-finite gating daily values")
            if daily.duplicated(["date", "variant", "p", "m"]).any():
                raise ValueError(f"{model} {kind}: duplicate gating daily rows")
    return {
        "profile": profile.name,
        "report_ensemble": ensemble_size,
        "headline_windows": list(headline_windows),
        "models": len(MODEL_NAMES),
        "windows": len(profile.windows),
        "search_records": len(search_records),
        "replicate_records": len(replicate_records),
        "training_record_source": "build",
        "structural_validation_passed": True,
    }


def write_manifest(
    profile: Profile,
    archive: Path,
    build_dir: Path,
    output_dir: Path,
    validation: dict[str, object],
    *,
    ensemble: int | None = None,
) -> None:
    """Write input, profile, validation, and output provenance hashes.

    Args:
        profile: Profile used for the experiment.
        archive: Prepared continuous-futures input archive.
        build_dir: Profile-specific cache directory.
        output_dir: Completed numerical-output directory.
        validation: Summary returned by :func:`validate_outputs`.
        ensemble: Reporting ensemble member count.
    """
    ensemble_size = report_ensemble_size(profile, ensemble)
    outputs = {}
    for path in sorted(output_dir.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            outputs[str(path.relative_to(output_dir))] = file_sha256(path)
    atomic_json(
        output_dir / "manifest.json",
        {
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "archive": str(archive.resolve()),
            "archive_sha256": file_sha256(archive),
            "profile": profile.to_dict(),
            "report_ensemble": ensemble_size,
            "validation": validation,
            "python": platform.python_version(),
            "outputs": outputs,
        },
    )
