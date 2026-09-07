"""Post-hoc ensemble-uncertainty gating with capped capital reallocation.

Everything here is a reduction over cached training artifacts (per-trial and
per-replicate test-position parquets). No model is retrained, no training,
feature, detector, or tensor cache is written or invalidated.

Policies (statistic x rule), all causal:

- ``ungated``     explicit ensemble-mean baseline.
- ``rank-hyst``   within-contract rank with two-state hysteresis: off below p/2, on at
  or above p.
- ``rank-cwhyst`` hysteresis participation with confidence-weighted
  reallocation: freed capital is redistributed across active contracts in
  proportion to their rank (warm-up days count 0.5), capped at m/N per
  contract; at m = 1 it reduces exactly to ``rank-hyst``.

Capital policy for binary rules: with ``N_t`` available contracts, ``A_t``
un-gated contracts and reallocation cap ``m``, the divisor is
``D_t = max(A_t, N_t / m)``; the residual weight is cash at zero return.
Costs act on post-division holdings through the per-ticker absolute holding
change on the concatenated calendar.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PRIMARY_WINDOWS, Profile
from .reporting import COST_GRID, _safe_name, performance_metrics
from .training import load_search_records
from .utils import atomic_csv, atomic_json, atomic_parquet, file_sha256, object_sha256

LOGGER = logging.getLogger(__name__)

DEFAULT_GATED_MODELS = (
    "LSTM",
    "LSTM-BOCPD",
    "LSTM-BOCPD-rich",
    "LSTM-BOCPD-sys",
    "LSTM-BOCPD-full",
)
DEFAULT_M_GRID = (1.0, 2.0, math.inf)
DEFAULT_P_GRID = (0.0, 0.10, 0.25, 0.50)
ENSEMBLE_KINDS = ("trial", "seed")
BURN_IN = 126
RANK_MIN_HISTORY = 63
MIN_ACTIVE_DAYS = 60
KAPPA_EPSILON = 1e-9

GATE_VARIANTS: dict[str, dict[str, object]] = {
    "ungated": {"statistic": "none", "rule": "none", "m_values": (1.0,)},
    "rank-hyst": {"statistic": "rank", "rule": "hyst", "m_values": DEFAULT_M_GRID},
    "rank-cwhyst": {"statistic": "rank", "rule": "cwhyst", "m_values": DEFAULT_M_GRID},
}
DEFAULT_VARIANTS = ("rank-hyst", "rank-cwhyst")


def _m_label(m: float) -> str:
    """Format a finite or infinite capital-cap multiplier for tables."""
    return "inf" if math.isinf(m) else f"{m:g}"


def _pivot(frame: pd.DataFrame, value: str) -> pd.DataFrame:
    """Pivot a long contract-level frame to a date-by-contract matrix."""
    return frame.pivot(index="date", columns="ticker", values=value).sort_index()


def member_tables(
    build_dir: Path,
    profile: Profile,
    model: str,
    year: int,
    kind: str,
    trial_members: int | None = None,
) -> dict[str, object]:
    """Member position matrices plus shared targets and leverage.

    ``trial_members`` overrides the trial-kind member count (default: top-5,
    or every trial when fewer than five exist); the seed kind always uses all
    replicates.

    Args:
        build_dir: Profile-specific cache directory.
        profile: Deep-momentum experiment profile.
        model: Gross model name.
        year: Test-window start year.
        kind: ``"trial"`` or ``"seed"`` ensemble source.
        trial_members: Optional number of validation-ranked trials to retain.

    Returns:
        Member positions, targets, leverage, axes, and source hashes.
    """
    directory = build_dir / "training" / f"w{year}" / model
    if kind == "trial":
        count = int(trial_members or min(5, profile.trials))
        records = [
            record
            for record in load_search_records(build_dir, year, model, profile.trials)
            if record["status"] == "success"
        ]
        if len(records) < count:
            raise RuntimeError(
                f"{model} w{year}: {len(records)} valid trials for a "
                f"{count}-member ensemble"
            )
        members = sorted(records, key=lambda record: -float(record["val_sharpe"]))[
            : count
        ]
        paths = [Path(record["prediction_path"]) for record in members]
        hashes = [str(record.get("prediction_sha256")) for record in members]
    elif kind == "seed":
        paths, hashes = [], []
        for replicate in range(profile.replicates):
            record_path = directory / f"replicate{replicate:03d}.json"
            record = json.loads(record_path.read_text())
            if record.get("status") != "success":
                raise RuntimeError(f"{model} w{year}: replicate {replicate} failed")
            paths.append(directory / f"positions_replicate{replicate:03d}.parquet")
            hashes.append(str(record.get("prediction_sha256")))
    else:
        raise KeyError(f"unknown ensemble kind: {kind}")

    frames = [pd.read_parquet(path) for path in paths]
    wide = [_pivot(frame, "position") for frame in frames]
    base = wide[0]
    for index, other in enumerate(wide[1:], start=1):
        if (
            not other.index.equals(base.index)
            or not other.columns.equals(base.columns)
            or not other.isna().equals(base.isna())
        ):
            raise RuntimeError(
                f"{model} w{year} {kind}: member {index} prediction grid differs"
            )
    sentinel = 8.9e307
    target = _pivot(frames[0], "target")
    if len(frames) > 1:
        check = _pivot(frames[1], "target")
        if not np.array_equal(
            target.fillna(sentinel).to_numpy(), check.fillna(sentinel).to_numpy()
        ):
            raise RuntimeError(f"{model} w{year} {kind}: member targets differ")
    return {
        "positions": np.stack([frame.to_numpy(dtype=float) for frame in wide]),
        "target": target,
        "lev": _pivot(frames[0], "lev"),
        "index": base.index,
        "columns": base.columns,
        "hashes": hashes,
    }


def ensemble_stats(positions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate ensemble mean, dispersion, and confidence ratio.

    Args:
        positions: Array with member as its first dimension.

    Returns:
        Member mean, sample standard deviation, and absolute mean-to-dispersion
        ratio; entries remain NaN where predictions are unavailable.
    """
    if positions.shape[0] < 2:
        raise ValueError("ensemble statistics require at least two members")
    mean = positions.mean(axis=0)
    standard_deviation = positions.std(axis=0, ddof=1)
    kappa = np.abs(mean) / (standard_deviation + KAPPA_EPSILON)
    return mean, standard_deviation, kappa



def within_contract_rank(
    kappa: pd.DataFrame, *, min_history: int = RANK_MIN_HISTORY
) -> pd.DataFrame:
    """Expanding rank of each contract's kappa against its own strict past.

    Returns the fraction of the contract's earlier finite kappa values lying
    below the current one (ties averaged), uniform on [0, 1] under exchange-
    ability. Days with fewer than ``min_history`` prior observations are NaN,
    which every rule treats as "trade ungated" (warm-up).

    Args:
        kappa: Date-by-contract confidence-ratio matrix.
        min_history: Prior finite observations required before assigning a rank.

    Returns:
        Strictly causal within-contract percentile ranks.
    """
    ranks = {}
    for column in kappa.columns:
        series = kappa[column]
        inclusive_rank = series.expanding().rank(method="average")
        prior_count = series.expanding().count() - 1
        score = (inclusive_rank - 1.0) / prior_count
        score[prior_count < min_history] = np.nan
        ranks[column] = score
    return pd.DataFrame(ranks)




def divisor(
    available: pd.Series | np.ndarray, active: pd.Series | np.ndarray, m: float
) -> np.ndarray:
    """Calculate the capped-reallocation capital divisor.

    Args:
        available: Number of available contracts on each date.
        active: Number of contracts retained by the gate on each date.
        m: Per-contract capital-cap multiplier, including infinity.

    Returns:
        Daily divisor ``max(active, available / m)``.
    """
    if not (m >= 1.0):
        raise ValueError("the reallocation cap m must be at least 1")
    available = np.asarray(available, dtype=float)
    active = np.asarray(active, dtype=float)
    floor = np.zeros_like(available) if math.isinf(m) else available / m
    return np.maximum(active, floor)




def hysteresis_mask(
    score: pd.DataFrame, on_level: float, off_level: float
) -> pd.DataFrame:
    """Two-state gate per contract: off when score < off_level, on when
    score >= on_level, held otherwise. NaN scores (warm-up) force on.

    Args:
        score: Date-by-contract causal confidence scores.
        on_level: Inclusive threshold that changes the state to active.
        off_level: Exclusive lower threshold that changes it to inactive.

    Returns:
        Boolean active-state matrix.
    """
    if not (0.0 <= off_level <= on_level):
        raise ValueError("hysteresis levels must satisfy 0 <= off <= on")
    values = score.to_numpy(dtype=float)
    states = np.ones_like(values, dtype=bool)
    for column in range(values.shape[1]):
        on = True
        for row in range(values.shape[0]):
            value = values[row, column]
            if np.isfinite(value):
                if value >= on_level:
                    on = True
                elif value < off_level:
                    on = False
            else:
                on = True
            states[row, column] = on
    return pd.DataFrame(states, index=score.index, columns=score.columns)



def _weights_for(
    variant: str,
    p: float,
    statistics: dict[str, pd.DataFrame],
    available: pd.DataFrame,
) -> tuple[pd.DataFrame, bool]:
    """Weight frame in [0, 1] (NaN off-panel) and whether the rule is binary."""
    settings = GATE_VARIANTS[variant]
    statistic, rule = settings["statistic"], settings["rule"]
    if p <= 0.0:
        return pd.DataFrame(
            1.0, index=available.index, columns=available.columns
        ).where(available), True
    if statistic == "rank":
        rank = statistics["rank"]
        if rule in ("hyst", "cwhyst"):
            weights = hysteresis_mask(rank, p, p / 2.0).astype(float)
        else:
            raise KeyError(f"unsupported rank rule: {rule}")
    else:
        raise KeyError(f"unknown gate statistic: {statistic}")
    return weights.where(available), True


def portfolio_from_weights(
    mean: pd.DataFrame,
    weights: pd.DataFrame,
    target: pd.DataFrame,
    lev: pd.DataFrame,
    m: float,
    *,
    binary: bool,
    cost_bps: float = 0.0,
) -> pd.DataFrame:
    """Daily portfolio under the capped-reallocation policy.

    Binary rules reallocate among active contracts (D_t = max(A_t, N_t/m));
    the soft rule keeps the divisor at the available count (no reallocation).

    Args:
        mean: Ensemble-mean positions.
        weights: Gating weights between zero and one.
        target: Volatility-scaled next-day contract returns.
        lev: Per-contract leverage used to calculate holdings and turnover.
        m: Per-contract capital-cap multiplier.
        binary: Whether ``weights`` represent binary participation.
        cost_bps: Transaction cost per unit of absolute turnover, in basis points.

    Returns:
        Daily gross/net return, turnover, availability, and activity counts.
    """
    available = target.notna()
    gated = (mean * weights).where(available)
    n_available = available.sum(axis=1).astype(float)
    if binary:
        n_active = (weights > 0).sum(axis=1).astype(float)
        day_divisor = pd.Series(divisor(n_available, n_active, m), index=mean.index)
    else:
        n_active = n_available
        day_divisor = n_available.copy()
    day_divisor = day_divisor.replace(0.0, np.nan)
    contribution = (gated * target).sum(axis=1, min_count=1).fillna(0.0)
    gross = (contribution / day_divisor).fillna(0.0)
    holdings = (gated * lev).div(day_divisor, axis=0)
    turnover = holdings.diff().abs().sum(axis=1, min_count=1).fillna(0.0)
    net = gross - float(cost_bps) * 1e-4 * turnover
    return pd.DataFrame(
        {
            "gross": gross,
            "net": net,
            "turnover": turnover,
            "n_available": n_available,
            "n_active": n_active,
        }
    )


def confidence_capital(
    mask: pd.DataFrame,
    rank: pd.DataFrame,
    available: pd.DataFrame,
    m: float,
) -> pd.DataFrame:
    """Per-contract capital shares under confidence-weighted reallocation.

    Every active contract keeps its baseline 1/N_t share; the capital freed
    by abstention, (N_t - A_t)/N_t, is redistributed across active contracts
    in proportion to their within-contract rank (warm-up days count as the
    neutral 0.5), subject to the per-contract cap m/N_t. Capital removed by
    the cap stays in cash (single pass, no re-redistribution). At m = 1 the
    cap binds everywhere and the allocation reduces to the equal-split
    idle-cash policy of the hysteresis rule.

    Args:
        mask: Boolean active-contract matrix.
        rank: Causal within-contract confidence ranks.
        available: Boolean availability matrix.
        m: Per-contract capital-cap multiplier.

    Returns:
        Date-by-contract allocated capital shares.
    """
    if not (m >= 1.0):
        raise ValueError("the reallocation cap m must be at least 1")
    n_available = available.sum(axis=1).astype(float).replace(0.0, np.nan)
    n_active = mask.sum(axis=1).astype(float)
    base = mask.astype(float).div(n_available, axis=0)
    confidence = rank.where(mask & available).fillna(0.5).where(mask & available)
    norm = confidence.sum(axis=1).replace(0.0, np.nan)
    freed = ((n_available - n_active) / n_available).clip(lower=0.0)
    extra = confidence.div(norm, axis=0).mul(freed, axis=0).fillna(0.0)
    capital = base + extra
    if not math.isinf(m):
        cap = pd.Series(m, index=capital.index) / n_available
        capital = capital.clip(upper=cap, axis=0)
    return capital.where(available)


def portfolio_from_capital(
    mean: pd.DataFrame,
    capital: pd.DataFrame,
    mask: pd.DataFrame,
    target: pd.DataFrame,
    lev: pd.DataFrame,
    *,
    cost_bps: float = 0.0,
) -> pd.DataFrame:
    """Construct a daily portfolio from explicit contract capital shares.

    Args:
        mean: Ensemble-mean positions.
        capital: Date-by-contract capital shares.
        mask: Boolean active-contract matrix.
        target: Volatility-scaled next-day contract returns.
        lev: Per-contract leverage used to calculate holdings and turnover.
        cost_bps: Transaction cost per unit of absolute turnover, in basis points.

    Returns:
        Daily gross/net return, turnover, availability, and activity counts.
    """
    available = target.notna()
    gated = (mean * capital).where(available)
    contribution = (gated * target).sum(axis=1, min_count=1).fillna(0.0)
    holdings = gated * lev
    turnover = holdings.diff().abs().sum(axis=1, min_count=1).fillna(0.0)
    net = contribution - float(cost_bps) * 1e-4 * turnover
    return pd.DataFrame(
        {
            "gross": contribution,
            "net": net,
            "turnover": turnover,
            "n_available": available.sum(axis=1).astype(float),
            "n_active": mask.sum(axis=1).astype(float),
        }
    )




def conditional_contract_sharpe(
    mean: pd.DataFrame,
    mask: pd.DataFrame,
    target: pd.DataFrame,
    *,
    lev: pd.DataFrame | None = None,
    cost_bps: float = 0.0,
    min_days: int = MIN_ACTIVE_DAYS,
) -> dict[str, float]:
    """Active-day-weighted mean of per-contract conditional Sharpe ratios.

    Net variants include exit costs on the first abstained day while retaining
    the active-day counts as the cross-contract aggregation weights.

    Args:
        mean: Ensemble-mean positions.
        mask: Boolean active-contract matrix.
        target: Volatility-scaled next-day contract returns.
        lev: Optional leverage matrix, required when ``cost_bps`` is nonzero.
        cost_bps: Transaction cost rate in basis points.
        min_days: Minimum active observations required for a contract estimate.

    Returns:
        Weighted conditional Sharpe and included/excluded contract counts.
    """
    if cost_bps and lev is None:
        raise ValueError("leverage is required for a net conditional Sharpe")
    sharpes, weights, excluded = [], [], 0
    for ticker in mean.columns:
        traded = mask[ticker]
        gross = mean[ticker] * target[ticker]
        active_days = int(gross[traded].dropna().shape[0])
        if active_days < min_days:
            excluded += 1
            continue
        if cost_bps:
            holding = (mean[ticker] * lev[ticker]).where(traded, 0.0)
            turnover = holding.diff().abs().fillna(0.0)
            included = traded | turnover.gt(0.0)
            captured = (gross.where(traded, 0.0) - cost_bps * 1e-4 * turnover)[
                included
            ].dropna()
        else:
            captured = gross[traded].dropna()
        deviation = float(captured.std(ddof=1))
        if not np.isfinite(deviation) or deviation <= 0:
            excluded += 1
            continue
        sharpes.append(float(captured.mean()) / deviation * math.sqrt(252.0))
        weights.append(float(active_days))
    if not sharpes:
        return {
            "conditional_sharpe": np.nan,
            "conditional_contracts": 0,
            "conditional_excluded": excluded,
        }
    return {
        "conditional_sharpe": float(np.average(sharpes, weights=weights)),
        "conditional_contracts": len(sharpes),
        "conditional_excluded": excluded,
    }


def _window_mean_metric(
    daily: pd.Series, windows: tuple[int, ...], span: int, metric: str
) -> float:
    """Calculate an equally weighted mean of a performance metric across windows."""
    values = [
        performance_metrics(daily.loc[f"{year}-01-01" : f"{year + span - 1}-12-31"])[
            metric
        ]
        for year in windows
    ]
    return float(np.nanmean(values))


def _assemble_model(
    build_dir: Path,
    profile: Profile,
    model: str,
    kind: str,
    trial_members: int | None = None,
) -> dict[str, object]:
    """Concatenate member statistics across the profile's windows."""
    means, kappas, targets, levs = [], [], [], []
    burn_flags, hashes = [], {}
    for year in profile.windows:
        tables = member_tables(
            build_dir, profile, model, year, kind, trial_members=trial_members
        )
        mean, _, kappa = ensemble_stats(tables["positions"])
        frame_kwargs = dict(index=tables["index"], columns=tables["columns"])
        means.append(pd.DataFrame(mean, **frame_kwargs))
        kappas.append(pd.DataFrame(kappa, **frame_kwargs))
        targets.append(tables["target"])
        levs.append(tables["lev"])
        flag = pd.Series(False, index=tables["index"])
        flag.iloc[:BURN_IN] = True
        burn_flags.append(flag)
        hashes[str(year)] = tables["hashes"]
    return {
        "mean": pd.concat(means).sort_index(),
        "kappa": pd.concat(kappas).sort_index(),
        "target": pd.concat(targets).sort_index(),
        "lev": pd.concat(levs).sort_index(),
        "burn": pd.concat(burn_flags).sort_index(),
        "hashes": hashes,
    }


def _per_window(frame: pd.DataFrame, profile: Profile, builder) -> pd.DataFrame:
    """Apply a per-window frame builder and concatenate on the full calendar."""
    pieces = []
    for year in profile.windows:
        window = frame.loc[
            f"{year}-01-01" : f"{year + profile.test_span - 1}-12-31"
        ]
        pieces.append(builder(window))
    return pd.concat(pieces).sort_index()





def run_gating(
    profile: Profile,
    build_dir: Path,
    output_dir: Path,
    *,
    models: tuple[str, ...] = DEFAULT_GATED_MODELS,
    m_grid: tuple[float, ...] | None = None,
    p_grid: tuple[float, ...] = DEFAULT_P_GRID,
    variants: tuple[str, ...] = DEFAULT_VARIANTS,
    trial_members: int | None = None,
) -> None:
    """Evaluate and persist the reported uncertainty-gating experiment.

    Args:
        profile: Deep-momentum experiment profile.
        build_dir: Profile-specific training cache directory.
        output_dir: Profile-specific numerical-output directory.
        models: Gross model names to gate.
        m_grid: Optional capital-cap grid replacing each variant's default.
        p_grid: Target abstention-rate thresholds.
        variants: Gating variants to evaluate in addition to the ungated case.
        trial_members: Optional number of validation-ranked trial members.
    """
    unknown = sorted(set(variants) - set(GATE_VARIANTS))
    if unknown:
        raise KeyError(f"unknown gate variants: {unknown}")
    gating_dir = output_dir / "gating"
    gating_dir.mkdir(parents=True, exist_ok=True)
    aggregate_windows = tuple(
        year for year in profile.windows if year in PRIMARY_WINDOWS
    ) or profile.windows

    assembled: dict[tuple[str, str], dict[str, object]] = {}
    for model in models:
        for kind in ENSEMBLE_KINDS:
            assembled[(model, kind)] = _assemble_model(
                build_dir, profile, model, kind, trial_members=trial_members
            )

    fingerprint = object_sha256(
        {
            "stage": "gating-v3",
            "profile": profile.to_dict(),
            "trial_members": int(trial_members or min(5, profile.trials)),
            "models": models,
            "kinds": ENSEMBLE_KINDS,
            "variants": ("ungated",) + variants,
            "m_override": [str(m) for m in m_grid] if m_grid else None,
            "p_grid": list(p_grid),
            "burn_in": BURN_IN,
            "rank_min_history": RANK_MIN_HISTORY,
            "min_active_days": MIN_ACTIVE_DAYS,
            "members": {
                f"{model}|{kind}": tables["hashes"]
                for (model, kind), tables in assembled.items()
            },
        }
    )
    metadata_path = gating_dir / "metadata.json"
    atomic_json(metadata_path, {"fingerprint": fingerprint, "status": "building"})

    summary_rows, window_rows, cost_rows, contract_rows = [], [], [], []
    daily_frames: dict[tuple[str, str], pd.DataFrame] = {}
    for (model, kind), tables in assembled.items():
        LOGGER.info("Gating %s [%s]", model, kind)
        mean, kappa = tables["mean"], tables["kappa"]
        target, lev, burn = tables["target"], tables["lev"], tables["burn"]
        available = kappa.notna()
        statistics: dict[str, object] = {
            "mean": mean,
            "kappa": kappa,
            "rank": _per_window(kappa, profile, within_contract_rank),
        }
        daily_records = []
        for variant in ("ungated",) + variants:
            rule = str(GATE_VARIANTS[variant]["rule"])
            variant_m = (
                (1.0,)
                if variant == "ungated"
                else tuple(m_grid or GATE_VARIANTS[variant]["m_values"])
            )
            variant_p = (0.0,) if variant == "ungated" else tuple(
                p for p in p_grid if p > 0.0
            )
            for p in variant_p:
                weights, binary = _weights_for(variant, p, statistics, available)
                scored = ~burn
                cells = available.loc[scored]
                withheld = (1.0 - weights.loc[scored]).where(cells)
                abstention = (
                    float(withheld.sum().sum() / cells.sum().sum())
                    if cells.sum().sum()
                    else np.nan
                )
                if binary:
                    mask = weights.fillna(0.0) > 0
                    conditional = conditional_contract_sharpe(mean, mask, target)
                    for conditional_cost in (2.0, 3.0):
                        net_conditional = conditional_contract_sharpe(
                            mean,
                            mask,
                            target,
                            lev=lev,
                            cost_bps=conditional_cost,
                        )
                        suffix = f"{conditional_cost:g}bp"
                        conditional[f"conditional_sharpe_{suffix}"] = net_conditional[
                            "conditional_sharpe"
                        ]
                        conditional[f"conditional_contracts_{suffix}"] = net_conditional[
                            "conditional_contracts"
                        ]
                        conditional[f"conditional_excluded_{suffix}"] = net_conditional[
                            "conditional_excluded"
                        ]
                else:
                    conditional = {
                        "conditional_sharpe": np.nan,
                        "conditional_contracts": 0,
                        "conditional_excluded": 0,
                    }
                per_contract = (1.0 - weights).where(available).mean()
                total_abstention = float(per_contract.sum())
                top5 = (
                    float(per_contract.nlargest(5).sum() / total_abstention)
                    if total_abstention > 0
                    else np.nan
                )
                contract_rows.append(
                    {
                        "model": model,
                        "ensemble": kind,
                        "variant": variant,
                        "p": p,
                        "mean_contract_abstention": float(per_contract.mean()),
                        "max_contract_abstention": float(per_contract.max()),
                        "top5_share": top5,
                    }
                )
                for m in variant_m:
                    if rule == "cwhyst":
                        mask = (weights.fillna(0.0) > 0) & available
                        capital = confidence_capital(
                            mask, statistics["rank"], available, m
                        )
                        daily = portfolio_from_capital(
                            mean, capital, mask, target, lev
                        )
                    else:
                        daily = portfolio_from_weights(
                            mean, weights, target, lev, m, binary=binary
                        )
                    row = {
                        "model": model,
                        "ensemble": kind,
                        "variant": variant,
                        "statistic": GATE_VARIANTS[variant]["statistic"],
                        "rule": rule,
                        "p": p,
                        "m": _m_label(m),
                        "sharpe": _window_mean_metric(
                            daily["gross"], aggregate_windows, profile.test_span,
                            "Sharpe",
                        ),
                        "mdd": _window_mean_metric(
                            daily["gross"], aggregate_windows, profile.test_span,
                            "MDD",
                        ),
                        "abstention_rate": abstention,
                        "mean_turnover": float(daily["turnover"].mean()),
                        **conditional,
                    }
                    summary_rows.append(row)
                    for year in profile.windows:
                        window_slice = daily["gross"].loc[
                            f"{year}-01-01" : f"{year + profile.test_span - 1}-12-31"
                        ]
                        window_rows.append(
                            {
                                "model": model,
                                "ensemble": kind,
                                "variant": variant,
                                "p": p,
                                "m": row["m"],
                                "window": year,
                                "sharpe": performance_metrics(window_slice)["Sharpe"],
                            }
                        )
                    for cost in COST_GRID:
                        net = daily["gross"] - cost * 1e-4 * daily["turnover"]
                        cost_rows.append(
                            {
                                "model": model,
                                "ensemble": kind,
                                "variant": variant,
                                "p": p,
                                "m": row["m"],
                                "cost_bps": cost,
                                "sharpe": _window_mean_metric(
                                    net, aggregate_windows, profile.test_span,
                                    "Sharpe",
                                ),
                            }
                        )
                    record = daily.reset_index(names="date")
                    record.insert(1, "variant", variant)
                    record.insert(2, "p", p)
                    record.insert(3, "m", row["m"])
                    daily_records.append(record)
        daily_frames[(model, kind)] = pd.concat(daily_records, ignore_index=True)

    summary = pd.DataFrame(summary_rows)
    atomic_csv(gating_dir / "gating_summary.csv", summary)
    atomic_csv(gating_dir / "gating_by_window.csv", pd.DataFrame(window_rows))
    atomic_csv(gating_dir / "gating_cost_sweep.csv", pd.DataFrame(cost_rows))
    atomic_csv(
        gating_dir / "gating_contract_abstention.csv", pd.DataFrame(contract_rows)
    )
    for (model, kind), frame in daily_frames.items():
        atomic_parquet(
            gating_dir / f"gating_daily_{_safe_name(model)}_{kind}.parquet", frame
        )
    output_paths = [
        gating_dir / "gating_summary.csv",
        gating_dir / "gating_by_window.csv",
        gating_dir / "gating_cost_sweep.csv",
        gating_dir / "gating_contract_abstention.csv",
        *[
            gating_dir / f"gating_daily_{_safe_name(model)}_{kind}.parquet"
            for model, kind in daily_frames
        ],
    ]
    atomic_json(
        metadata_path,
        {
            "fingerprint": fingerprint,
            "status": "complete",
            "models": list(models),
            "kinds": list(ENSEMBLE_KINDS),
            "variants": ["ungated", *variants],
            "trial_members": int(trial_members or min(5, profile.trials)),
            "p_grid": list(p_grid),
            "rows": len(summary),
            "outputs": {
                str(path.relative_to(gating_dir)): file_sha256(path)
                for path in sorted(output_paths)
            },
        },
    )
    LOGGER.info("Gating outputs written to %s", gating_dir)
