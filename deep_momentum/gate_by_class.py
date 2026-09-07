"""Per-asset-class and per-window breakdown of the uncertainty gate.

For each model and asset class this reports, under the trial ensemble:
abstention rate, per-contract conditional Sharpe (ungated vs hysteresis
gate at p = 0.50), the class-sleeve portfolio Sharpe (class contracts
only, divided by the class's available count, freed capital idle), and
the mean capital share the confidence-weighted refinement assigns to the
class relative to its baseline share. Read-only over the training caches;
writes two deterministic CSV outputs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from experiment_data import CLASS_ORDER, UNIVERSE

from .config import Profile
from .gating import (
    DEFAULT_GATED_MODELS,
    MIN_ACTIVE_DAYS,
    _assemble_model,
    _per_window,
    confidence_capital,
    hysteresis_mask,
    within_contract_rank,
)
from .utils import atomic_csv, atomic_json, file_sha256, object_sha256

P_TARGET = 0.5
COSTS = (0.0, 2.0, 3.0)


def _sleeve(
    mean: pd.DataFrame,
    capital: pd.DataFrame,
    mask: pd.DataFrame,
    target: pd.DataFrame,
    lev: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """Calculate class-sleeve gross return and turnover.

    Args:
        mean: Ensemble-mean positions for the class contracts.
        capital: Explicit contract capital shares.
        mask: Boolean active-contract matrix.
        target: Volatility-scaled next-day contract returns.
        lev: Per-contract leverage used to calculate holdings.

    Returns:
        Daily gross return and absolute turnover series.
    """
    available = target.notna()
    gated = (mean * capital).where(available)
    gross = (gated * target).sum(axis=1, min_count=1).fillna(0.0)
    holdings = gated * lev
    turnover = holdings.diff().abs().sum(axis=1, min_count=1).fillna(0.0)
    return gross, turnover


def _sharpe(series: pd.Series) -> float:
    """Calculate annualised Sharpe ratio for a daily return series."""
    series = series.dropna()
    if len(series) < 2 or series.std(ddof=1) == 0:
        return np.nan
    return float(series.mean() / series.std(ddof=1) * np.sqrt(252.0))


def run_gate_by_class(
    profile: Profile,
    build_dir: Path,
    output_dir: Path,
    *,
    gate_members: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute asset-class diagnostics from cached member positions.

    Args:
        profile: Deep-momentum experiment profile.
        build_dir: Profile-specific training cache directory.
        output_dir: Profile-specific numerical-output directory.
        gate_members: Number of validation-ranked trial members, or the default.

    Returns:
        Aggregate asset-class summary and window-level class results.
    """
    member_count = int(gate_members or min(5, profile.trials))
    rows = []
    window_rows = []
    for model in DEFAULT_GATED_MODELS:
        tables = _assemble_model(
            build_dir, profile, model, "trial", trial_members=member_count
        )
        mean, kappa = tables["mean"], tables["kappa"]
        target, lev = tables["target"], tables["lev"]
        available = kappa.notna()
        rank = _per_window(kappa, profile, within_contract_rank)
        mask = hysteresis_mask(rank, P_TARGET, P_TARGET / 2.0) & available
        capital = confidence_capital(mask, rank, available, np.inf)
        captured = (mean * target).where(available)
        base_share = available.astype(float).div(
            available.sum(axis=1).replace(0, np.nan), axis=0
        )
        classes = tuple(
            asset_class
            for asset_class in CLASS_ORDER
            if any(UNIVERSE[t][0] == asset_class for t in mean.columns)
        )
        for asset_class in classes:
            tickers = [t for t in mean.columns if UNIVERSE[t][0] == asset_class]
            avail_c = available[tickers]
            mask_c = mask[tickers]
            cap_c = captured[tickers]
            n_avail = avail_c.sum(axis=1).replace(0, np.nan)
            abstention = float(
                (avail_c & ~mask_c).sum().sum() / avail_c.sum().sum()
            )
            # per-contract conditional Sharpe, active-day weighted
            def conditional(select: pd.DataFrame) -> float:
                """Aggregate active-day conditional Sharpe within this class."""
                values, weights = [], []
                for t in tickers:
                    series = cap_c[t][select[t]].dropna()
                    if len(series) >= MIN_ACTIVE_DAYS:
                        values.append(_sharpe(series))
                        weights.append(len(series))
                if not values:
                    return np.nan
                return float(np.average(values, weights=weights))

            sleeve_ungated = _sharpe(cap_c.sum(axis=1, min_count=1) / n_avail)
            sleeve_gated = _sharpe(
                cap_c.where(mask_c).sum(axis=1, min_count=1).fillna(0.0) / n_avail
            )
            share_base = float(base_share[tickers].sum(axis=1).mean())
            share_cw = float(capital[tickers].sum(axis=1, min_count=1).mean())
            # per-window class sleeves under class-internal capital policies
            n_avail_c = avail_c.sum(axis=1).replace(0, np.nan)
            policies = {
                "ungated": avail_c.astype(float).div(n_avail_c, axis=0),
                "hyst": mask_c.astype(float).div(n_avail_c, axis=0),
                "confidence": confidence_capital(
                    mask_c, rank[tickers], avail_c, np.inf
                ),
            }
            sleeves = {
                name: _sleeve(
                    mean[tickers], shares, mask_c if name != "ungated" else avail_c,
                    target[tickers], lev[tickers],
                )
                for name, shares in policies.items()
            }
            for year in profile.windows:
                span = slice(f"{year}-01-01", f"{year + profile.test_span - 1}-12-31")
                for name, (gross, turnover) in sleeves.items():
                    for cost in COSTS:
                        net = gross - cost * 1e-4 * turnover
                        window_rows.append(
                            {
                                "model": model,
                                "asset_class": asset_class,
                                "policy": name,
                                "window": year,
                                "cost_bps": cost,
                                "sharpe": _sharpe(net.loc[span]),
                            }
                        )
            rows.append(
                {
                    "model": model,
                    "asset_class": asset_class,
                    "contracts": len(tickers),
                    "abstention": abstention,
                    "cond_ungated": conditional(avail_c),
                    "cond_gated": conditional(mask_c),
                    "sleeve_ungated": sleeve_ungated,
                    "sleeve_hyst": sleeve_gated,
                    "share_baseline": share_base,
                    "share_confidence": share_cw,
                }
            )
    frame = pd.DataFrame(rows)
    gating_dir = output_dir / "gating"
    window_frame = pd.DataFrame(window_rows)
    atomic_csv(gating_dir / "gate_by_class.csv", frame)
    atomic_csv(gating_dir / "gate_by_class_windows.csv", window_frame)
    atomic_json(
        gating_dir / "gate_by_class_metadata.json",
        {
            "fingerprint": object_sha256(
                {
                    "stage": "gate-by-class-v1",
                    "profile": profile.to_dict(),
                    "gate_members": member_count,
                    "p": P_TARGET,
                    "costs": COSTS,
                }
            ),
            "gate_members": member_count,
            "rows": len(frame),
            "window_rows": len(window_frame),
            "outputs": {
                "gate_by_class.csv": file_sha256(gating_dir / "gate_by_class.csv"),
                "gate_by_class_windows.csv": file_sha256(
                    gating_dir / "gate_by_class_windows.csv"
                ),
            },
        },
    )
    return frame, window_frame
