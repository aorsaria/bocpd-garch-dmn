"""Paired circular-block bootstrap for uncertainty-gating Sharpe deltas."""

from __future__ import annotations

import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Profile
from .gating import DEFAULT_GATED_MODELS
from .reporting import _safe_name
from .utils import atomic_csv, atomic_json, file_sha256, object_sha256


LOGGER = logging.getLogger(__name__)
BLOCK_LENGTH = 21
BOOTSTRAP_SEED = 20260828
BOOTSTRAP_COSTS = (0.0, 2.0, 3.0)
BOOTSTRAP_COLUMNS = (
    "kind",
    "policy",
    "cost_bps",
    "model",
    "delta",
    "p_value",
    "block_length",
    "B",
    "seed",
)
POLICIES = {
    "hysteresis": ("rank-hyst", 0.5, "1"),
    "confidence": ("rank-cwhyst", 0.5, "inf"),
}


def _sharpe(values: np.ndarray) -> float:
    """Calculate annualised Sharpe ratio without an annualised mean adjustment."""
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return np.nan
    deviation = float(np.std(values, ddof=1))
    if not np.isfinite(deviation) or deviation <= 0.0:
        return np.nan
    return float(np.mean(values) / deviation * np.sqrt(252.0))


def _circular_indices(
    length: int, block_length: int, rng: np.random.Generator
) -> np.ndarray:
    """Draw one circular block-bootstrap index vector.

    Args:
        length: Required output length and size of the sampled series.
        block_length: Number of consecutive positions in each circular block.
        rng: NumPy random generator.

    Returns:
        Bootstrap positions of length ``length``.
    """
    if length < 1 or block_length < 1:
        raise ValueError("series and block lengths must be positive")
    blocks = int(math.ceil(length / block_length))
    starts = rng.integers(0, length, size=blocks)
    offsets = np.arange(block_length)
    return ((starts[:, None] + offsets[None, :]) % length).ravel()[:length]


def _window_arrays(
    gated: pd.Series,
    ungated: pd.Series,
    profile: Profile,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Align gated and ungated returns and split them by test window."""
    aligned = pd.concat(
        [gated.rename("gated"), ungated.rename("ungated")], axis=1, join="inner"
    ).dropna()
    if aligned.empty:
        raise ValueError("gated and ungated series have no common observations")
    output: list[tuple[np.ndarray, np.ndarray]] = []
    for year in profile.windows:
        block = aligned.loc[
            f"{year}-01-01" : f"{year + profile.test_span - 1}-12-31"
        ]
        if len(block) < 2:
            raise ValueError(f"window {year} has fewer than two aligned observations")
        output.append(
            (
                block["gated"].to_numpy(dtype=float),
                block["ungated"].to_numpy(dtype=float),
            )
        )
    return output


def window_mean_sharpe_difference(
    gated: pd.Series, ungated: pd.Series, profile: Profile
) -> float:
    """Difference between equally weighted per-window Sharpe means.

    Args:
        gated: Daily gated-strategy returns.
        ungated: Daily reference-strategy returns.
        profile: Profile defining test windows and their span.

    Returns:
        Mean window-level gated-minus-ungated Sharpe difference.
    """
    arrays = _window_arrays(gated, ungated, profile)
    differences = [_sharpe(left) - _sharpe(right) for left, right in arrays]
    if not np.isfinite(differences).all():
        raise ValueError("non-finite window Sharpe in bootstrap input")
    return float(np.mean(differences))


def paired_circular_bootstrap(
    gated: pd.Series,
    ungated: pd.Series,
    profile: Profile,
    *,
    samples: int,
    block_length: int = BLOCK_LENGTH,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float, np.ndarray]:
    """Apply the paired circular-block bootstrap to a Sharpe difference.

    Args:
        gated: Daily gated-strategy returns.
        ungated: Daily reference-strategy returns on the same calendar.
        profile: Profile defining test windows and their span.
        samples: Number of bootstrap replications.
        block_length: Circular-block length in observations.
        seed: Deterministic bootstrap seed.

    Returns:
        Observed difference, one-sided p-value, and bootstrap differences.
    """
    if samples < 1:
        raise ValueError("bootstrap sample count must be positive")
    arrays = _window_arrays(gated, ungated, profile)
    observed = float(
        np.mean([_sharpe(left) - _sharpe(right) for left, right in arrays])
    )
    if not np.isfinite(observed):
        raise ValueError("observed bootstrap statistic is non-finite")
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=float)
    for replicate in range(samples):
        window_deltas = []
        for left, right in arrays:
            indices = _circular_indices(len(left), block_length, rng)
            window_deltas.append(_sharpe(left[indices]) - _sharpe(right[indices]))
        draws[replicate] = float(np.mean(window_deltas))
    if not np.isfinite(draws).all():
        raise ValueError("bootstrap produced a non-finite statistic")
    return observed, float(np.mean(draws <= 0.0)), draws


def _policy_frame(frame: pd.DataFrame, variant: str, p: float, m: str) -> pd.DataFrame:
    """Select one gating policy from a long-form daily-results frame."""
    labels = frame["m"].astype(str).replace({"1.0": "1", "inf": "inf"})
    selected = frame[
        (frame["variant"] == variant)
        & np.isclose(frame["p"].astype(float), p)
        & (labels == m)
    ].copy()
    if selected.empty:
        raise ValueError(f"missing gating policy {variant}, p={p}, m={m}")
    selected["date"] = pd.to_datetime(selected["date"])
    if selected["date"].duplicated().any():
        raise ValueError(f"duplicate dates for gating policy {variant}, p={p}, m={m}")
    return selected.set_index("date").sort_index()


def _load_model_policy(
    gating_dir: Path, model: str, kind: str, policy: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load aligned gated and ungated daily results for one model policy."""
    path = gating_dir / f"gating_daily_{_safe_name(model)}_{kind}.parquet"
    frame = pd.read_parquet(path)
    variant, p, m = POLICIES[policy]
    gated = _policy_frame(frame, variant, p, m)
    ungated = _policy_frame(frame, "ungated", 0.0, "1")
    common = gated.index.intersection(ungated.index)
    if len(common) != len(gated) or len(common) != len(ungated):
        raise ValueError(f"{model} {kind}: gated and ungated calendars differ")
    return gated.loc[common], ungated.loc[common]


def run_bootstrap(
    profile: Profile,
    output_dir: Path,
    *,
    models: tuple[str, ...] = DEFAULT_GATED_MODELS,
) -> pd.DataFrame:
    """Evaluate all pre-specified gating policies, costs, and ensembles.

    Args:
        profile: Deep-momentum profile providing windows and replication count.
        output_dir: Profile-specific numerical-output directory.
        models: Gross model names included individually and in the composite.

    Returns:
        Bootstrap statistics for every ensemble kind, policy, cost, and model.
    """
    gating_dir = output_dir / "gating"
    samples = int(profile.bootstrap_samples)
    rows: list[dict[str, object]] = []
    for kind in ("trial", "seed"):
        for policy in POLICIES:
            loaded = {
                model: _load_model_policy(gating_dir, model, kind, policy)
                for model in models
            }
            for cost in BOOTSTRAP_COSTS:
                series: dict[str, tuple[pd.Series, pd.Series]] = {}
                for model, (gated, ungated) in loaded.items():
                    series[model] = (
                        gated["gross"] - cost * 1e-4 * gated["turnover"],
                        ungated["gross"] - cost * 1e-4 * ungated["turnover"],
                    )
                gated_composite = pd.concat(
                    [pair[0].rename(model) for model, pair in series.items()], axis=1
                ).mean(axis=1)
                ungated_composite = pd.concat(
                    [pair[1].rename(model) for model, pair in series.items()], axis=1
                ).mean(axis=1)
                series["COMPOSITE"] = (gated_composite, ungated_composite)
                for model, (gated_net, ungated_net) in series.items():
                    delta, p_value, _ = paired_circular_bootstrap(
                        gated_net,
                        ungated_net,
                        profile,
                        samples=samples,
                        block_length=BLOCK_LENGTH,
                        seed=BOOTSTRAP_SEED,
                    )
                    rows.append(
                        {
                            "kind": kind,
                            "policy": policy,
                            "cost_bps": cost,
                            "model": model,
                            "delta": delta,
                            "p_value": p_value,
                            "block_length": BLOCK_LENGTH,
                            "B": samples,
                            "seed": BOOTSTRAP_SEED,
                        }
                    )
                    LOGGER.info(
                        "Bootstrap %s/%s/%g bps/%s: delta=%.6f p=%.6f",
                        kind,
                        policy,
                        cost,
                        model,
                        delta,
                        p_value,
                    )
    output = pd.DataFrame(rows, columns=BOOTSTRAP_COLUMNS)
    path = gating_dir / "bootstrap.csv"
    atomic_csv(path, output)
    atomic_json(
        gating_dir / "bootstrap_metadata.json",
        {
            "fingerprint": object_sha256(
                {
                    "stage": "gating-bootstrap-v1",
                    "profile": profile.to_dict(),
                    "models": models,
                    "policies": POLICIES,
                    "costs": BOOTSTRAP_COSTS,
                    "block_length": BLOCK_LENGTH,
                    "samples": samples,
                    "seed": BOOTSTRAP_SEED,
                }
            ),
            "rows": len(output),
            "output_sha256": file_sha256(path),
        },
    )
    return output


__all__ = [
    "BLOCK_LENGTH",
    "BOOTSTRAP_COLUMNS",
    "BOOTSTRAP_SEED",
    "_circular_indices",
    "paired_circular_bootstrap",
    "run_bootstrap",
    "window_mean_sharpe_difference",
]
