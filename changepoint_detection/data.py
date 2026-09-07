"""Canonical archive loading, return construction, and training priors."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import io
import json
import logging
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from experiment_data import STUDY_START, TICKERS, UNIVERSE

from .models import GarchFit, RegimePrior, fit_garch11, prior_from_fits
from .progress import ProgressReporter
from .storage import atomic_write_csv, atomic_write_json

LOGGER = logging.getLogger(__name__)

CLASS_ORDER = ("CM", "EQ", "FI", "FX")
TRAIN_END = pd.Timestamp("2006-12-31")
TEST_START = pd.Timestamp("2007-01-01")
STUDY_START_DATE = pd.Timestamp(STUDY_START)


@dataclass(frozen=True)
class TrainingInputs:
    """Prepared statistical inputs for the market BOCPD experiment.

    Attributes:
        returns: Study-period percentage log returns keyed by ticker.
        scales: Pre-2007 return standard deviations keyed by ticker.
        fits: Per-contract standardised GARCH(1,1) estimates.
        priors: Hierarchical regime priors keyed by asset class.
        archive_sha256: Hash of the prepared archive used to derive the inputs.
    """

    returns: dict[str, pd.Series]
    scales: dict[str, float]
    fits: dict[str, GarchFit]
    priors: dict[str, RegimePrior]
    archive_sha256: str


class TrainingInputCacheMismatch(ValueError):
    """Cached priors were not produced from the requested canonical archive."""


def file_sha256(path: Path) -> str:
    """Return the hexadecimal SHA-256 digest of a file.

    Args:
        path: File to hash.

    Returns:
        Lower-case hexadecimal digest.
    """
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_study_closes(archive_path: Path) -> dict[str, pd.Series]:
    """Load positive closes, applying the study boundary before any computation.

    This function is the sole archive-reading chokepoint for the package.  The
    unified archive contains earlier observations for indicator initialisation
    in other experiments; changepoint inputs begin at ``STUDY_START``.
    """
    archive_path = Path(archive_path)
    if not archive_path.exists():
        raise FileNotFoundError(archive_path)
    LOGGER.info("Loading study-period closes from %s", archive_path)
    result: dict[str, pd.Series] = {}
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        for ticker in TICKERS:
            member = f"CLCDATA/{ticker}_RAD.CSV"
            if member not in names:
                raise ValueError(f"canonical archive is missing {member}")
            frame = pd.read_csv(
                io.BytesIO(archive.read(member)),
                header=None,
                names=("date", "open", "high", "low", "close", "volume", "oi"),
                parse_dates=["date"],
                date_format="%m/%d/%Y",
            )
            if frame["date"].duplicated().any() or not frame["date"].is_monotonic_increasing:
                raise ValueError(f"{ticker}: dates must be unique and increasing")
            close = (
                frame.set_index("date")["close"]
                .astype(float)
                .loc[STUDY_START_DATE:]
            )
            if len(close) == 0 or close.index.min() < STUDY_START_DATE:
                raise ValueError(f"{ticker}: no valid study-period close series")
            if close.isna().any() or (~np.isfinite(close)).any() or (close <= 0).any():
                raise ValueError(f"{ticker}: close series contains invalid values")
            result[ticker] = close.rename(ticker)
    if set(result) != set(TICKERS):
        raise RuntimeError("close panel does not match the configured universe")
    LOGGER.info(
        "Loaded study-period closes for %d contracts (boundary=%s)",
        len(result),
        STUDY_START,
    )
    return result


def load_returns(archive_path: Path) -> dict[str, pd.Series]:
    """Calculate per-contract study-period log returns in percent.

    Args:
        archive_path: Prepared continuous-futures ZIP archive.

    Returns:
        Percentage log-return series keyed by ticker.
    """
    closes = _load_study_closes(archive_path)
    result: dict[str, pd.Series] = {}
    for ticker, close in closes.items():
        returns = 100.0 * np.log(close).diff().dropna()
        if len(returns) == 0 or (~np.isfinite(returns)).any():
            raise ValueError(f"{ticker}: invalid return series")
        result[ticker] = returns.rename(ticker)
    LOGGER.info("Calculated study-period returns for %d contracts", len(result))
    return result


def build_training_inputs(
    archive_path: Path,
    *,
    starts: int,
    seed: int,
    min_observations: int = 1000,
) -> TrainingInputs:
    """Fit pre-2007 GARCH models and construct asset-class regime priors.

    Args:
        archive_path: Prepared continuous-futures ZIP archive.
        starts: Independent optimisation starts for each initial GARCH fit.
        seed: Root seed used to derive deterministic per-contract fit seeds.
        min_observations: Minimum number of pre-2007 returns required per contract.

    Returns:
        Returns, scales, fitted GARCH parameters, priors, and archive hash.
    """
    returns = load_returns(archive_path)
    scales: dict[str, float] = {}
    fits: dict[str, GarchFit] = {}
    progress = ProgressReporter("GARCH QMLE fits", len(TICKERS), logger=LOGGER)
    for offset, ticker in enumerate(TICKERS):
        training = returns[ticker].loc[:TRAIN_END].to_numpy(dtype=float)
        if len(training) < min_observations:
            raise ValueError(
                f"{ticker}: {len(training)} pre-2007 observations; "
                f"at least {min_observations} required"
            )
        scale = float(np.std(training, ddof=0))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError(f"{ticker}: invalid training scale")
        scales[ticker] = scale
        fit_seed = seed + offset
        candidates: list[GarchFit] = []
        try:
            candidates.append(
                fit_garch11(training / scale, starts=starts, seed=fit_seed)
            )
        except RuntimeError:
            pass
        if not candidates or not candidates[0].converged:
            try:
                candidates.append(
                    fit_garch11(
                        training / scale,
                        starts=3 * starts,
                        seed=fit_seed + 1_000_003,
                    )
                )
            except RuntimeError:
                pass
        usable = [fit for fit in candidates if fit.has_finite_interior_solution]
        if not usable:
            raise RuntimeError(f"{ticker}: GARCH QMLE produced no usable solution")
        fits[ticker] = min(
            usable, key=lambda candidate: candidate.negative_log_likelihood
        )
        progress.update(offset + 1, detail=ticker)
    priors = {
        asset_class: prior_from_fits(
            fits[ticker]
            for ticker in TICKERS
            if UNIVERSE[ticker][0] == asset_class
        )
        for asset_class in CLASS_ORDER
    }
    nonconverged = sum(not fit.converged for fit in fits.values())
    LOGGER.info(
        "GARCH QMLE summary: %d usable fits; %d with a non-converged optimizer flag",
        len(fits),
        nonconverged,
    )
    return TrainingInputs(returns, scales, fits, priors, file_sha256(archive_path))


def save_training_inputs(inputs: TrainingInputs, output_dir: Path) -> None:
    """Persist fitted GARCH parameters and priors for deterministic reuse.

    Args:
        inputs: Statistical inputs to persist.
        output_dir: Directory receiving CSV and JSON metadata.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    fit_rows = []
    for ticker, fit in inputs.fits.items():
        fit_rows.append(
            {
                "ticker": ticker,
                "asset_class": UNIVERSE[ticker][0],
                "training_scale": inputs.scales[ticker],
                **fit.to_dict(),
            }
        )
    atomic_write_csv(pd.DataFrame(fit_rows), output_dir / "garch_fits.csv")
    payload = {
        "train_end": str(TRAIN_END.date()),
        "archive_sha256": inputs.archive_sha256,
        "scales": inputs.scales,
        "priors": {key: prior.to_dict() for key, prior in inputs.priors.items()},
    }
    atomic_write_json(output_dir / "priors.json", payload)
    LOGGER.info("Saved GARCH fits and priors to %s", output_dir)


def load_training_inputs(archive_path: Path, output_dir: Path) -> TrainingInputs:
    """Load cached fit metadata while deriving returns from the archive.

    Args:
        archive_path: Prepared archive whose hash must match the cache metadata.
        output_dir: Directory containing ``priors.json`` and ``garch_fits.csv``.

    Returns:
        Reconstructed statistical inputs.

    Raises:
        FileNotFoundError: If either cache file is absent.
        TrainingInputCacheMismatch: If the cache refers to another archive.
    """
    prior_path = output_dir / "priors.json"
    fit_path = output_dir / "garch_fits.csv"
    if not prior_path.exists() or not fit_path.exists():
        raise FileNotFoundError("cached training inputs are incomplete")
    payload = json.loads(prior_path.read_text())
    actual_archive_sha256 = file_sha256(archive_path)
    cached_archive_sha256 = payload.get("archive_sha256")
    if cached_archive_sha256 != actual_archive_sha256:
        reason = (
            "missing archive hash"
            if cached_archive_sha256 is None
            else "archive hash mismatch"
        )
        raise TrainingInputCacheMismatch(f"cached training inputs: {reason}")
    fit_frame = pd.read_csv(fit_path).set_index("ticker")
    fits = {
        ticker: GarchFit(
            mean=float(row["mean"]),
            omega=float(row["omega"]),
            alpha=float(row["alpha"]),
            beta=float(row["beta"]),
            hbar=float(row["hbar"]),
            persistence=float(row["persistence"]),
            negative_log_likelihood=float(row["negative_log_likelihood"]),
            converged=bool(row["converged"]),
            iterations=int(row["iterations"]),
        )
        for ticker, row in fit_frame.iterrows()
    }
    inputs = TrainingInputs(
        returns=load_returns(archive_path),
        scales={ticker: float(value) for ticker, value in payload["scales"].items()},
        fits=fits,
        priors={
            asset_class: RegimePrior.from_dict(values)
            for asset_class, values in payload["priors"].items()
        },
        archive_sha256=actual_archive_sha256,
    )
    LOGGER.info("Loaded cached GARCH fits and priors from %s", output_dir)
    return inputs
