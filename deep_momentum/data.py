"""Canonical archive loading and causal deep-momentum price features."""

from __future__ import annotations

import io
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd

from experiment_data import (
    CSV_COLUMNS,
    END_DATE,
    HISTORY_START,
    STUDY_START,
    TICKERS,
    UNIVERSE,
)

from .config import MACD_PAIRS, RETURN_HORIZONS, VOLATILITY_TARGET


def selected_tickers(profile_tickers: tuple[str, ...]) -> tuple[str, ...]:
    """Validate a requested contract subset and preserve canonical ordering.

    Args:
        profile_tickers: Requested identifiers, or an empty tuple for all 50.

    Returns:
        Selected contract identifiers in canonical universe order.
    """
    if not profile_tickers:
        return tuple(TICKERS)
    unknown = set(profile_tickers) - set(TICKERS)
    if unknown:
        raise ValueError(f"unknown profile tickers: {sorted(unknown)}")
    return tuple(ticker for ticker in TICKERS if ticker in profile_tickers)


def load_closes(
    archive_path: Path, tickers: tuple[str, ...] | None = None
) -> dict[str, pd.Series]:
    """Load prepared close histories through the package's archive reader.

    The returned series include pre-study rows when available.  Callers must
    explicitly apply ``STUDY_START`` before detector, estimation, or reporting
    computations; only rolling indicator construction may retain warm-up rows.

    Args:
        archive_path: Prepared continuous-futures ZIP archive.
        tickers: Requested contracts, or ``None`` for the full universe.

    Returns:
        Positive finite close-price series keyed by ticker.
    """
    archive_path = Path(archive_path)
    if not archive_path.exists():
        raise FileNotFoundError(archive_path)
    wanted = tuple(tickers or TICKERS)
    result: dict[str, pd.Series] = {}
    with zipfile.ZipFile(archive_path) as archive:
        names = set(archive.namelist())
        for ticker in wanted:
            member = f"CLCDATA/{ticker}_RAD.CSV"
            if member not in names:
                raise ValueError(f"canonical archive is missing {member}")
            frame = pd.read_csv(
                io.BytesIO(archive.read(member)),
                header=None,
                names=CSV_COLUMNS,
                parse_dates=["date"],
                date_format="%m/%d/%Y",
            )
            if frame["date"].duplicated().any() or not frame["date"].is_monotonic_increasing:
                raise ValueError(f"{ticker}: dates must be unique and increasing")
            close = frame.set_index("date")["close"].astype(float).loc[
                HISTORY_START:END_DATE
            ]
            if close.empty or close.isna().any() or (~np.isfinite(close)).any():
                raise ValueError(f"{ticker}: close series contains non-finite values")
            if (close <= 0).any():
                raise ValueError(f"{ticker}: close series contains non-positive values")
            result[ticker] = close.rename(ticker)
    return result


def study_closes(closes: dict[str, pd.Series]) -> dict[str, pd.Series]:
    """Return study-period views suitable for all non-indicator consumers.

    Args:
        closes: Close histories that may include pre-study warm-up rows.

    Returns:
        Views beginning no earlier than the configured study boundary.
    """
    boundary = pd.Timestamp(STUDY_START)
    output = {ticker: close.loc[boundary:] for ticker, close in closes.items()}
    for ticker, close in output.items():
        if close.empty or close.index.min() < boundary:
            raise ValueError(f"{ticker}: invalid study-period close boundary")
    return output


def log_returns_percent(close: pd.Series) -> pd.Series:
    """Calculate percentage log returns.

    Args:
        close: Positive close-price series.

    Returns:
        Percentage log returns with the first missing difference removed.
    """
    return (100.0 * np.log(close).diff()).dropna().rename(close.name)


def winsorize_price(price: pd.Series, *, half_life: int = 252, width: float = 5.0) -> pd.Series:
    """Apply causal exponentially weighted price clipping.

    Args:
        price: Chronologically ordered price series.
        half_life: Half-life for expanding exponentially weighted moments.
        width: Number of conditional standard deviations in each clipping bound.

    Returns:
        Causally clipped price series.
    """
    moments = price.ewm(halflife=half_life, min_periods=2, adjust=True)
    mean, scale = moments.mean(), moments.std()
    return price.clip(mean - width * scale, mean + width * scale)


def _macd_zscores(price: pd.Series) -> pd.DataFrame:
    """Calculate the three volatility-normalised MACD signals."""
    columns: dict[str, pd.Series] = {}
    for short, long in MACD_PAIRS:
        short_hl = np.log(0.5) / np.log(1.0 - 1.0 / short)
        long_hl = np.log(0.5) / np.log(1.0 - 1.0 / long)
        difference = price.ewm(halflife=short_hl).mean() - price.ewm(
            halflife=long_hl
        ).mean()
        price_scale = price.rolling(63, min_periods=63).std()
        quotient = difference / price_scale
        quotient_scale = quotient.rolling(252, min_periods=252).std()
        columns[f"macd_{short}_{long}"] = quotient / quotient_scale
    return pd.DataFrame(columns)


def build_base_features(close: pd.Series) -> pd.DataFrame:
    """Build inputs at t and the next-observation volatility-scaled target.

    Args:
        close: Positive close-price history, including warm-up rows when available.

    Returns:
        Base return/MACD features, leverage, target, and target date.
    """
    price = winsorize_price(close)
    daily_return = price.pct_change(fill_method=None)
    daily_volatility = daily_return.ewm(span=60, min_periods=60).std()
    annualised_volatility = daily_volatility * np.sqrt(252.0)
    frame = pd.DataFrame(
        {
            "daily_return": daily_return,
            "daily_volatility": daily_volatility,
            "lev": VOLATILITY_TARGET / annualised_volatility,
        }
    )
    frame["target"] = daily_return.shift(-1) * frame["lev"]
    frame["target_date"] = pd.Series(price.index, index=price.index).shift(-1)
    for horizon in RETURN_HORIZONS:
        frame[f"norm_ret_{horizon}"] = price.pct_change(
            horizon, fill_method=None
        ) / (daily_volatility * np.sqrt(float(horizon)))
    frame = frame.join(_macd_zscores(price))
    return frame.replace([np.inf, -np.inf], np.nan)


def backtest_start(ticker: str, window: int) -> pd.Timestamp:
    """Return a contract's effective out-of-sample start.

    Args:
        ticker: Contract identifier.
        window: Nominal test-window start year.

    Returns:
        Later of the window start and the contract's eligibility year.
    """
    return pd.Timestamp(f"{max(window, int(UNIVERSE[ticker][1]))}-01-01")
