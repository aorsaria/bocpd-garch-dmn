"""Reusable causal price-feature cache for the deep-momentum models."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from changepoint_detection.progress import map_jobs
from experiment_data import STUDY_START

from .config import Profile
from .data import (
    build_base_features,
    load_closes,
    selected_tickers,
)
from .utils import (
    assert_cache_fingerprint,
    atomic_json,
    atomic_parquet,
    file_sha256,
    object_sha256,
)


LOGGER = logging.getLogger(__name__)


def _base_job(arguments):
    """Construct and persist study-period base features for one contract."""
    ticker, close, start, history_start, output = arguments
    series = close.loc[history_start or start :]
    # Indicators are initialised on the warm-up rows; the stored frame keeps
    # the study coverage only, so no training row can predate `start`.
    frame = build_base_features(series).loc[start:].reset_index(names="date")
    frame.insert(1, "ticker", ticker)
    atomic_parquet(Path(output), frame)
    return ticker, len(frame)


def prepare_base_features(
    archive: Path,
    profile: Profile,
    build_dir: Path,
    *,
    workers: int,
) -> Path:
    """Build or reuse the fingerprinted base-feature cache.

    Args:
        archive: Prepared continuous-futures ZIP archive.
        profile: Deep-momentum experiment profile.
        build_dir: Profile-specific cache directory.
        workers: Maximum number of contract feature jobs.

    Returns:
        Directory containing one base-feature Parquet file per contract.
    """
    root = build_dir / "features" / "base"
    metadata_path = root / "metadata.json"
    tickers = selected_tickers(profile.tickers)
    archive_sha256 = file_sha256(archive)
    recipe: dict[str, object] = {
        "stage": "base-features-v2",
        "archive": archive_sha256,
        "tickers": tickers,
        "data_start": profile.data_start,
    }
    if profile.history_start is not None:
        recipe["history"] = {
            "start": profile.history_start,
            "archive": archive_sha256,
        }
    fingerprint = object_sha256(recipe)
    metadata_existed = metadata_path.exists()
    complete = assert_cache_fingerprint(metadata_path, fingerprint)
    if complete:
        metadata = json.loads(metadata_path.read_text())
        complete = (
            metadata.get("status", "complete") == "complete"
            and set(metadata.get("rows", ())) == set(tickers)
        )
    expected = [root / f"{ticker}.parquet" for ticker in tickers]
    if complete and all(path.exists() for path in expected):
        LOGGER.info("Base feature cache is complete (%d contracts)", len(tickers))
        return root
    rebuild_all = not metadata_existed and any(path.exists() for path in expected)
    if rebuild_all:
        LOGGER.warning(
            "Base feature files exist without metadata; rebuilding all %d contracts",
            len(tickers),
        )
    atomic_json(
        metadata_path,
        {
            "fingerprint": fingerprint,
            "status": "building",
            "archive_sha256": archive_sha256,
            "tickers": list(tickers),
            "data_start": profile.data_start,
            "history_start": profile.history_start,
            "history_sha256": archive_sha256,
        },
    )
    closes = load_closes(archive, tickers)
    if profile.history_start is not None:
        warmup_count = sum(
            bool((close.index < pd.Timestamp(STUDY_START)).any())
            for close in closes.values()
        )
        LOGGER.info(
            "Warm-up history active from %s (%d of %d contracts have history rows)",
            profile.history_start,
            warmup_count,
            len(tickers),
        )
    jobs = [
        (
            ticker,
            closes[ticker],
            profile.data_start,
            profile.history_start,
            str(root / f"{ticker}.parquet"),
        )
        for ticker in tickers
        if rebuild_all or not (root / f"{ticker}.parquet").exists()
    ]
    map_jobs(
        _base_job, jobs, workers, label="Deep-momentum base features", logger=LOGGER
    )
    rows = {
        ticker: len(pd.read_parquet(root / f"{ticker}.parquet", columns=["date"]))
        for ticker in tickers
    }
    atomic_json(
        metadata_path,
        {
            "fingerprint": fingerprint,
            "status": "complete",
            "archive_sha256": archive_sha256,
            "tickers": list(tickers),
            "history_start": profile.history_start,
            "history_sha256": archive_sha256,
            "rows": rows,
        },
    )
    return root


def load_base_frame(root: Path, ticker: str) -> pd.DataFrame:
    """Load one cached base-feature frame.

    Args:
        root: Base-feature cache directory.
        ticker: Contract identifier.

    Returns:
        Cached frame indexed by date.
    """
    return pd.read_parquet(root / f"{ticker}.parquet").set_index("date")
