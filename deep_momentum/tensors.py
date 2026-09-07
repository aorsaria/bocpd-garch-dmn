"""Per-window tensors shared by models with the same feature family."""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from experiment_data import STUDY_START, UNIVERSE

from .config import (
    FEATURE_FAMILIES,
    MODEL_SPECS,
    SEQUENCE_LENGTH,
    ModelSpec,
    Profile,
)
from .data import backtest_start, selected_tickers
from .detectors import load_bocpd_frame, load_cusum_frame
from .features import load_base_frame, prepare_base_features
from .utils import (
    assert_cache_fingerprint,
    atomic_json,
    atomic_npz,
    file_sha256,
    object_sha256,
)


LOGGER = logging.getLogger(__name__)
SPEC_BY_FAMILY = {spec.family: spec for spec in MODEL_SPECS}


def _common_window_tickers(
    tickers: tuple[str, ...], calibration: dict[str, object]
) -> tuple[str, ...]:
    """Intersect requested contracts with the window's calibrated universe."""
    eligible = set(calibration["eligible_tickers"])
    return tuple(ticker for ticker in tickers if ticker in eligible)


def _blocks(
    features: np.ndarray,
    targets: np.ndarray,
    leverage: np.ndarray,
    sequence_length: int = SEQUENCE_LENGTH,
) -> tuple[np.ndarray, np.ndarray]:
    """Pad a contract history into fixed-length training sequences.

    Args:
        features: Observation-by-feature input matrix.
        targets: Volatility-scaled next-day returns.
        leverage: Contract leverage aligned with ``targets``.
        sequence_length: Number of observations in each padded block.

    Returns:
        Feature blocks and target/mask/leverage blocks.
    """
    count, width = features.shape
    blocks = int(np.ceil(count / sequence_length))
    padding = blocks * sequence_length - count
    x = np.pad(features, ((0, padding), (0, 0))).reshape(
        blocks, sequence_length, width
    )
    target = np.pad(targets, (0, padding)).reshape(blocks, sequence_length, 1)
    mask = np.pad(np.ones(count), (0, padding)).reshape(blocks, sequence_length, 1)
    lev = np.pad(leverage, (0, padding)).reshape(blocks, sequence_length, 1)
    y = np.concatenate((target, mask, lev), axis=-1)
    return x.astype("float32"), y.astype("float32")


def _systemic_frame(build_dir: Path, year: int, tickers: tuple[str, ...]) -> pd.DataFrame:
    """Aggregate contract BOCPD probabilities into systemic daily features."""
    young = {}
    for ticker in tickers:
        path = build_dir / "detectors" / f"w{year}" / "bocpd" / f"{ticker}.parquet"
        if path.exists():
            young[ticker] = load_bocpd_frame(build_dir, year, ticker)["bocpd_score"]
    panel = pd.DataFrame(young).sort_index()
    active = panel.notna().sum(axis=1)
    return pd.DataFrame(
        {
            "sys_score": panel.mean(axis=1),
            "sys_frac": panel.gt(0.5).sum(axis=1) / active.replace(0, np.nan),
        }
    )


def assemble_feature_frames(
    family: str,
    year: int,
    tickers: tuple[str, ...],
    base_root: Path,
    build_dir: Path,
) -> dict[str, pd.DataFrame]:
    """Assemble per-contract frames for one model feature family.

    Args:
        family: Registered feature-family identifier.
        year: Test-window start year.
        tickers: Contracts eligible for this window.
        base_root: Directory containing cached base features.
        build_dir: Profile-specific detector cache directory.

    Returns:
        Joined feature frames keyed by ticker.
    """
    systemic = (
        _systemic_frame(build_dir, year, tickers)
        if family in {"bocpd_sys", "bocpd_full"}
        else None
    )
    output = {}
    for ticker in tickers:
        frame = load_base_frame(base_root, ticker).drop(columns=["ticker"], errors="ignore")
        if family.startswith("bocpd"):
            detector_path = (
                build_dir / "detectors" / f"w{year}" / "bocpd" / f"{ticker}.parquet"
            )
            if not detector_path.exists():
                continue
            frame = frame.join(load_bocpd_frame(build_dir, year, ticker), how="inner")
            if systemic is not None:
                frame = frame.join(systemic, how="left")
        elif family == "cusum":
            detector_path = (
                build_dir / "detectors" / f"w{year}" / "cusum" / f"{ticker}.parquet"
            )
            if not detector_path.exists():
                continue
            frame = frame.join(load_cusum_frame(build_dir, year, ticker), how="inner")
        elif family != "base":
            raise KeyError(f"unknown feature family: {family}")
        output[ticker] = frame
    return output


def build_window_tensors(
    frames: dict[str, pd.DataFrame],
    spec: ModelSpec,
    year: int,
    profile: Profile,
) -> dict[str, object]:
    """Construct chronological training, validation, and test tensors.

    Args:
        frames: Per-contract feature and target frames.
        spec: Model definition specifying ordered feature columns.
        year: Test-window start year.
        profile: Experiment profile defining dates and sequence policy.

    Returns:
        Arrays used by training, validation aggregation, and test reporting.
    """
    train_x, train_y, valid_x, valid_y = [], [], [], []
    valid_dates: list[pd.DatetimeIndex] = []
    test_x, test_date, test_signal_date = [], [], []
    test_ticker, test_target, test_leverage = [], [], []
    eligible = []
    train_end = pd.Timestamp(f"{year - 1}-12-31")
    test_end = pd.Timestamp(f"{year + profile.test_span - 1}-12-31")
    columns = list(spec.feature_columns)
    prepared: dict[str, tuple[pd.DataFrame, int]] = {}
    for ticker, raw in frames.items():
        if not raw.empty and raw.index.min() < pd.Timestamp(STUDY_START):
            raise ValueError(f"{ticker}: pre-study row entered tensor construction")
        frame = raw.replace([np.inf, -np.inf], np.nan).dropna(
            subset=columns + ["target", "target_date", "lev"]
        )
        training_start = max(
            pd.Timestamp(profile.data_start), pd.Timestamp(STUDY_START)
        )
        training = frame.loc[training_start:train_end]
        training = training.loc[training["target_date"] <= train_end]
        split = int(0.9 * len(training))
        if len(training) - split < SEQUENCE_LENGTH:
            continue
        eligible.append(ticker)
        prepared[ticker] = (frame, split)
        xa, ya = _blocks(
            training[columns].to_numpy(dtype=float)[:split],
            training["target"].to_numpy(dtype=float)[:split],
            training["lev"].to_numpy(dtype=float)[:split],
        )
        xb, yb = _blocks(
            training[columns].to_numpy(dtype=float)[split:],
            training["target"].to_numpy(dtype=float)[split:],
            training["lev"].to_numpy(dtype=float)[split:],
        )
        train_x.append(xa)
        train_y.append(ya)
        valid_x.append(xb)
        valid_y.append(yb)
        valid_dates.append(pd.DatetimeIndex(training["target_date"].iloc[split:]))

    if not eligible:
        raise ValueError(f"{spec.family} window {year}: no eligible contracts")
    all_validation_dates = sorted(
        set().union(*(set(index) for index in valid_dates))
    )
    date_code = {date: index for index, date in enumerate(all_validation_dates)}
    valid_codes = []
    for dates, target_blocks in zip(valid_dates, valid_y):
        codes = np.zeros(target_blocks.shape[0] * target_blocks.shape[1], dtype=np.int64)
        codes[: len(dates)] = [date_code[date] for date in dates]
        valid_codes.append(codes)

    for ticker in eligible:
        frame, _ = prepared[ticker]
        index = frame.index
        target_dates = pd.DatetimeIndex(frame["target_date"])
        start = backtest_start(ticker, year)
        locations = np.flatnonzero(
            (target_dates >= start) & (target_dates <= test_end)
        )
        locations = locations[locations >= SEQUENCE_LENGTH - 1]
        if len(locations) == 0:
            continue
        values = frame[columns].to_numpy(dtype="float32")
        windows = np.stack(
            [values[position - SEQUENCE_LENGTH + 1 : position + 1] for position in locations]
        )
        test_x.append(windows)
        test_date.extend(target_dates[locations].strftime("%Y-%m-%d").tolist())
        test_signal_date.extend(index[locations].strftime("%Y-%m-%d").tolist())
        test_ticker.extend([ticker] * len(locations))
        test_target.extend(frame["target"].to_numpy(dtype=float)[locations].tolist())
        test_leverage.extend(frame["lev"].to_numpy(dtype=float)[locations].tolist())
    if not test_x:
        raise ValueError(f"{spec.family} window {year}: no test observations")
    return {
        "x_train": np.concatenate(train_x),
        "y_train": np.concatenate(train_y),
        "x_valid": np.concatenate(valid_x),
        "y_valid": np.concatenate(valid_y),
        "valid_date_codes": np.concatenate(valid_codes),
        "valid_date_count": np.asarray([max(1, len(all_validation_dates))]),
        "x_test": np.concatenate(test_x),
        "test_date": np.asarray(test_date),
        "test_signal_date": np.asarray(test_signal_date),
        "test_ticker": np.asarray(test_ticker),
        "test_target": np.asarray(test_target, dtype="float64"),
        "test_leverage": np.asarray(test_leverage, dtype="float64"),
        "eligible_tickers": np.asarray(eligible),
        "feature_columns": np.asarray(columns),
    }


def prepare_tensor_caches(
    archive: Path,
    profile: Profile,
    build_dir: Path,
    *,
    workers: int,
) -> Path:
    """Build or reuse all fingerprinted feature-family tensor caches.

    Args:
        archive: Prepared continuous-futures ZIP archive.
        profile: Deep-momentum experiment profile.
        build_dir: Profile-specific cache directory.
        workers: Maximum number of feature jobs used by prerequisite stages.

    Returns:
        Root directory of per-window NPZ tensor caches.
    """
    base_root = prepare_base_features(
        archive, profile, build_dir, workers=workers
    )
    tensor_root = build_dir / "tensors"
    base_fingerprint = json.loads((base_root / "metadata.json").read_text())[
        "fingerprint"
    ]
    detector_metadata = {
        year: json.loads(
            (build_dir / "detectors" / f"w{year}" / "features.json").read_text()
        )
        for year in profile.windows
    }
    calibration_metadata = {
        year: json.loads(
            (build_dir / "detectors" / f"w{year}" / "calibration.json").read_text()
        )
        for year in profile.windows
    }
    input_fingerprint = object_sha256(
        {
            "stage": "model-tensors-v5",
            "archive": file_sha256(archive),
            "profile": profile.to_dict(),
            "families": FEATURE_FAMILIES,
            "sequence_length": SEQUENCE_LENGTH,
            "base_features": base_fingerprint,
            "detector_features": {
                year: metadata["fingerprint"]
                for year, metadata in detector_metadata.items()
            },
        }
    )
    tickers = selected_tickers(profile.tickers)
    for year in profile.windows:
        window_tickers = _common_window_tickers(
            tickers, calibration_metadata[year]
        )
        base_eligible: tuple[str, ...] | None = None
        for family in FEATURE_FAMILIES:
            path = tensor_root / f"w{year}" / f"{family}.npz"
            metadata_path = path.with_suffix(".json")
            fingerprint = object_sha256(
                {"inputs": input_fingerprint, "window": year, "family": family}
            )
            if assert_cache_fingerprint(metadata_path, fingerprint) and path.exists():
                with np.load(path, allow_pickle=False) as cached:
                    eligible = tuple(cached["eligible_tickers"].tolist())
                if family == "base":
                    base_eligible = eligible
                    if profile.full_scale and eligible != window_tickers:
                        raise RuntimeError(
                            f"window {year}: cached LSTM eligibility {eligible} "
                            f"differs from detector eligibility {window_tickers}"
                        )
                elif profile.full_scale and base_eligible is not None and eligible != base_eligible:
                    raise RuntimeError(
                        f"window {year}: cached {family} eligibility differs from base"
                    )
                continue
            LOGGER.info("Building tensors: window=%d family=%s", year, family)
            frames = assemble_feature_frames(
                family, year, window_tickers, base_root, build_dir
            )
            arrays = build_window_tensors(frames, SPEC_BY_FAMILY[family], year, profile)
            eligible = tuple(arrays["eligible_tickers"].tolist())
            if family == "base":
                base_eligible = eligible
                if profile.full_scale and eligible != window_tickers:
                    raise RuntimeError(
                        f"window {year}: LSTM eligibility {eligible} differs from "
                        f"detector eligibility {window_tickers}"
                    )
            elif profile.full_scale and base_eligible is not None and eligible != base_eligible:
                raise RuntimeError(
                    f"window {year}: {family} eligibility {eligible} differs from "
                    f"base eligibility {base_eligible}"
                )
            atomic_npz(path, **arrays)
            atomic_json(
                metadata_path,
                {
                    "fingerprint": fingerprint,
                    "window": year,
                    "family": family,
                    "feature_columns": arrays["feature_columns"].tolist(),
                    "eligible_tickers": list(eligible),
                    "train_sequences": int(arrays["x_train"].shape[0]),
                    "validation_sequences": int(arrays["x_valid"].shape[0]),
                    "test_rows": int(arrays["x_test"].shape[0]),
                },
            )
    atomic_json(
        tensor_root / "metadata.json",
        {
            "fingerprint": input_fingerprint,
            "archive_sha256": file_sha256(archive),
            "windows": list(profile.windows),
            "families": list(FEATURE_FAMILIES),
        },
    )
    return tensor_root


def load_tensors(path: Path) -> dict[str, np.ndarray]:
    """Load every array from a tensor NPZ.

    Args:
        path: Tensor-cache NPZ file.

    Returns:
        Arrays keyed by their stored names; pickled values are not permitted.
    """
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}
