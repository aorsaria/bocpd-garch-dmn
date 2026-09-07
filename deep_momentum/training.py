"""Deterministic, resumable random search and seed replication."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import logging
import multiprocessing as mp
import os
from pathlib import Path
import platform
import socket
import time
import traceback

import numpy as np
import pandas as pd

from changepoint_detection.progress import ProgressReporter

from .config import MODEL_BY_NAME, MODEL_NAMES, Profile
from .tensors import load_tensors
from .utils import atomic_json, atomic_parquet, file_sha256, object_sha256, stable_seed


LOGGER = logging.getLogger(__name__)
HP_GRID = {
    "hidden": (5, 10, 20, 40, 80, 160),
    "dropout": (0.1, 0.2, 0.3, 0.4, 0.5),
    "batch_size": (64, 128, 256),
    "lr": (1e-4, 1e-3, 1e-2, 1e-1),
    "clipnorm": (1e-2, 1.0, 1e2),
}


def draw_hyperparameters(seed: int) -> dict[str, float | int]:
    """Draw one random-search configuration from the fixed grid.

    Args:
        seed: Deterministic configuration seed.

    Returns:
        Selected value for every hyperparameter in ``HP_GRID``.
    """
    rng = np.random.default_rng(seed)
    return {
        name: values[int(rng.integers(0, len(values)))]
        for name, values in HP_GRID.items()
    }


def trial_seed(profile: Profile, year: int, model: str, trial: int) -> int:
    """Derive a search-trial training seed.

    Args:
        profile: Experiment profile providing the root seed.
        year: Test-window start year.
        model: Registered model identifier.
        trial: Zero-based search-trial index.

    Returns:
        Stable positive integer seed.
    """
    return stable_seed(profile.seed, "training", year, model, trial)


def replicate_seed(profile: Profile, year: int, model: str, replicate: int) -> int:
    """Derive a selected-configuration replication seed.

    Args:
        profile: Experiment profile providing the root seed.
        year: Test-window start year.
        model: Registered model identifier.
        replicate: Zero-based replication index.

    Returns:
        Stable positive integer seed.
    """
    return stable_seed(profile.seed, "replicate", year, model, replicate)


def _worker_environment() -> None:
    """Configure the Keras backend and single-threaded numerical workers."""
    os.environ.setdefault("KERAS_BACKEND", "torch")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"


def _environment_record() -> dict[str, object]:
    """Collect runtime versions and host metadata for a training record."""
    import keras
    import torch

    return {
        "backend": keras.backend.backend(),
        "keras": keras.__version__,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "python": platform.python_version(),
        "machine": platform.machine(),
        "host": socket.gethostname(),
    }


def _train_once(
    tensor_path: Path,
    model_name: str,
    hyperparameters: dict[str, object],
    seed: int,
    max_epochs: int,
    patience: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Train one LSTM and return validation metadata and test positions.

    Args:
        tensor_path: NPZ cache for the model's window and feature family.
        model_name: Registered model identifier.
        hyperparameters: Hidden size, dropout, batch size, learning rate, and
            gradient clipping norm.
        seed: Deterministic fit seed.
        max_epochs: Maximum training epochs.
        patience: Validation-Sharpe early-stopping patience.

    Returns:
        Successful fit metadata and long-form test predictions.
    """
    _worker_environment()
    import keras
    import torch

    from .network import DiversifiedValidationSharpe, build_lstm

    keras.utils.set_random_seed(int(seed))
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)
    arrays = load_tensors(tensor_path)
    spec = MODEL_BY_NAME[model_name]
    model = build_lstm(arrays["x_train"].shape[-1], hyperparameters, spec.cost_bps)
    callback = DiversifiedValidationSharpe(
        arrays["x_valid"],
        arrays["y_valid"],
        arrays["valid_date_codes"],
        int(arrays["valid_date_count"][0]),
        patience=patience,
        cost_bps=spec.cost_bps,
    )
    model.fit(
        arrays["x_train"],
        arrays["y_train"],
        batch_size=int(hyperparameters["batch_size"]),
        epochs=max_epochs,
        shuffle=True,
        verbose=0,
        callbacks=(callback, keras.callbacks.TerminateOnNaN()),
    )
    if not np.isfinite(callback.best):
        raise FloatingPointError("training produced no finite validation Sharpe")
    positions = model.predict(arrays["x_test"], batch_size=512, verbose=0)[:, -1, 0]
    predictions = pd.DataFrame(
        {
            "date": pd.to_datetime(arrays["test_date"]),
            "signal_date": pd.to_datetime(arrays["test_signal_date"]),
            "ticker": arrays["test_ticker"],
            "target": arrays["test_target"],
            "lev": arrays["test_leverage"],
            "position": positions,
        }
    )
    predictions["captured_return"] = predictions["position"] * predictions["target"]
    result = {
        "status": "success",
        "hp": hyperparameters,
        "seed": int(seed),
        "val_sharpe": float(callback.best),
        "best_epoch": callback.best_epoch,
        "epochs_run": len(callback.history),
        "environment": _environment_record(),
    }
    return result, predictions


def _training_job(arguments):
    """Execute one fault-isolated training job and persist its result record."""
    (
        tensor_path,
        model_name,
        year,
        index,
        mode,
        hyperparameters,
        seed,
        max_epochs,
        patience,
        result_path,
        prediction_path,
        fingerprint,
    ) = arguments
    started = time.monotonic()
    try:
        result, predictions = _train_once(
            Path(tensor_path),
            model_name,
            hyperparameters,
            seed,
            max_epochs,
            patience,
        )
        atomic_parquet(Path(prediction_path), predictions)
        result["prediction_sha256"] = file_sha256(Path(prediction_path))
    except Exception as error:
        result = {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "traceback": traceback.format_exc(limit=12),
            "hp": hyperparameters,
            "seed": int(seed),
        }
    result.update(
        {
            "fingerprint": fingerprint,
            "model": model_name,
            "window": year,
            "mode": mode,
            "index": index,
            "wall_seconds": round(time.monotonic() - started, 3),
            "tensor_sha256": file_sha256(Path(tensor_path)),
        }
    )
    atomic_json(Path(result_path), result)
    return year, model_name, index, mode, result["status"], result.get("val_sharpe")


def _execute(jobs: list[tuple], workers: int, label: str) -> list[tuple]:
    """Execute training jobs sequentially or in fresh spawned processes."""
    if not jobs:
        LOGGER.info("%s: all jobs cached", label)
        return []
    reporter = ProgressReporter(label, len(jobs), logger=LOGGER)
    if workers <= 1:
        results = []
        for completed, job in enumerate(jobs, start=1):
            result = _training_job(job)
            results.append(result)
            reporter.update(completed, detail=f"w{result[0]} {result[1]} #{result[2]}")
        return results
    context = mp.get_context("spawn")
    results = []
    with ProcessPoolExecutor(
        max_workers=min(workers, len(jobs)),
        mp_context=context,
        max_tasks_per_child=1,
    ) as executor:
        futures = {executor.submit(_training_job, job): job for job in jobs}
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            reporter.update(completed, detail=f"w{result[0]} {result[1]} #{result[2]}")
    return results


def _cached_record(
    path: Path, prediction_path: Path, fingerprint: str
) -> dict[str, object] | None:
    """Validate and return a cached training record when it is reusable."""
    if not path.exists():
        return None
    record = json.loads(path.read_text())
    if record.get("fingerprint") != fingerprint:
        raise RuntimeError(f"stale training cache at {path}")
    if record.get("status") == "success":
        if not prediction_path.exists() or "prediction_sha256" not in record:
            return None
        if file_sha256(prediction_path) != record["prediction_sha256"]:
            raise RuntimeError(f"corrupt prediction cache at {prediction_path}")
    return record


def train_search(profile: Profile, build_dir: Path, *, workers: int) -> None:
    """Run or resume every random-search model/window trial.

    Args:
        profile: Deep-momentum experiment profile.
        build_dir: Profile-specific tensor and training cache directory.
        workers: Maximum concurrent training processes.
    """
    jobs = []
    for year in profile.windows:
        for model_name in MODEL_NAMES:
            spec = MODEL_BY_NAME[model_name]
            tensor_path = build_dir / "tensors" / f"w{year}" / f"{spec.family}.npz"
            tensor_hash = file_sha256(tensor_path)
            directory = build_dir / "training" / f"w{year}" / model_name
            for trial in range(profile.trials):
                seed = trial_seed(profile, year, model_name, trial)
                hyperparameters = draw_hyperparameters(seed)
                fingerprint = object_sha256(
                    {
                        "stage": "search-trial-v2",
                        "tensor": tensor_hash,
                        "model": model_name,
                        "cost_bps": spec.cost_bps,
                        "trial": trial,
                        "seed": seed,
                        "hp": hyperparameters,
                        "max_epochs": profile.max_epochs,
                        "patience": profile.patience,
                    }
                )
                result_path = directory / f"trial{trial:03d}.json"
                prediction_path = directory / f"positions_trial{trial:03d}.parquet"
                if (
                    _cached_record(result_path, prediction_path, fingerprint)
                    is not None
                ):
                    continue
                jobs.append(
                    (
                        str(tensor_path),
                        model_name,
                        year,
                        trial,
                        "search",
                        hyperparameters,
                        seed,
                        profile.max_epochs,
                        profile.patience,
                        str(result_path),
                        str(prediction_path),
                        fingerprint,
                    )
                )
    _execute(jobs, workers, "Deep-momentum random-search trials")
    for year in profile.windows:
        for model_name in MODEL_NAMES:
            records = load_search_records(build_dir, year, model_name, profile.trials)
            valid = [record for record in records if record["status"] == "success"]
            if len(valid) < profile.ensemble:
                raise RuntimeError(
                    f"{model_name} window {year}: {len(valid)} valid trials; "
                    f"at least {profile.ensemble} required"
                )


def load_search_records(
    build_dir: Path, year: int, model_name: str, trials: int | None = None
) -> list[dict[str, object]]:
    """Load search records for one model/window in trial order.

    Args:
        build_dir: Profile-specific training cache directory.
        year: Test-window start year.
        model_name: Registered model identifier.
        trials: Optional exclusive upper bound on trial indices.

    Returns:
        Cached trial records with resolved result and prediction paths.
    """
    directory = build_dir / "training" / f"w{year}" / model_name
    records = []
    for path in sorted(directory.glob("trial*.json")):
        record = json.loads(path.read_text())
        if trials is not None and int(record["index"]) >= trials:
            continue
        record["result_path"] = str(path)
        record["prediction_path"] = str(
            directory / f"positions_trial{int(record['index']):03d}.parquet"
        )
        records.append(record)
    return records


def train_replicates(profile: Profile, build_dir: Path, *, workers: int) -> None:
    """Retrain each selected configuration with the prescribed seed replications.

    Args:
        profile: Deep-momentum experiment profile.
        build_dir: Profile-specific tensor and training cache directory.
        workers: Maximum concurrent training processes.
    """
    jobs = []
    for year in profile.windows:
        for model_name in MODEL_NAMES:
            records = [
                record
                for record in load_search_records(build_dir, year, model_name, profile.trials)
                if record["status"] == "success"
            ]
            if not records:
                raise RuntimeError(f"{model_name} window {year}: search results are missing")
            best = max(records, key=lambda record: float(record["val_sharpe"]))
            spec = MODEL_BY_NAME[model_name]
            tensor_path = build_dir / "tensors" / f"w{year}" / f"{spec.family}.npz"
            tensor_hash = file_sha256(tensor_path)
            directory = build_dir / "training" / f"w{year}" / model_name
            for replicate in range(profile.replicates):
                seed = replicate_seed(profile, year, model_name, replicate)
                fingerprint = object_sha256(
                    {
                        "stage": "seed-replicate-v2",
                        "tensor": tensor_hash,
                        "model": model_name,
                        "cost_bps": spec.cost_bps,
                        "replicate": replicate,
                        "seed": seed,
                        "hp": best["hp"],
                        "selected_trial": best["index"],
                        "max_epochs": profile.max_epochs,
                        "patience": profile.patience,
                    }
                )
                result_path = directory / f"replicate{replicate:03d}.json"
                prediction_path = directory / f"positions_replicate{replicate:03d}.parquet"
                if (
                    _cached_record(result_path, prediction_path, fingerprint)
                    is not None
                ):
                    continue
                jobs.append(
                    (
                        str(tensor_path),
                        model_name,
                        year,
                        replicate,
                        "replicate",
                        best["hp"],
                        seed,
                        profile.max_epochs,
                        profile.patience,
                        str(result_path),
                        str(prediction_path),
                        fingerprint,
                    )
                )
    _execute(jobs, workers, "Deep-momentum seed replications")
    for year in profile.windows:
        for model_name in MODEL_NAMES:
            directory = build_dir / "training" / f"w{year}" / model_name
            records = [json.loads(path.read_text()) for path in directory.glob("replicate*.json")]
            success = [record for record in records if record["status"] == "success"]
            if len(success) != profile.replicates:
                raise RuntimeError(
                    f"{model_name} window {year}: expected {profile.replicates} "
                    f"successful replicates, found {len(success)}"
                )
