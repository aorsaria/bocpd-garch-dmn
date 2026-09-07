"""Structural validation and cryptographic manifests for experiment outputs."""

from __future__ import annotations

from datetime import datetime, timezone
import logging
from pathlib import Path
import platform
import subprocess

import numpy as np
import pandas as pd

from .config import ExperimentConfig
from .data import file_sha256
from .storage import atomic_write_json

LOGGER = logging.getLogger(__name__)

REQUIRED_OUTPUTS = (
    "garch_fits.csv",
    "priors.json",
    "nu_selection.csv",
    "nu_selection_classes.csv",
    "nu_selection_contracts.csv",
    "nu_selected.json",
    "simulation_arl0.csv",
    "simulation_single_change.csv",
    "simulation_continuous.csv",
    "synthetic_thresholds.json",
    "continuous_example.npz",
    "event_thresholds.csv",
    "event_alarms.csv",
    "event_summary.csv",
    "event_summary_combined.csv",
    "covid_by_class.csv",
    "event_awareness.csv",
    "daily_detector_features.parquet",
)

def validate_outputs(config: ExperimentConfig, output_dir: Path) -> dict[str, object]:
    """Validate the presence, schemas, and numerical invariants of BOCPD outputs.

    Args:
        config: Profile used to produce the outputs.
        output_dir: Directory containing completed experiment files.

    Returns:
        Compact validation summary suitable for a manifest.
    """
    missing = [name for name in REQUIRED_OUTPUTS if not (output_dir / name).exists()]
    if missing:
        raise FileNotFoundError("missing changepoint outputs: " + ", ".join(missing))
    features = pd.read_parquet(output_dir / "daily_detector_features.parquet")
    expected = {
        "date",
        "ticker",
        "asset_class",
        "detector",
        "cp_prob",
        "young_prob",
        "expected_run_length",
        "run_length_variance",
        "expected_variance",
        "expected_volatility",
        "predictive_log_likelihood",
        "standardised_return",
    }
    if set(features.columns) != expected:
        raise ValueError("daily detector feature schema is not stable")
    numeric = features.select_dtypes(include=[np.number])
    if not np.isfinite(numeric.to_numpy()).all():
        raise ValueError("daily detector features contain non-finite values")
    tolerance = 1e-12
    if not features["cp_prob"].between(-tolerance, 1 + tolerance).all() or not features[
        "young_prob"
    ].between(-tolerance, 1 + tolerance).all():
        raise ValueError("posterior probabilities lie outside [0, 1]")
    if (features["cp_prob"] > features["young_prob"] + tolerance).any():
        raise ValueError("changepoint probability exceeds young-regime probability")

    with np.load(output_dir / "continuous_example.npz", allow_pickle=False) as example:
        required_arrays = {
            "values",
            "variances",
            "changepoints",
            "stream_index",
            "stream_seed",
            "monitoring_streams",
            "detection_window",
            "selection_criterion_met",
            "selection_rule",
            *(f"alarms_{name}" for name in ("CUSUM", "CUSUM-var", "CUSUM-mv", "EWMA", "BOCPD", "BOCPD-GARCH")),
        }
        if set(example.files) != required_arrays:
            raise ValueError("continuous-monitoring example schema is not stable")

    validation = {
        "profile": config.name,
        "full_scale": config.full_scale,
        "structural_validation_passed": True,
        "feature_rows": len(features),
        "tickers": int(features["ticker"].nunique()),
    }
    LOGGER.info(
        "Structural validation passed: %d feature rows across %d contracts",
        validation["feature_rows"],
        validation["tickers"],
    )
    return validation


def write_manifest(
    config: ExperimentConfig,
    archive: Path,
    output_dir: Path,
    validation: dict[str, object],
) -> None:
    """Write provenance, configuration, validation, and output hashes.

    Args:
        config: Profile used for the experiment.
        archive: Prepared input archive.
        output_dir: Completed experiment output directory.
        validation: Summary returned by :func:`validate_outputs`.
    """
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=output_dir.parents[2],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except Exception:
        commit = None
    hashes = {
        relative: file_sha256(output_dir / relative) for relative in REQUIRED_OUTPUTS
    }
    payload = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "archive": str(archive.resolve()),
        "archive_sha256": file_sha256(archive),
        "configuration": config.to_dict(),
        "validation": validation,
        "python": platform.python_version(),
        "git_commit": commit,
        "outputs": hashes,
    }
    atomic_write_json(output_dir / "manifest.json", payload)
    LOGGER.info("Wrote manifest with hashes for %d outputs", len(hashes))
