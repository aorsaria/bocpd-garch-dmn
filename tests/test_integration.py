"""Bounded integration tests for the dissertation experiments."""

from __future__ import annotations

import ast
from dataclasses import replace
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
import pandas as pd

from changepoint_detection.config import ExperimentConfig
from changepoint_detection.data import (
    build_training_inputs,
    load_training_inputs,
    save_training_inputs,
)
from changepoint_detection.particle import make_detector
from changepoint_detection.simulation import SYNTHETIC_ALGORITHMS, run_simulation
from deep_momentum.config import MODEL_BY_NAME, load_profile
from deep_momentum.data import build_base_features, load_closes
from deep_momentum.gating import (
    confidence_capital,
    ensemble_stats,
    hysteresis_mask,
    portfolio_from_capital,
    within_contract_rank,
)
from deep_momentum.tensors import build_window_tensors
from deep_momentum.training import _train_once
from deep_momentum.utils import atomic_npz
from tests.archive_factory import deterministic_zip, frame_to_csv_bytes


def _price_frame(start: str, end: str | None, periods: int | None, seed: int) -> pd.DataFrame:
    """Create a deterministic positive OHLC fixture for a prepared archive.

    Args:
        start: First business date.
        end: Last business date, or ``None`` when ``periods`` is supplied.
        periods: Number of business dates, or ``None`` when ``end`` is supplied.
        seed: Seed for simulated log-price increments.

    Returns:
        Seven-field CLC-style data frame indexed by date.
    """
    index = pd.bdate_range(start, end=end, periods=periods)
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.00015, 0.008, len(index))))
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.001,
            "low": close * 0.999,
            "close": close,
            "vol": np.full(len(index), 1_000),
            "oi": np.full(len(index), 500),
        },
        index=index.rename("date"),
    )


class BocpdIntegrationTests(unittest.TestCase):
    def test_synthetic_experiment_writes_every_reported_detector_summary(self):
        config = ExperimentConfig(
            name="integration",
            full_scale=False,
            seed=123,
            particles=12,
            max_run_lengths=12,
            hazard=0.02,
            young_window=10,
            burn_in=50,
            detection_window=20,
            target_arl0=30,
            null_calibration_streams=2,
            null_validation_streams=3,
            null_horizon=120,
            single_change_trials=2,
            single_change_time=60,
            single_change_horizon=120,
            monitoring_streams=2,
            monitoring_horizon=150,
            minimum_regime_duration=30,
            real_calibration_observations=100,
            selection_observations=100,
            selection_seeds=1,
            garch_fit_starts=1,
            representative_tickers=("ZG", "ES", "TY", "JN"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = run_simulation(
                config, root / "outputs", root / "build", workers=1
            )
            arl = pd.read_csv(root / "outputs" / "simulation_arl0.csv")
            single = pd.read_csv(
                root / "outputs" / "simulation_single_change.csv"
            )
            continuous = pd.read_csv(
                root / "outputs" / "simulation_continuous.csv"
            )
            with np.load(
                root / "outputs" / "continuous_example.npz", allow_pickle=False
            ) as example:
                self.assertEqual(int(example["monitoring_streams"]), 2)
                self.assertEqual(len(example["values"]), config.monitoring_horizon)

        self.assertEqual(set(result["thresholds"]), set(SYNTHETIC_ALGORITHMS))
        self.assertEqual(set(arl["detector"]), set(SYNTHETIC_ALGORITHMS))
        self.assertEqual(len(single), 4 * len(SYNTHETIC_ALGORITHMS))
        self.assertEqual(len(continuous), len(SYNTHETIC_ALGORITHMS))

    def test_prepared_archive_to_garch_prior_and_bocpd_posterior(self):
        contracts = ("CC", "KC", "ES", "SP", "TY", "US", "AN", "JN")
        members = {
            f"CLCDATA/{ticker}_RAD.CSV": frame_to_csv_bytes(
                _price_frame("1990-01-01", None, 180, seed)
            )
            for seed, ticker in enumerate(contracts, start=1)
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "prepared.zip"
            deterministic_zip(archive, members)
            # Two contracts per class are sufficient to exercise empirical
            # class-prior construction without a proprietary data fixture.
            with mock.patch("changepoint_detection.data.TICKERS", contracts):
                inputs = build_training_inputs(
                    archive, starts=1, seed=5, min_observations=100
                )
                save_training_inputs(inputs, root / "outputs")
                cached = load_training_inputs(archive, root / "outputs")

            values = cached.returns["ES"].to_numpy() / cached.scales["ES"]
            detector = make_detector(
                "bocpd-garch-t",
                cached.priors["EQ"],
                nu=6.0,
                particles=16,
                max_run_lengths=16,
                seed=17,
            )
            posterior = detector.run(values)

        self.assertEqual(set(inputs.priors), {"CM", "EQ", "FI", "FX"})
        self.assertEqual(len(inputs.fits), len(contracts))
        self.assertEqual(tuple(posterior.columns), detector.OUTPUT_COLUMNS)
        self.assertTrue(np.isfinite(posterior.to_numpy()).all())
        self.assertTrue(posterior["cp_prob"].between(0.0, 1.0).all())


class DeepMomentumIntegrationTests(unittest.TestCase):
    def test_prepared_archive_to_lstm_training_and_uncertainty_gate(self):
        tickers = ("ES", "ZG")
        members = {
            f"CLCDATA/{ticker}_RAD.CSV": frame_to_csv_bytes(
                _price_frame("2010-01-01", "2020-12-31", None, seed)
            )
            for seed, ticker in enumerate(tickers, start=11)
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "prepared.zip"
            deterministic_zip(archive, members)
            closes = load_closes(archive, tickers)
            frames = {
                ticker: build_base_features(close)
                for ticker, close in closes.items()
            }
            profile = replace(
                load_profile("smoke"),
                windows=(2020,),
                test_span=1,
                tickers=tickers,
            )
            arrays = build_window_tensors(
                frames, MODEL_BY_NAME["LSTM"], 2020, profile
            )
            tensor_path = root / "base.npz"
            atomic_npz(tensor_path, **arrays)
            hyperparameters = {
                "hidden": 5,
                "dropout": 0.1,
                "batch_size": 256,
                "lr": 0.001,
                "clipnorm": 1.0,
            }
            predictions = []
            for seed in (101, 102):
                result, frame = _train_once(
                    tensor_path,
                    "LSTM",
                    hyperparameters,
                    seed,
                    max_epochs=1,
                    patience=1,
                )
                self.assertEqual(result["status"], "success")
                predictions.append(frame)

            position_tables = [
                frame.pivot(index="date", columns="ticker", values="position")
                .sort_index()
                for frame in predictions
            ]
            pd.testing.assert_index_equal(
                position_tables[0].index, position_tables[1].index
            )
            mean, _, kappa = ensemble_stats(
                np.stack([frame.to_numpy() for frame in position_tables])
            )
            mean_frame = pd.DataFrame(
                mean,
                index=position_tables[0].index,
                columns=position_tables[0].columns,
            )
            kappa_frame = pd.DataFrame(
                kappa, index=mean_frame.index, columns=mean_frame.columns
            )
            rank = within_contract_rank(kappa_frame, min_history=10)
            available = kappa_frame.notna()
            mask = hysteresis_mask(rank, on_level=0.5, off_level=0.25) & available
            capital = confidence_capital(mask, rank, available, m=2.0)
            target = predictions[0].pivot(
                index="date", columns="ticker", values="target"
            ).sort_index()
            leverage = predictions[0].pivot(
                index="date", columns="ticker", values="lev"
            ).sort_index()
            portfolio = portfolio_from_capital(
                mean_frame,
                capital,
                mask,
                target,
                leverage,
                cost_bps=2.0,
            )

        self.assertEqual(arrays["x_train"].shape[-2:], (63, 8))
        self.assertEqual(len(predictions[0]), len(tickers) * len(portfolio))
        self.assertGreater(int((available & ~mask).sum().sum()), 0)
        self.assertTrue((capital.fillna(0.0).sum(axis=1) <= 1.0 + 1e-12).all())
        self.assertTrue(
            np.isfinite(portfolio[["gross", "net", "turnover"]].to_numpy()).all()
        )


class PackageContractTests(unittest.TestCase):
    def test_production_callables_have_docstrings_and_public_parameter_sections(self):
        root = Path(__file__).resolve().parents[1]
        sources = [root / "experiment_data.py"]
        sources.extend((root / "changepoint_detection").glob("*.py"))
        sources.extend((root / "deep_momentum").glob("*.py"))
        missing: list[str] = []
        missing_parameters: list[str] = []
        for path in sorted(sources):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                    continue
                if ast.get_docstring(node) is None:
                    missing.append(f"{path.relative_to(root)}:{node.lineno}")
            for node in tree.body:
                if not isinstance(node, ast.FunctionDef) or node.name.startswith("_"):
                    continue
                parameters = [
                    argument.arg
                    for argument in (
                        node.args.posonlyargs + node.args.args + node.args.kwonlyargs
                    )
                    if argument.arg not in {"self", "cls"}
                ]
                if parameters and "Args:" not in (ast.get_docstring(node) or ""):
                    missing_parameters.append(
                        f"{path.relative_to(root)}:{node.lineno}:{node.name}"
                    )
        self.assertEqual(missing, [])
        self.assertEqual(missing_parameters, [])

    def test_packages_do_not_depend_on_omitted_pipelines(self):
        root = Path(__file__).resolve().parents[1]
        forbidden = {"pinnacle_data", "classical_momentum", "appendix_data"}
        offenders: list[str] = []
        for package in (root / "changepoint_detection", root / "deep_momentum"):
            for path in package.glob("*.py"):
                tree = ast.parse(path.read_text(), filename=str(path))
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        names = [alias.name for alias in node.names]
                    elif isinstance(node, ast.ImportFrom):
                        names = [node.module or ""]
                    else:
                        continue
                    if any(name.split(".", 1)[0] in forbidden for name in names):
                        offenders.append(str(path.relative_to(root)))
        self.assertEqual(offenders, [])


if __name__ == "__main__":
    unittest.main()
