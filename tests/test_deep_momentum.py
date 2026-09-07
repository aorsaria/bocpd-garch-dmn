from __future__ import annotations

import ast
from dataclasses import replace
import json
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

os.environ.setdefault("KERAS_BACKEND", "torch")

import numpy as np
import pandas as pd

from deep_momentum.bootstrap import (
    BOOTSTRAP_COLUMNS,
    _circular_indices,
    paired_circular_bootstrap,
)
from deep_momentum.config import (
    BASE_FEATURES,
    HISTORY_START,
    MODEL_BY_NAME,
    MODEL_NAMES,
    PRIMARY_WINDOWS,
    SEQUENCE_LENGTH,
    STUDY_START,
    load_profile,
)
from deep_momentum.data import build_base_features, load_closes, study_closes
from deep_momentum.detectors import prepare_window_detector_features
from deep_momentum.features import _base_job, prepare_base_features
from deep_momentum.gating import (
    confidence_capital,
    conditional_contract_sharpe,
    divisor,
    ensemble_stats,
    hysteresis_mask,
    portfolio_from_capital,
    portfolio_from_weights,
    within_contract_rank,
)
from deep_momentum.reporting import (
    REQUIRED_OUTPUTS,
    _portfolio,
    performance_metrics,
    report_ensemble_size,
)
from deep_momentum.tensors import _blocks, build_window_tensors
from deep_momentum.training import draw_hyperparameters, replicate_seed, trial_seed
from deep_momentum.utils import (
    assert_cache_fingerprint,
    atomic_json,
    object_sha256,
)
from experiment_data import CSV_COLUMNS, UNIVERSE
from tests.archive_factory import deterministic_zip, frame_to_csv_bytes


class RegistryTests(unittest.TestCase):
    def test_requested_roster_and_feature_columns(self):
        self.assertEqual(len(MODEL_NAMES), 9)
        self.assertEqual(MODEL_BY_NAME["LSTM"].feature_columns, BASE_FEATURES)
        self.assertEqual(len(MODEL_BY_NAME["LSTM-BOCPD-full"].feature_columns), 14)
        self.assertEqual(MODEL_BY_NAME["LSTM@2bp"].cost_bps, 2.0)
        self.assertNotIn("LSTM-CPD-21", MODEL_NAMES)
        self.assertNotIn("LSTM-CPD-63", MODEL_NAMES)

    def test_full_profile_protocol(self):
        profile = load_profile("full")
        self.assertEqual(profile.trials, 50)
        self.assertEqual(profile.replicates, 20)
        self.assertEqual(profile.particles, 500)
        self.assertEqual(profile.max_run_lengths, 250)
        self.assertEqual(profile.bootstrap_samples, 2000)
        self.assertEqual(profile.history_start, HISTORY_START)
        self.assertEqual(profile.data_start, STUDY_START)
        self.assertEqual(profile.windows, PRIMARY_WINDOWS)
        self.assertNotIn("history_start", profile.to_dict())

    def test_smoke_profile_has_a_complete_bounded_structure(self):
        profile = load_profile("smoke")
        self.assertEqual(len(MODEL_NAMES) * len(profile.windows) * profile.trials, 18)
        self.assertEqual(
            len(MODEL_NAMES) * len(profile.windows) * profile.replicates, 18
        )
        self.assertEqual(report_ensemble_size(profile), 2)
        self.assertEqual(profile.bootstrap_samples, 50)
        self.assertEqual(profile.history_start, HISTORY_START)


class BoundaryAndFeatureTests(unittest.TestCase):
    @staticmethod
    def _close(start: str, end: str, seed: int = 1) -> pd.Series:
        index = pd.bdate_range(start, end)
        rng = np.random.default_rng(seed)
        return pd.Series(
            100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, len(index)))),
            index=index,
            name="ES",
        )

    def test_target_uses_next_return_and_current_volatility(self):
        close = self._close("2010-01-01", "2014-12-31")
        frame = build_base_features(close)
        valid = frame.dropna(subset=list(BASE_FEATURES) + ["target", "lev"])
        date = valid.index[10]
        location = frame.index.get_loc(date)
        self.assertAlmostEqual(
            frame.loc[date, "target"],
            frame["lev"].iloc[location] * frame["daily_return"].iloc[location + 1],
        )
        self.assertEqual(frame.loc[date, "target_date"], frame.index[location + 1])

    def test_unified_history_warms_features_but_is_not_stored(self):
        close = self._close("1987-01-01", "1992-12-31")
        with tempfile.TemporaryDirectory() as temporary:
            warmed_path = Path(temporary) / "warmed.parquet"
            bare_path = Path(temporary) / "bare.parquet"
            _base_job(
                ("ES", close, STUDY_START, HISTORY_START, str(warmed_path))
            )
            _base_job(("ES", close, STUDY_START, STUDY_START, str(bare_path)))
            warmed = pd.read_parquet(warmed_path).set_index("date")
            bare = pd.read_parquet(bare_path).set_index("date")
        self.assertGreaterEqual(warmed.index.min(), pd.Timestamp(STUDY_START))
        self.assertEqual(warmed.index.tolist(), bare.index.tolist())
        self.assertGreater(
            warmed[list(BASE_FEATURES)].notna().all(axis=1).sum(),
            bare[list(BASE_FEATURES)].notna().all(axis=1).sum(),
        )

    def test_archive_chokepoint_and_study_view_are_distinct(self):
        dates = pd.DatetimeIndex(
            ["1989-12-29", "1990-01-02", "1990-01-03"], name="date"
        )
        frame = pd.DataFrame(
            {
                "open": [100.0, 110.0, 121.0],
                "high": [100.0, 110.0, 121.0],
                "low": [100.0, 110.0, 121.0],
                "close": [100.0, 110.0, 121.0],
                "vol": [1, 1, 1],
                "oi": [1, 1, 1],
            },
            index=dates,
        )
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "pinnacle.zip"
            deterministic_zip(
                archive, {"CLCDATA/ES_RAD.CSV": frame_to_csv_bytes(frame)}
            )
            loaded = load_closes(archive, ("ES",))
        self.assertLess(loaded["ES"].index.min(), pd.Timestamp(STUDY_START))
        bounded = study_closes(loaded)
        self.assertEqual(bounded["ES"].index.min(), pd.Timestamp("1990-01-02"))
        returns = np.log(bounded["ES"]).diff().dropna()
        self.assertEqual(returns.index.tolist(), [pd.Timestamp("1990-01-03")])

    def test_tensor_builder_rejects_pre_study_rows(self):
        required = list(MODEL_BY_NAME["LSTM"].feature_columns) + [
            "target",
            "target_date",
            "lev",
        ]
        frame = pd.DataFrame(
            {column: [0.0] for column in required},
            index=pd.DatetimeIndex(["1989-12-29"]),
        )
        with self.assertRaisesRegex(ValueError, "pre-study"):
            build_window_tensors(
                {"ES": frame}, MODEL_BY_NAME["LSTM"], 2020, load_profile("smoke")
            )

    def test_detector_inputs_are_bounded_by_study_policy(self):
        profile = load_profile("smoke")
        dates = pd.bdate_range("1989-01-02", "2020-12-31")
        returns = {"ES": pd.Series(0.1, index=dates)}
        calibration = {
            "fingerprint": "calibration",
            "eligible_tickers": ["ES"],
            "training_end": "2019-12-31",
            "scales": {"ES": 1.0},
            "priors": {UNIVERSE["ES"][0]: {}},
        }
        captured: list[tuple[object, list[tuple[object, ...]]]] = []

        def fake_map(function, jobs, workers, **kwargs):
            captured.append((function, jobs))
            return []

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "pinnacle.zip"
            archive.write_bytes(b"archive")
            with mock.patch("deep_momentum.detectors.map_jobs", fake_map):
                with self.assertRaises(FileNotFoundError):
                    # The missing outputs stop the stage after job construction.
                    prepare_window_detector_features(
                        archive,
                        profile,
                        root,
                        2020,
                        returns,
                        calibration,
                        6.0,
                        workers=1,
                    )
        bocpd_jobs = captured[0][1]
        self.assertGreaterEqual(bocpd_jobs[0][1].index.min(), pd.Timestamp("2015-01-01"))


class CacheAndTensorTests(unittest.TestCase):
    def test_feature_fingerprint_contains_unified_history(self):
        profile = load_profile("smoke")
        close = BoundaryAndFeatureTests._close("1987-01-01", "2020-12-31")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "archive.zip"
            archive.touch()
            with (
                mock.patch(
                    "deep_momentum.features.selected_tickers", return_value=("ES",)
                ),
                mock.patch(
                    "deep_momentum.features.file_sha256", return_value="archive-a"
                ),
                mock.patch(
                    "deep_momentum.features.load_closes", return_value={"ES": close}
                ),
            ):
                prepare_base_features(archive, profile, root, workers=1)
            metadata = json.loads(
                (root / "features" / "base" / "metadata.json").read_text()
            )
        expected = object_sha256(
            {
                "stage": "base-features-v2",
                "archive": "archive-a",
                "tickers": ("ES",),
                "data_start": profile.data_start,
                "history": {"start": HISTORY_START, "archive": "archive-a"},
            }
        )
        self.assertEqual(metadata["fingerprint"], expected)

    def test_stale_cache_is_a_hard_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "metadata.json"
            atomic_json(path, {"fingerprint": "old"})
            with self.assertRaisesRegex(RuntimeError, "stale cache"):
                assert_cache_fingerprint(path, "new")

    def test_blocks_mask_padding_and_leverage(self):
        x = np.ones((65, 2))
        target = np.arange(65, dtype=float)
        leverage = np.full(65, 3.0)
        xb, yb = _blocks(x, target, leverage)
        self.assertEqual(xb.shape, (2, SEQUENCE_LENGTH, 2))
        self.assertEqual(int(yb[..., 1].sum()), 65)
        self.assertEqual(float(yb[1, 2, 1]), 0.0)


class ReproducibilityTests(unittest.TestCase):
    def test_hyperparameter_draw_and_seeds_are_deterministic(self):
        profile = load_profile("smoke")
        seed = trial_seed(profile, 2020, "LSTM", 0)
        self.assertEqual(draw_hyperparameters(seed), draw_hyperparameters(seed))
        self.assertNotEqual(seed, replicate_seed(profile, 2020, "LSTM", 0))

    def test_full_training_seeds_are_unique(self):
        profile = load_profile("full")
        values = [
            trial_seed(profile, year, model, trial)
            for year in profile.windows
            for model in MODEL_NAMES
            for trial in range(profile.trials)
        ] + [
            replicate_seed(profile, year, model, replicate)
            for year in profile.windows
            for model in MODEL_NAMES
            for replicate in range(profile.replicates)
        ]
        self.assertEqual(len(values), len(set(values)))

    def test_top_five_is_the_default_reporting_ensemble(self):
        profile = load_profile("full")
        self.assertEqual(report_ensemble_size(profile), 5)
        self.assertEqual(report_ensemble_size(profile, 5), 5)

    def test_profit_loss_ratio_excludes_zero_days(self):
        metrics = performance_metrics(pd.Series([0.01, -0.005, 0.0, -0.015]))
        self.assertAlmostEqual(metrics["Ave. P / Ave. L"], 1.0)


class GatingTests(unittest.TestCase):
    def test_hysteresis_on_off_thresholds(self):
        index = pd.bdate_range("2020-01-01", periods=7)
        score = pd.DataFrame(
            {"A": [0.6, 0.3, 0.15, 0.09, 0.3, 0.6, np.nan]}, index=index
        )
        mask = hysteresis_mask(score, 0.5, 0.25)
        self.assertEqual(mask["A"].tolist(), [True, True, False, False, False, True, True])

    def test_capital_divisor(self):
        self.assertEqual(divisor([8], [3], 1.0)[0], 8.0)
        self.assertEqual(divisor([8], [3], 2.0)[0], 4.0)
        self.assertEqual(divisor([8], [3], np.inf)[0], 3.0)

    def test_confidence_cap_and_m_one_equivalence(self):
        index = pd.bdate_range("2020-01-01", periods=5)
        columns = list("ABCD")
        rank = pd.DataFrame(
            np.tile([0.2, 0.4, 0.8, 0.9], (5, 1)), index=index, columns=columns
        )
        available = pd.DataFrame(True, index=index, columns=columns)
        mask = rank >= 0.4
        mean = pd.DataFrame(0.5, index=index, columns=columns)
        target = pd.DataFrame(0.01, index=index, columns=columns)
        lev = pd.DataFrame(1.0, index=index, columns=columns)
        capital = confidence_capital(mask, rank, available, 1.0)
        weighted = portfolio_from_capital(mean, capital, mask, target, lev)
        reference = portfolio_from_weights(
            mean, mask.astype(float), target, lev, 1.0, binary=True
        )
        pd.testing.assert_series_equal(weighted["gross"], reference["gross"])
        pd.testing.assert_series_equal(weighted["turnover"], reference["turnover"])
        uncapped = confidence_capital(mask, rank, available, math.inf)
        self.assertTrue((uncapped.sum(axis=1) <= 1.0 + 1e-12).all())

    def test_conditional_sharpe_floor_and_weighting(self):
        index = pd.bdate_range("2020-01-01", periods=200)
        mean = pd.DataFrame({"A": 1.0, "B": 1.0}, index=index)
        target = pd.DataFrame(
            {
                "A": np.tile([0.02, -0.01], 100),
                "B": np.tile([0.05, -0.04], 100),
            },
            index=index,
        )
        mask = pd.DataFrame({"A": True, "B": False}, index=index)
        mask.loc[index[:30], "B"] = True
        result = conditional_contract_sharpe(mean, mask, target, min_days=60)
        self.assertEqual(result["conditional_contracts"], 1)
        self.assertEqual(result["conditional_excluded"], 1)
        expected = target["A"].mean() / target["A"].std(ddof=1) * np.sqrt(252.0)
        self.assertAlmostEqual(result["conditional_sharpe"], expected)

    def test_within_contract_rank_is_strictly_causal(self):
        rng = np.random.default_rng(11)
        index = pd.bdate_range("2019-01-01", periods=400)
        kappa = pd.DataFrame(rng.normal(size=(400, 2)), index=index)
        full = within_contract_rank(kappa, min_history=50)
        prefix = within_contract_rank(kappa.iloc[:200], min_history=50)
        pd.testing.assert_frame_equal(full.iloc[:200], prefix)


class BootstrapTests(unittest.TestCase):
    def test_circular_blocks_wrap(self):
        rng = mock.Mock()
        rng.integers.return_value = np.array([3, 3])
        indices = _circular_indices(5, 4, rng)
        np.testing.assert_array_equal(indices, [3, 4, 0, 1, 3])

    def test_pairing_and_seed_stability(self):
        profile = replace(
            load_profile("smoke"), windows=(2020,), test_span=1, bootstrap_samples=25
        )
        index = pd.bdate_range("2020-01-01", periods=120)
        rng = np.random.default_rng(4)
        ungated = pd.Series(rng.normal(0.0002, 0.01, len(index)), index=index)
        gated = ungated.copy()
        first = paired_circular_bootstrap(gated, ungated, profile, samples=25)
        second = paired_circular_bootstrap(gated, ungated, profile, samples=25)
        self.assertEqual(first[0], 0.0)
        self.assertEqual(first[1], 1.0)
        np.testing.assert_array_equal(first[2], np.zeros(25))
        np.testing.assert_array_equal(first[2], second[2])


class SeparationAndStructureTests(unittest.TestCase):
    def test_pipeline_packages_do_not_import_matplotlib(self):
        root = Path(__file__).resolve().parents[1]
        packages = (root / "deep_momentum", root / "changepoint_detection")
        offenders = []
        for package in packages:
            for path in package.rglob("*.py"):
                tree = ast.parse(path.read_text())
                for node in ast.walk(tree):
                    names = []
                    if isinstance(node, ast.Import):
                        names = [alias.name for alias in node.names]
                    elif isinstance(node, ast.ImportFrom):
                        names = [node.module or ""]
                    if any(name == "matplotlib" or name.startswith("matplotlib.") for name in names):
                        offenders.append(str(path.relative_to(root)))
        self.assertEqual(offenders, [])

    def test_required_outputs_cover_smoke_gating_and_bootstrap(self):
        self.assertIn("gating/gating_summary.csv", REQUIRED_OUTPUTS)
        self.assertIn("gating/bootstrap.csv", REQUIRED_OUTPUTS)
        self.assertIn("gating/gate_by_class_windows.csv", REQUIRED_OUTPUTS)
        self.assertTrue(
            any(name.endswith("gating_daily_lstm_trial.parquet") for name in REQUIRED_OUTPUTS)
        )
        self.assertEqual(
            BOOTSTRAP_COLUMNS,
            (
                "kind",
                "policy",
                "cost_bps",
                "model",
                "delta",
                "p_value",
                "block_length",
                "B",
                "seed",
            ),
        )


@unittest.skipUnless(
    __import__("importlib").util.find_spec("keras")
    and __import__("importlib").util.find_spec("torch"),
    "Keras/PyTorch dependencies are not installed",
)
class NetworkTests(unittest.TestCase):
    def test_lstm_shape_and_bounded_output(self):
        from deep_momentum.network import build_lstm

        model = build_lstm(
            8,
            {
                "hidden": 5,
                "dropout": 0.1,
                "batch_size": 64,
                "lr": 1e-3,
                "clipnorm": 1.0,
            },
        )
        output = model.predict(np.ones((2, 63, 8), dtype="float32"), verbose=0)
        self.assertEqual(output.shape, (2, 63, 1))
        self.assertTrue(np.all(np.abs(output) <= 1.0))


if __name__ == "__main__":
    unittest.main()
