from __future__ import annotations

import ast
import json
import logging
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from changepoint_detection.baselines import (
    ControlChart,
    first_crossing,
    upcrossing_alarms,
)
from changepoint_detection.models import (
    GarchFit,
    ObservationModel,
    RegimePrior,
    SYNTHETIC_PRIOR,
    fit_garch11,
    prior_from_fits,
)
from changepoint_detection.data import (
    TrainingInputCacheMismatch,
    load_returns,
    load_training_inputs,
)
from changepoint_detection.particle import ParticleBOCPD, make_detector
from changepoint_detection.progress import map_jobs
from changepoint_detection.real_data import outside_event_counts
from changepoint_detection.real_data import (
    _chart_alarm_count_and_exposure,
    _rank_innovation_scores,
    combined_event_delay,
    event_awareness,
)
from changepoint_detection.simulation import (
    PRE_CHANGE,
    match_alarms,
    select_monitoring_example,
    simulate_garch_stream,
    summarise_detection_delays,
)
from changepoint_detection.storage import clean_changepoint_artifacts
from experiment_data import STUDY_START, TICKERS
from tests.archive_factory import deterministic_zip, frame_to_csv_bytes


class ObservationModelTests(unittest.TestCase):
    def test_gaussian_log_likelihood(self):
        model = ObservationModel("garch-gaussian")
        actual = model.log_likelihood(1.0, np.array([0.0]), np.array([4.0]))[0]
        expected = -0.5 * (np.log(2 * np.pi) + np.log(4.0) + 0.25)
        self.assertAlmostEqual(actual, expected)

    def test_standardised_student_t_has_unit_variance(self):
        rng = np.random.default_rng(7)
        nu = 6.0
        draws = rng.standard_t(nu, 500_000) * np.sqrt((nu - 2) / nu)
        self.assertAlmostEqual(float(np.var(draws)), 1.0, delta=0.02)

    def test_student_t_converges_to_gaussian(self):
        gaussian = ObservationModel("garch-gaussian")
        student = ObservationModel("garch-student-t", 1_000_000.0)
        mean = np.array([-0.2, 0.3])
        variance = np.array([0.7, 1.8])
        np.testing.assert_allclose(
            student.log_likelihood(0.4, mean, variance),
            gaussian.log_likelihood(0.4, mean, variance),
            rtol=2e-5,
            atol=2e-5,
        )

    def test_invalid_student_t_degrees_of_freedom(self):
        with self.assertRaises(ValueError):
            ObservationModel("garch-student-t", 2.0)


class PriorAndGarchTests(unittest.TestCase):
    def test_prior_samples_are_stationary(self):
        sample = SYNTHETIC_PRIOR.sample(np.random.default_rng(2), 10_000, True)
        persistence = sample["alpha"] + sample["beta"]
        self.assertTrue(np.all(sample["omega"] > 0))
        self.assertTrue(np.all(sample["beta"] >= 0))
        self.assertTrue(np.all(persistence < 1))

    def test_garch_qmle_recovers_a_long_run_scale(self):
        values, _, _ = simulate_garch_stream(
            np.random.default_rng(4), 3000, regimes=[(0, PRE_CHANGE)]
        )
        fit = fit_garch11(values, starts=2, seed=9)
        self.assertTrue(fit.converged)
        self.assertAlmostEqual(fit.hbar, 1.0, delta=0.35)
        self.assertLess(fit.persistence, 1.0)

    def test_finite_interior_nonconverged_fit_can_inform_prior(self):
        fits = [
            GarchFit(
                mean=mean,
                omega=0.05,
                alpha=0.10,
                beta=0.85,
                hbar=1.0,
                persistence=0.95,
                negative_log_likelihood=100.0 + mean,
                converged=False,
                iterations=3000,
            )
            for mean in (-0.1, 0.1)
        ]
        prior = prior_from_fits(fits)
        self.assertAlmostEqual(prior.mu_mean, 0.0)


class TrainingInputCacheTests(unittest.TestCase):
    def test_missing_archive_hash_invalidates_cache(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "pinnacle_clc.zip"
            archive.write_bytes(b"replacement archive")
            output = root / "outputs"
            output.mkdir()
            (output / "priors.json").write_text(json.dumps({"priors": {}}))
            (output / "garch_fits.csv").write_text("ticker\n")
            with self.assertRaises(TrainingInputCacheMismatch):
                load_training_inputs(archive, output)


class StudyBoundaryTests(unittest.TestCase):
    def test_archive_loading_excludes_warmup_before_calculating_returns(self):
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "pinnacle.zip"
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
            members = {
                f"CLCDATA/{ticker}_RAD.CSV": frame_to_csv_bytes(frame)
                for ticker in TICKERS
            }
            deterministic_zip(archive, members)

            returns = load_returns(archive)

            expected_date = pd.Timestamp("1990-01-03")
            expected_return = 100.0 * np.log(121.0 / 110.0)
            for ticker, series in returns.items():
                self.assertEqual(series.index.tolist(), [expected_date], ticker)
                self.assertGreaterEqual(series.index.min(), pd.Timestamp(STUDY_START))
                self.assertAlmostEqual(float(series.iloc[0]), expected_return)


class ProgressLoggingTests(unittest.TestCase):
    def test_batch_progress_reports_completion_and_preserves_order(self):
        logger = logging.getLogger("tests.changepoint.progress")
        with self.assertLogs(logger, level="INFO") as captured:
            results = map_jobs(
                abs,
                [-3, -1, -2],
                workers=1,
                label="Test batch",
                logger=logger,
            )
        self.assertEqual(results, [3, 1, 2])
        messages = "\n".join(captured.output)
        self.assertIn("Test batch: started (3 items)", messages)
        self.assertIn("3/3 (100.0%)", messages)
        self.assertIn("elapsed=", messages)
        self.assertIn("ETA=", messages)

    def test_none_is_a_valid_batch_result(self):
        results = map_jobs(
            lambda _: None,
            [1],
            workers=1,
            label="None-valued batch",
            logger=logging.getLogger("tests.changepoint.progress.none"),
        )
        self.assertEqual(results, [None])


class ParticleFilterTests(unittest.TestCase):
    def test_first_observation_uses_long_run_variance_without_extra_forecast(self):
        seed = 17
        particles = 80
        proposal = SYNTHETIC_PRIOR.sample(
            np.random.default_rng(seed), (1, particles), True
        )
        model = ObservationModel("garch-gaussian")
        likelihood = model.log_likelihood(0.3, proposal["mu"], proposal["hbar"])
        weights = np.exp(likelihood - np.max(likelihood))
        weights /= weights.sum()
        expected = float(np.sum(weights * proposal["hbar"]))
        detector = ParticleBOCPD(
            model,
            SYNTHETIC_PRIOR,
            particles=particles,
            seed=seed,
            ess_fraction=1e-6,
        )
        step = detector.update(0.3)
        self.assertAlmostEqual(step.expected_variance, expected)
        self.assertEqual(step.cp_prob, 1.0)
        np.testing.assert_array_equal(detector.run_lengths, [0])

    def test_complete_reset_does_not_inherit_prechange_variance(self):
        prior = RegimePrior(
            mu_mean=0,
            mu_sd=0.1,
            log_hbar_mean=0,
            log_hbar_sd=0.1,
            alpha_low=0.05,
            alpha_high=0.1,
            persistence_low=0.8,
            persistence_high=0.9,
            hbar_cap=2.0,
        )
        detector = make_detector(
            "bocpd-garch", prior, particles=100, max_run_lengths=20, seed=3
        )
        detector.update(0.0)
        detector.variance[:] = 1e6
        detector.update(0.1)
        row_zero = int(np.flatnonzero(detector.run_lengths == 0)[0])
        self.assertTrue(np.all(detector.variance[row_zero] <= prior.hbar_cap))

    def test_probabilities_normalise_and_cap_is_exact(self):
        detector = make_detector(
            "bocpd-garch",
            SYNTHETIC_PRIOR,
            particles=30,
            max_run_lengths=5,
            prune_threshold=0,
            seed=5,
        )
        frame = detector.run(np.random.default_rng(5).normal(size=80))
        self.assertLessEqual(len(detector.run_lengths), 5)
        self.assertIn(0, detector.run_lengths)
        self.assertTrue(np.all(np.diff(detector.run_lengths) >= 0))
        self.assertAlmostEqual(float(np.exp(detector.log_joint).sum()), 1.0)
        self.assertLess(detector.max_pooling_error, 1e-10)
        self.assertTrue((frame["cp_prob"] <= frame["young_prob"] + 1e-12).all())

    def test_determinism_and_output_schema(self):
        values = pd.Series(np.random.default_rng(6).normal(size=100))
        settings = dict(particles=30, max_run_lengths=10, seed=11)
        first = make_detector("bocpd", SYNTHETIC_PRIOR, **settings).run(values)
        second = make_detector("bocpd", SYNTHETIC_PRIOR, **settings).run(values)
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(tuple(first.columns), ParticleBOCPD.OUTPUT_COLUMNS)


class BaselineTests(unittest.TestCase):
    def test_cusum_detects_mean_shift(self):
        rng = np.random.default_rng(8)
        values = np.r_[rng.normal(size=250), rng.normal(3.0, 1.0, 100)]
        chart = ControlChart("cusum-mean")
        result = chart.run(values, 8.0, reestimate_after_alarm=False)
        self.assertTrue(np.any((result.alarms >= 250) & (result.alarms < 270)))

    def test_variance_cusum_detects_variance_shift(self):
        rng = np.random.default_rng(9)
        values = np.r_[rng.normal(size=250), rng.normal(scale=5.0, size=100)]
        chart = ControlChart("cusum-variance")
        result = chart.run(values, 10.0, reestimate_after_alarm=True)
        hit = result.alarms[(result.alarms >= 250) & (result.alarms < 280)]
        self.assertTrue(len(hit))
        self.assertFalse(result.monitored[hit[0] + 1 : hit[0] + 51].any())

    def test_upcrossing_only_reports_transition(self):
        statistic = np.array([0.0, 0.2, 0.8, 0.9, 0.3, 0.7])
        np.testing.assert_array_equal(
            upcrossing_alarms(statistic, 0.5, start=0), [2, 5]
        )
        self.assertEqual(first_crossing(statistic, 0.5, start=0), 2)


class ExperimentMetricTests(unittest.TestCase):
    def test_alarm_matching_is_one_to_one_and_uses_half_open_window(self):
        matched, unmatched, delays = match_alarms(
            np.array([8, 10, 12, 31, 35, 90]),
            np.array([10, 30]),
            5,
        )
        np.testing.assert_array_equal(matched, [10, 31])
        np.testing.assert_array_equal(unmatched, [8, 12, 35, 90])
        np.testing.assert_array_equal(delays, [0, 1])

    def test_monitoring_example_uses_first_stream_with_two_metric_lead(self):
        def result(index, rows):
            return {
                "stream_index": index,
                "counts": {
                    name: {
                        "correct": correct,
                        "false": false,
                        "missed": missed,
                        "delays": [],
                    }
                    for name, (correct, false, missed) in rows.items()
                },
            }

        first = result(
            0,
            {
                "BOCPD": (8, 6, 7),
                "BOCPD-GARCH": (9, 16, 6),
            },
        )
        second = result(
            1,
            {
                "CUSUM": (6, 9, 12),
                "BOCPD": (12, 16, 6),
                "BOCPD-GARCH": (15, 11, 3),
            },
        )
        selected, criterion_met = select_monitoring_example([second, first])
        self.assertEqual(selected["stream_index"], 1)
        self.assertTrue(criterion_met)

    def test_gaussian_winner_still_selects_best_finite_student_t_nu(self):
        pooled = pd.Series(
            {
                "t(4)": -1.4,
                "t(6)": -1.2,
                "t(10)": -1.3,
                "Gaussian": -1.0,
            }
        )
        winner, selected_t, selected_nu, gaussian_preferred = (
            _rank_innovation_scores(pooled)
        )
        self.assertEqual(winner, "Gaussian")
        self.assertEqual(selected_t, "t(6)")
        self.assertEqual(selected_nu, 6.0)
        self.assertTrue(gaussian_preferred)

    def test_fifty_day_miss_is_distinct_from_end_of_followup_censoring(self):
        summary = summarise_detection_delays([10, 75, None, 49], 50)
        self.assertEqual(summary["miss_rate_50"], 0.5)
        self.assertEqual(summary["miss_rate_horizon"], 0.25)
        self.assertEqual(summary["mean_delay_detected"], (10 + 75 + 49) / 3)

    def test_off_event_rate_excludes_event_windows_from_exposure(self):
        alarms = np.array([20, 55, 75, 130])
        windows = [(50, 80, "event")]
        count, exposure = outside_event_counts(alarms, 150, windows, burn_in=10)
        self.assertEqual(count, 2)
        self.assertEqual(exposure, 110)

    def test_awareness_excludes_event_date_alarm_and_includes_threshold_equality(self):
        alarms = np.array([100])
        self.assertFalse(event_awareness(alarms, 100))
        self.assertTrue(
            event_awareness(
                alarms,
                100,
                statistic_on_event=0.25,
                threshold=0.25,
            )
        )
        self.assertTrue(event_awareness(np.array([37]), 100))
        self.assertFalse(event_awareness(np.array([36]), 100))

    def test_chart_calibration_uses_actual_monitored_exposure(self):
        chart = ControlChart("cusum-mean", burn_in=5)
        values = np.array(
            [
                0,
                1,
                0,
                -1,
                0,
                100,
                9,
                10,
                11,
                10,
                9,
                -100,
                -11,
                -10,
                -9,
                -10,
                -11,
                100,
            ],
            dtype=float,
        )
        alarms, exposure = _chart_alarm_count_and_exposure(chart, values, 1.0)
        self.assertGreaterEqual(alarms, 2)
        self.assertLess(exposure, len(values) - chart.burn_in)

    def test_combined_event_delay_covers_pre_event_and_posterior_only_branches(self):
        self.assertEqual(combined_event_delay(np.array([85]), 100), -15)
        self.assertEqual(combined_event_delay(np.array([149]), 100), 49)
        self.assertIsNone(combined_event_delay(np.array([84, 150]), 100))
        self.assertEqual(
            combined_event_delay(
                np.array([], dtype=int),
                100,
                statistic_on_event=0.25,
                threshold=0.25,
            ),
            0,
        )


class PresentationSeparationTests(unittest.TestCase):
    def test_package_does_not_import_the_plotting_library(self):
        package = Path(__file__).resolve().parents[1] / "changepoint_detection"
        forbidden = "matplot" + "lib"
        offenders = []
        for path in package.rglob("*.py"):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    names = [node.module or ""]
                else:
                    continue
                if any(name == forbidden or name.startswith(forbidden + ".") for name in names):
                    offenders.append(str(path.relative_to(package)))
        self.assertEqual(offenders, [])


class CleanupTests(unittest.TestCase):
    def test_changepoint_cleanup_is_scoped_to_its_generated_trees(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            build_file = root / "build" / "changepoint" / "full" / "state.bin"
            output_file = root / "outputs" / "changepoint" / "full" / "result.csv"
            retained_build = root / "build" / "audit" / "report.json"
            retained_archive = root / "pinnacle.zip"
            for path in (build_file, output_file, retained_build):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("generated\n")
            retained_archive.write_text("archive\n")

            build, output = clean_changepoint_artifacts(root)

            self.assertEqual(build, root / "build" / "changepoint")
            self.assertEqual(output, root / "outputs" / "changepoint")
            self.assertFalse(build.exists())
            self.assertFalse(output.exists())
            self.assertTrue(retained_build.is_file())
            self.assertTrue(retained_archive.is_file())


if __name__ == "__main__":
    unittest.main()
